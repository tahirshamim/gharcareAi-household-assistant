"""HTTP-level tests. They run only when the web packages are installed (pip install -r requirements-app.txt)."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

HAVE_WEB = all(importlib.util.find_spec(m) for m in ("fastapi", "httpx", "multipart"))
DOCS = Path(__file__).parent.parent / "examples" / "sample_docs"


@unittest.skipUnless(HAVE_WEB, "fastapi/httpx/python-multipart not installed (pip install -r requirements-app.txt)")
class ApiTests(unittest.TestCase):
    TOKEN = None

    def client(self, token=None, max_upload=20 * 1024 * 1024):
        from fastapi.testclient import TestClient
        from gharcare.api.app import create_app
        from gharcare.api.service import AppService
        from gharcare.config import Settings
        from gharcare.llm import FakeLLMProvider
        from tests.test_app_service import TODAY, model
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        service = AppService.create(Settings(data_dir=tmp.name, embedder="hashing", vector_store="memory"), llm=FakeLLMProvider(responder=model),
                                    clock=lambda: TODAY, workflow_backend="local")
        self.addCleanup(service.close)
        return TestClient(create_app(service, user_id="ayesha", token=token, max_upload_bytes=max_upload))

    def test_health_index_static_and_security_headers(self):
        c = self.client()
        self.assertTrue(c.get("/api/health").json()["ok"])
        page = c.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("GharCare", page.text)
        self.assertIn("default-src 'self'", page.headers["content-security-policy"])
        self.assertEqual(page.headers["x-content-type-options"], "nosniff")
        self.assertEqual(c.get("/static/app.js").status_code, 200)
        self.assertEqual(c.get("/static/style.css").status_code, 200)

    def test_full_flow_over_http(self):
        c = self.client()
        created = c.post("/api/appliances", json={"name": "Fridge", "brand": "Haier"})
        self.assertEqual(created.status_code, 201)
        fid = created.json()["id"]
        manual = DOCS / "xr200_sample_manual_SYNTHETIC.txt"
        up = c.post("/api/documents", files={"file": (manual.name, manual.read_bytes(), "text/plain")}, data={"doc_type": "manual", "appliance_id": fid})
        self.assertEqual((up.status_code, up.json()["indexed"]), (201, True))
        ask = c.post("/api/ask", json={"question": "How do I clear ice build-up in the freezer?", "appliance_id": fid}).json()
        self.assertEqual(ask["status"], "answered")
        doc_id = c.get("/api/documents").json()[0]["id"]
        download = c.get(f"/api/documents/{doc_id}/file")
        self.assertEqual(download.content, manual.read_bytes())
        self.assertIn("attachment", download.headers["content-disposition"])
        self.assertEqual(c.delete(f"/api/appliances/{fid}").status_code, 204)

    def test_errors_map_to_clean_statuses(self):
        c = self.client()
        self.assertEqual(c.put("/api/appliances/nope", json={"name": "x"}).status_code, 404)
        bad = c.post("/api/appliances", json={"name": "X", "purchase_date": "14/03/2024"})
        self.assertEqual(bad.status_code, 422)
        self.assertIn("not a valid date", bad.json()["error"])
        self.assertEqual(c.get("/api/drafts?status=nonsense").status_code, 422)
        self.assertEqual(c.get("/api/expenses?months=0").status_code, 422)
        self.assertEqual(c.post("/api/ask", json={"question": "  "}).status_code, 422)
        self.assertEqual(c.post("/api/documents", files={"file": ("x.jpg", b"\xff\xd8", "image/jpeg")}, data={"doc_type": "receipt"}).status_code, 422)

    def test_upload_size_limit(self):
        c = self.client(max_upload=1000)
        r = c.post("/api/documents", files={"file": ("big.txt", b"x" * 5000, "text/plain")}, data={"doc_type": "manual"})
        self.assertEqual(r.status_code, 413)

    def test_access_token(self):
        c = self.client(token="s3cret-token")
        self.assertEqual(c.get("/api/health").status_code, 200)                    # health stays open
        self.assertEqual(c.get("/api/appliances").status_code, 401)
        self.assertEqual(c.get("/api/appliances", headers={"Authorization": "Bearer wrong"}).status_code, 401)
        self.assertEqual(c.get("/api/appliances", headers={"Authorization": "Bearer s3cret-token"}).status_code, 200)


if __name__ == "__main__":
    unittest.main()
