"""Real-browser tests of the web page (headless Chrome/Chromium via Playwright), against a real uvicorn server
with a scripted fake model. They caught two bugs the static checks missed: inline styles silently blocked by
the Content Security Policy, and a favicon 404.

Run them with:  pip install playwright   and either  python -m playwright install chromium   or point
GHARCARE_BROWSER at an installed Chrome/Chromium executable. They skip when neither is available."""
import importlib.util
import os
import shutil
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

DOCS = Path(__file__).parent.parent / "examples" / "sample_docs"
HAVE_WEB = all(importlib.util.find_spec(m) for m in ("fastapi", "uvicorn", "multipart"))
HAVE_PLAYWRIGHT = importlib.util.find_spec("playwright") is not None


def _browser_executable():
    explicit = os.environ.get("GHARCARE_BROWSER")
    if explicit:
        return explicit
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    return None          # fall back to Playwright's own downloaded browser, if any


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipUnless(HAVE_WEB and HAVE_PLAYWRIGHT, "needs fastapi/uvicorn/python-multipart and playwright")
class BrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import uvicorn
        from playwright.sync_api import sync_playwright
        from gharcare.api.app import create_app
        from gharcare.api.service import AppService
        from gharcare.config import Settings
        from gharcare.llm import FakeLLMProvider
        from tests.test_app_service import TODAY, model

        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.service = AppService.create(Settings(data_dir=cls._tmp.name, embedder="hashing", vector_store="memory"),
                                        llm=FakeLLMProvider(responder=model), clock=lambda: TODAY, workflow_backend="local")
        port = _free_port()
        cls.base = f"http://127.0.0.1:{port}"
        cls.server = uvicorn.Server(uvicorn.Config(create_app(cls.service, user_id="ayesha"), host="127.0.0.1", port=port, log_level="warning"))
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        deadline = time.time() + 15
        while not cls.server.started and time.time() < deadline:
            time.sleep(0.05)
        cls._pw = sync_playwright().start()
        try:
            exe = _browser_executable()
            cls.browser = cls._pw.chromium.launch(executable_path=exe, args=["--no-sandbox"]) if exe else cls._pw.chromium.launch()
        except Exception as exc:             # no usable browser on this machine
            cls._pw.stop()
            cls.server.should_exit = True
            raise unittest.SkipTest(f"no Chromium available ({exc.__class__.__name__}); set GHARCARE_BROWSER or run: python -m playwright install chromium")

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls._pw.stop()
        cls.server.should_exit = True
        cls.thread.join(timeout=10)
        cls.service.close()
        cls._tmp.cleanup()

    def setUp(self):
        self.page = self.browser.new_page()
        self.problems = []
        self.page.on("console", lambda m: self.problems.append(f"console {m.type}: {m.text}") if m.type in ("error", "warning") else None)
        self.page.on("pageerror", lambda e: self.problems.append(f"page error: {e}"))
        self.page.on("response", lambda r: self.problems.append(f"HTTP {r.status} {r.url}") if r.status >= 400 else None)
        self.page.on("dialog", lambda d: (self.problems.append(f"dialog: {d.message}"), d.dismiss()))
        self.page.goto(self.base)
        self.page.wait_for_selector("#tabs button")

    def tearDown(self):
        self.page.close()

    def tab(self, label):
        self.page.click(f"#tabs button:text-is('{label}')")
        self.page.wait_for_timeout(300)

    def assert_clean(self):
        self.assertEqual(self.problems, [], "the browser reported problems")
        import re
        text = self.page.inner_text("#view")
        self.assertIsNone(re.search(r"\b(null|undefined|NaN|\[object Object\])\b", text), "a missing value was printed on the page")

    def add_fridge(self, name):
        self.tab("Appliances")
        self.page.get_by_label("Name", exact=True).fill(name)
        self.page.get_by_label("Brand", exact=True).fill("Haier")
        self.page.click("#view button:text-is('Add appliance')")
        self.page.wait_for_selector(f"#view td:text-is('Haier {name}')")

    def upload(self, path, appliance_label):
        self.tab("Documents")
        self.page.set_input_files("#view input[type=file]", str(path))
        self.page.select_option("#view select >> nth=1", label=appliance_label)
        self.page.click("#view button:text-is('Upload')")
        self.page.wait_for_selector("#toast:has-text('passages are now searchable')")

    def ask(self, question):
        before = self.page.locator(".msg.bot").count()
        self.page.fill("#view input.input-grow", question)
        self.page.click("#view button:text-is('Ask')")
        reply = self.page.locator(".msg.bot").nth(before)      # (wait_for_function would need eval, which the CSP blocks)
        reply.wait_for()
        return reply

    def test_every_tab_renders_without_errors(self):
        for label in ["Home", "Ask", "Appliances", "Documents", "Review", "Warranties", "Maintenance", "Repairs", "Expenses", "Report"]:
            with self.subTest(tab=label):
                self.tab(label)
                self.assertTrue(self.page.inner_text("#view").strip())
                self.assert_clean()

    def test_inputs_get_their_css_sizes(self):
        # inline styles were blocked by the CSP before; the class-based sizes must actually apply
        width = self.page.eval_on_selector("#view input.input-grow", "el => getComputedStyle(el).minWidth")
        self.assertEqual(width, "240px")
        self.assert_clean()

    def test_add_upload_ask_with_sources(self):
        self.add_fridge("Kitchen fridge")
        self.upload(DOCS / "xr200_sample_manual_SYNTHETIC.txt", "Haier Kitchen fridge")
        self.tab("Ask")
        self.page.select_option("#appliance-select", label="Haier Kitchen fridge")
        bot = self.ask("How do I clear ice build-up in the freezer?")
        self.assertIn("6 hours", bot.inner_text())
        self.assertIn("Sources", bot.locator(".sources").inner_text())
        self.assertEqual(bot.locator(".cite").first.inner_text(), "1")             # [1] became a citation marker
        self.assertIn("p.", bot.locator(".sources").inner_text())
        safety = self.ask("I can smell gas near the fridge")
        self.assertIn("safety", safety.get_attribute("class"))
        self.assert_clean()

    def test_home_lists_what_needs_attention(self):
        from datetime import date
        from gharcare.domain import Appliance, MaintenanceTask
        a = self.service.app.appliances.add(Appliance(user_id="ayesha", name="Split AC", brand="Gree"))
        self.service.app.maintenance.add(MaintenanceTask(user_id="ayesha", appliance_id=a.id, title="AC service", next_due_date=date(2026, 1, 1)))
        self.page.reload()
        self.page.wait_for_selector(".attention li")
        self.assertIn("AC service is overdue", self.page.inner_text(".attention"))
        self.page.click(".attention li.urgent button")
        self.page.wait_for_selector("#view h1:text-is('Maintenance')")
        self.assert_clean()

    def test_clarification_offers_appliance_buttons(self):
        self.add_fridge("Second fridge")
        self.add_fridge("Third fridge")
        self.tab("Ask")
        self.page.select_option("#appliance-select", label="All appliances")
        bot = self.ask("What does error code E4 mean?")
        self.assertGreaterEqual(bot.locator(".choices button").count(), 2)
        self.assert_clean()

    def test_document_text_is_never_run_as_html(self):
        self.add_fridge("Test fridge")
        evil = Path(self._tmp.name) / "evil.txt"
        evil.write_text("DEFROSTING\nLeave the doors open for about 6 hours. <img src=x onerror=alert(1)> <b>bold</b>\n", encoding="utf-8")
        self.upload(evil, "Haier Test fridge")
        self.tab("Ask")
        self.page.select_option("#appliance-select", label="Haier Test fridge")
        fake = self.service.assistant._llm                 # make the judge reject, so the RAW passages are displayed
        original = fake._responder
        fake._responder = lambda m: '{"supported": false}' if m[0].content.startswith("You are a strict fact checker") else "Leave the doors open [1]."
        try:
            bot = self.ask("How long should the doors stay open to defrost?")
        finally:
            fake._responder = original
        self.assertIn("<img src=x onerror=alert(1)>", bot.inner_text())   # shown literally, as text
        self.assertEqual(self.page.locator("#view img").count(), 0)
        self.assertEqual(self.page.locator("#view .msg b").count(), 0)
        self.assert_clean()                     # includes: no alert() dialog appeared


if __name__ == "__main__":
    unittest.main()
