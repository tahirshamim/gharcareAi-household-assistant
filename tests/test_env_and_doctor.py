import tempfile
import unittest
from pathlib import Path

import logging

from gharcare.doctor import diagnose, live_check
from gharcare.errors import LLMUnavailableError
from gharcare.llm import FakeLLMProvider
from gharcare.logging_utils import RedactingFilter, redact, secrets_from_env
from gharcare.env import load_env_file, parse_env_text


class ParseTests(unittest.TestCase):
    def test_syntax(self):
        text = ('# comment\n\nA=1\n  B = two words  \nexport C=3\nD="quoted # not a comment"\nE=\'single\'\n'
                'F=value # inline comment\nG=abc=def==\nbad key=1\nNOEQUALS\n1BAD=x\nH=\n')
        self.assertEqual(parse_env_text(text), {"A": "1", "B": "two words", "C": "3", "D": "quoted # not a comment",
                                                "E": "single", "F": "value", "G": "abc=def==", "H": ""})

    def test_bom_and_windows_line_endings(self):
        self.assertEqual(parse_env_text("\ufeffA=1\r\nB=2\r\n"), {"A": "1", "B": "2"})


class LoadTests(unittest.TestCase):
    def write(self, text, encoding="utf-8"):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / ".env"
        path.write_text(text, encoding=encoding)
        return path

    def test_loads_values_returns_only_names_and_skips_empty(self):
        env = {}
        names = load_env_file(self.write("GEMINI_API_KEY=secret-123\nGHARCARE_LLM_MODEL=\nOTHER=x\n"), environ=env)
        self.assertEqual(sorted(names), ["GEMINI_API_KEY", "OTHER"])
        self.assertEqual(env["GEMINI_API_KEY"], "secret-123")
        self.assertNotIn("GHARCARE_LLM_MODEL", env)                       # blank placeholder == not set
        self.assertNotIn("secret-123", names)

    def test_real_environment_wins_unless_override(self):
        path = self.write("K=from-file\n")
        env = {"K": "from-env"}
        self.assertEqual(load_env_file(path, environ=env), [])
        self.assertEqual(env["K"], "from-env")
        load_env_file(path, override=True, environ=env)
        self.assertEqual(env["K"], "from-file")

    def test_missing_file_is_fine_and_bom_files_work(self):
        self.assertEqual(load_env_file("/no/such/.env", environ={}), [])
        env = {}
        load_env_file(self.write("K=v\n", encoding="utf-8-sig"), environ=env)
        self.assertEqual(env, {"K": "v"})


class DoctorTests(unittest.TestCase):
    def tmpdir(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def by_name(self, checks):
        return {c.name: c for c in checks}

    def test_missing_key_and_model_are_reported_as_required_failures(self):
        c = self.by_name(diagnose({"GHARCARE_LLM_PROVIDER": "gemini"}, self.tmpdir(), find_spec=lambda m: None))
        self.assertFalse(c["GEMINI_API_KEY"].ok)
        self.assertFalse(c["GHARCARE_LLM_MODEL"].ok)
        self.assertTrue(c["GEMINI_API_KEY"].required)

    def test_configured_gemini_passes_and_never_echoes_the_key(self):
        env = {"GHARCARE_LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "AIza-super-secret", "GHARCARE_LLM_MODEL": "some-model"}
        checks = diagnose(env, self.tmpdir(), find_spec=lambda m: object())
        self.assertTrue(all(c.ok for c in checks if c.required))
        self.assertNotIn("AIza-super-secret", " ".join(c.detail for c in checks))

    def test_package_requirements_follow_the_chosen_embedder_and_store(self):
        none_installed = lambda m: None
        c = self.by_name(diagnose({"GHARCARE_EMBEDDER": "hashing", "GHARCARE_VECTOR_STORE": "memory"}, self.tmpdir(), none_installed))
        self.assertFalse(c["chromadb"].required)
        c = self.by_name(diagnose({"GHARCARE_EMBEDDER": "sentence-transformers", "GHARCARE_VECTOR_STORE": "chroma"}, self.tmpdir(), none_installed))
        self.assertTrue(c["chromadb"].required and c["sentence-transformers"].required)
        self.assertIn("pip install chromadb", c["chromadb"].detail)

    def test_notepad_dot_txt_trap_is_detected(self):
        d = self.tmpdir()
        (d / ".env.txt").write_text("A=1")
        c = self.by_name(diagnose({}, d, find_spec=lambda m: None))
        self.assertIn("Rename-Item", c[".env file"].detail)
        (d / ".env").write_text("A=1")
        self.assertTrue(self.by_name(diagnose({}, d, find_spec=lambda m: None))[".env file"].ok)


class LiveCheckAndRedactionTests(unittest.TestCase):
    KEY = "AIza-super-secret-key-123"

    def test_live_check_success_and_failure(self):
        ok = live_check(FakeLLMProvider(["OK"]))
        self.assertTrue(ok.ok)
        bad = live_check(FakeLLMProvider([LLMUnavailableError("404 model not found")]))
        self.assertFalse(bad.ok)
        self.assertIn("LLMUnavailableError: 404 model not found", bad.detail)

    def test_the_key_is_redacted_from_error_details(self):
        bad = live_check(FakeLLMProvider([LLMUnavailableError(f"request to ...?key={self.KEY} failed")]), (self.KEY,))
        self.assertNotIn(self.KEY, bad.detail)
        self.assertIn("***", bad.detail)

    def test_redaction_helpers(self):
        self.assertEqual(redact(f"a {self.KEY} b", [self.KEY]), "a *** b")
        self.assertEqual(redact("short", ["abc"]), "short")                  # ignores trivially short 'secrets'
        self.assertEqual(secrets_from_env({"GEMINI_API_KEY": self.KEY, "OTHER": "x"}), [self.KEY])

    def test_log_filter_scrubs_records(self):
        record = logging.LogRecord("x", logging.WARNING, __file__, 1, "failed with key %s", (self.KEY,), None)
        RedactingFilter([self.KEY]).filter(record)
        self.assertNotIn(self.KEY, record.getMessage())
        self.assertIn("***", record.getMessage())


if __name__ == "__main__":
    unittest.main()


class CodeLocationTests(unittest.TestCase):
    def test_warns_when_another_copy_is_loaded(self):
        import tempfile
        from gharcare.doctor import code_location_check
        with tempfile.TemporaryDirectory() as d:
            here = Path(d) / "new"
            (here / "src" / "gharcare").mkdir(parents=True)
            old = Path(d) / "old" / "src" / "gharcare"
            old.mkdir(parents=True)
            bad = code_location_check(here, old)
            self.assertFalse(bad.ok)
            self.assertIn("pip install -e .", bad.detail)
            self.assertTrue(code_location_check(here, here / "src" / "gharcare").ok)
            self.assertTrue(code_location_check(Path(d), old).ok)       # not inside a project folder: nothing to compare
