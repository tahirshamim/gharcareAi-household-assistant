import unittest
from pathlib import Path

from gharcare.config import Settings
from gharcare.errors import (
    DomainValidationError, LLMRateLimitError, LLMResponseError, LLMUnavailableError,
)
from gharcare.llm import FakeLLMProvider, LLMMessage, RetryingLLMProvider

MSG = [LLMMessage("user", "hello")]


class FakeProviderTests(unittest.TestCase):
    def test_returns_script_in_order_and_records_calls(self):
        fake = FakeLLMProvider(["one", "two"])
        self.assertEqual(fake.generate(MSG).text, "one")
        self.assertEqual(fake.generate(MSG).text, "two")
        self.assertEqual(len(fake.calls), 2)
        with self.assertRaises(LLMResponseError):
            fake.generate(MSG)  # exhausted script fails loudly instead of inventing text

    def test_responder_callable(self):
        fake = FakeLLMProvider(responder=lambda m: m[-1].content.upper())
        self.assertEqual(fake.generate(MSG).text, "HELLO")


class RetryTests(unittest.TestCase):
    def test_retries_transient_errors_with_exponential_backoff(self):
        sleeps = []
        fake = FakeLLMProvider([LLMRateLimitError("429"), LLMUnavailableError("503"), "finally"])
        llm = RetryingLLMProvider(fake, max_attempts=4, base_delay=1, sleep=sleeps.append)
        self.assertEqual(llm.generate(MSG).text, "finally")
        self.assertEqual(sleeps, [1, 2])
        self.assertEqual(len(fake.calls), 3)

    def test_gives_up_after_max_attempts_and_raises_last_error(self):
        sleeps = []
        fake = FakeLLMProvider([LLMRateLimitError("a"), LLMRateLimitError("b"), LLMRateLimitError("c")])
        llm = RetryingLLMProvider(fake, max_attempts=3, base_delay=1, sleep=sleeps.append)
        with self.assertRaises(LLMRateLimitError):
            llm.generate(MSG)
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(sleeps, [1, 2])  # no pointless sleep after the final failure

    def test_non_retryable_errors_are_not_retried(self):
        fake = FakeLLMProvider([LLMResponseError("blocked"), "never reached"])
        llm = RetryingLLMProvider(fake, sleep=lambda s: self.fail("should not sleep"))
        with self.assertRaises(LLMResponseError):
            llm.generate(MSG)
        self.assertEqual(len(fake.calls), 1)

    def test_delay_is_capped(self):
        sleeps = []
        fake = FakeLLMProvider([LLMUnavailableError("x")] * 5 + ["ok"])
        RetryingLLMProvider(fake, max_attempts=6, base_delay=10, max_delay=25, sleep=sleeps.append).generate(MSG)
        self.assertEqual(sleeps, [10, 20, 25, 25, 25])


class RetryVisibilityTests(unittest.TestCase):
    def test_retries_are_logged_not_silent(self):
        fake = FakeLLMProvider([LLMRateLimitError("429"), "ok"])
        with self.assertLogs("gharcare.llm.retry", level="WARNING") as logs:
            RetryingLLMProvider(fake, sleep=lambda s: None).generate(MSG)
        self.assertIn("retry 1/3 in 1s", logs.output[0])


class SettingsTests(unittest.TestCase):
    def test_defaults_and_derived_paths(self):
        s = Settings()
        self.assertEqual(s.db_path.name, "gharcare.db")
        self.assertEqual(s.artifacts_dir.name, "artifacts")

    def test_from_env(self):
        s = Settings.from_env({"GHARCARE_DATA_DIR": "/tmp/gc", "GHARCARE_MAX_UPLOAD_MB": "5",
                               "GHARCARE_LLM_PROVIDER": " Ollama ", "GHARCARE_LLM_MODEL": "  "})
        self.assertEqual(s.data_dir, Path("/tmp/gc"))  # compare as Paths: Windows uses backslashes
        self.assertEqual(s.max_upload_bytes, 5 * 1024 * 1024)
        self.assertEqual(s.llm_provider, "ollama")
        self.assertIsNone(s.llm_model)

    def test_rag_settings_from_env_and_factories(self):
        from gharcare.retrieval import HashingEmbedder, InMemoryVectorStore
        from gharcare.retrieval.factory import build_embedder, build_vector_store
        s = Settings.from_env({"GHARCARE_EMBEDDER": "Hashing", "GHARCARE_VECTOR_STORE": "memory",
                               "GHARCARE_EMBEDDING_MODEL": "BAAI/bge-m3"})
        self.assertEqual((s.embedder, s.vector_store, s.embedding_model), ("hashing", "memory", "BAAI/bge-m3"))
        self.assertIsInstance(build_embedder(s), HashingEmbedder)
        self.assertIsInstance(build_vector_store(s), InMemoryVectorStore)
        self.assertEqual(s.chroma_dir.name, "chroma")
        with self.assertRaises(DomainValidationError):
            build_embedder(Settings(embedder="magic"))
        with self.assertRaises(DomainValidationError):
            build_vector_store(Settings(vector_store="magic"))

    def test_bad_values_fail_clearly(self):
        with self.assertRaises(DomainValidationError):
            Settings.from_env({"GHARCARE_MAX_UPLOAD_MB": "lots"})
        with self.assertRaises(DomainValidationError):
            Settings(max_upload_bytes=0)


if __name__ == "__main__":
    unittest.main()
