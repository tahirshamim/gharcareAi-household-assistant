import unittest

from gharcare.config import Settings
from gharcare.errors import DomainValidationError, LLMError, LLMRateLimitError, LLMResponseError, LLMUnavailableError
from gharcare.llm import FakeLLMProvider, LangChainChatProvider, LLMMessage, RetryingLLMProvider, build_llm
from gharcare.llm.langchain_adapter import map_exception

MSGS = [LLMMessage("system", "be brief"), LLMMessage("user", "hi"), LLMMessage("assistant", "hello")]


class Reply:
    def __init__(self, content, usage=None):
        self.content, self.usage_metadata = content, usage


class FakeChat:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.sent = result, error, None

    def invoke(self, messages):
        self.sent = messages
        if self.error:
            raise self.error
        return self.result


class AdapterTests(unittest.TestCase):
    def test_roles_are_translated_and_usage_is_read(self):
        chat = FakeChat(Reply("  answer  ", {"input_tokens": 5, "output_tokens": 2}))
        out = LangChainChatProvider(chat, name="gemini", model="m1").generate(MSGS)
        self.assertEqual(chat.sent, [("system", "be brief"), ("human", "hi"), ("ai", "hello")])
        self.assertEqual((out.text, out.model, out.input_tokens, out.output_tokens), ("answer", "m1", 5, 2))

    def test_content_blocks_are_flattened(self):
        chat = FakeChat(Reply([{"type": "text", "text": "Hello "}, "world", {"type": "image"}]))
        self.assertEqual(LangChainChatProvider(chat).generate(MSGS).text, "Hello world")

    def test_empty_response_is_an_error(self):
        with self.assertRaises(LLMResponseError):
            LangChainChatProvider(FakeChat(Reply("   "))).generate(MSGS)

    def test_exception_mapping(self):
        self.assertIsInstance(map_exception(RuntimeError("429 Too Many Requests")), LLMRateLimitError)
        self.assertIsInstance(map_exception(RuntimeError("RESOURCE_EXHAUSTED quota")), LLMRateLimitError)
        self.assertIsInstance(map_exception(ConnectionError("connection refused")), LLMUnavailableError)
        self.assertIsInstance(map_exception(RuntimeError("503 Service Unavailable")), LLMUnavailableError)
        other = map_exception(ValueError("invalid api key"))
        self.assertIs(type(other), LLMError)                       # not retryable

    def test_rate_limits_are_retried_through_the_wrapper(self):
        class Flaky:
            calls = 0
            def invoke(self, messages):
                Flaky.calls += 1
                if Flaky.calls < 3:
                    raise RuntimeError("429 rate limit")
                return Reply("ok")
        sleeps = []
        llm = RetryingLLMProvider(LangChainChatProvider(Flaky()), sleep=sleeps.append)
        self.assertEqual(llm.generate(MSGS).text, "ok")
        self.assertEqual(sleeps, [1.0, 2.0])


class FactoryTests(unittest.TestCase):
    def test_configuration_errors_are_clear_and_precede_any_import(self):
        with self.assertRaises(DomainValidationError):
            build_llm(Settings(llm_provider="magic", llm_model="m"), env={})
        with self.assertRaisesRegex(DomainValidationError, "GHARCARE_LLM_MODEL"):
            build_llm(Settings(llm_provider="gemini"), env={"GEMINI_API_KEY": "k"})
        with self.assertRaisesRegex(DomainValidationError, "GEMINI_API_KEY"):
            build_llm(Settings(llm_provider="gemini", llm_model="m"), env={})


if __name__ == "__main__":
    unittest.main()
