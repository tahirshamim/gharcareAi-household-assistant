"""Exception hierarchy. Every error GharCare raises on purpose derives from GharCareError,
so API layers can catch one base type and map subclasses to HTTP status codes."""


class GharCareError(Exception):
    """Base class for all expected application errors."""


class DomainValidationError(GharCareError, ValueError):
    """Input violates a domain rule (empty name, negative cost, bad identifier...)."""


class NotFoundError(GharCareError):
    """Entity does not exist, or does not belong to the requesting user."""


class StorageError(GharCareError):
    """Artifact or database storage failed or rejected the request."""


class LLMError(GharCareError):
    """Base class for language-model failures."""


class LLMRateLimitError(LLMError):
    """Provider rate limit hit (retryable). `retry_after` is the wait in seconds the provider asked for, if any."""

    def __init__(self, message: str = "", *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMUnavailableError(LLMError):
    """Provider temporarily unreachable (retryable)."""


class LLMResponseError(LLMError):
    """Provider answered, but the response is unusable, e.g. blocked or empty (not retryable)."""


class IngestionError(GharCareError):
    """A document could not be read or processed (corrupt file, missing parser, ...)."""


class NoExtractableTextError(IngestionError):
    """The file contains no readable text, e.g. a scanned image-only PDF (OCR is not supported yet)."""


class EmbeddingError(GharCareError):
    """The embedding model failed or returned unusable output."""


class WorkflowError(GharCareError):
    """The assistant workflow graph is misconfigured or ran into an impossible state."""


class LLMQuotaExhaustedError(LLMError):
    """A daily/plan quota is used up. Retrying is pointless (unlike a per-minute rate limit)."""
