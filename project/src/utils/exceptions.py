"""Custom exceptions for the platform."""


class PlatformError(Exception):
    """Base platform exception."""


class SchemaValidationError(PlatformError):
    """Raised when data fails schema validation."""


class IndexNotBuiltError(PlatformError):
    """Raised when a retrieval index is not built."""


class EmbeddingCacheError(PlatformError):
    """Raised when embedding cache operations fail."""
