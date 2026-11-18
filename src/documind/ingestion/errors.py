"""Exceptions for document loading, parsing, and ingestion pipelines."""


class IngestionError(Exception):
    """Base exception for all ingestion-related errors."""


class EmptyPDFError(ValueError, IngestionError):
    """Raised when a PDF file yields no extractable text content.

    Indicates the document may be scanned, image-only without OCR, blank,
    or corrupted.
    """
