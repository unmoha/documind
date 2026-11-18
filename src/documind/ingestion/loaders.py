"""Document loaders for PDF, TXT, and Markdown files.

Extracts text content and preserves source metadata including filename,
page numbers, and cryptographic SHA-256 content hashes.
"""

import hashlib
import io
from pathlib import Path

import pypdf
from pypdf.errors import EmptyFileError, PdfReadError

from documind.ingestion.errors import EmptyPDFError
from documind.ingestion.models import DocumentPage


def compute_file_hash(data: bytes) -> str:
    """Compute the SHA-256 hexadecimal digest of raw document bytes.

    Args:
        data: Raw file bytes.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    return hashlib.sha256(data).hexdigest()


def load_pdf(file_path: Path | str) -> list[DocumentPage]:
    """Load and extract text from a PDF file page by page using pypdf.

    Args:
        file_path: Filesystem path to the PDF file.

    Returns:
        A list of DocumentPage objects, one per page in the PDF.
        Returns an empty list if the PDF file has zero bytes or zero pages.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is not a valid PDF or cannot be parsed.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    file_bytes = path.read_bytes()
    if len(file_bytes) == 0:
        return []

    file_hash = compute_file_hash(file_bytes)

    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    except (EmptyFileError, PdfReadError) as err:
        raise ValueError(f"Failed to parse PDF file '{path.name}': {err}") from err

    pages: list[DocumentPage] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        extracted = (
            (page.extract_text() or "").replace("\r\n", "\n").replace("\r", "\n")
        )
        pages.append(
            DocumentPage(
                text=extracted,
                filename=path.name,
                page_number=page_idx,
                file_hash=file_hash,
            )
        )

    if not any(page.text.strip() for page in pages):
        raise EmptyPDFError(
            f"PDF file '{path.name}' contains no extractable text. "
            "The document may be scanned, image-only, or blank."
        )

    return pages


def load_txt(file_path: Path | str, encoding: str = "utf-8") -> list[DocumentPage]:
    """Load and extract text from a plain text file.

    Args:
        file_path: Filesystem path to the text file.
        encoding: Text encoding (defaults to 'utf-8').

    Returns:
        A list containing a single DocumentPage with page_number=1,
        or an empty list if the file has zero bytes.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If decoding the file with the specified encoding fails.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    file_bytes = path.read_bytes()
    if len(file_bytes) == 0:
        return []

    file_hash = compute_file_hash(file_bytes)

    try:
        text = file_bytes.decode(encoding).replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError as err:
        raise ValueError(
            f"Failed to decode text file '{path.name}' with encoding '{encoding}': {err}"
        ) from err

    return [
        DocumentPage(
            text=text,
            filename=path.name,
            page_number=1,
            file_hash=file_hash,
        )
    ]


def load_markdown(file_path: Path | str, encoding: str = "utf-8") -> list[DocumentPage]:
    """Load and extract text from a Markdown file.

    Args:
        file_path: Filesystem path to the Markdown file.
        encoding: Text encoding (defaults to 'utf-8').

    Returns:
        A list containing a single DocumentPage with page_number=1,
        or an empty list if the file has zero bytes.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If decoding the file with the specified encoding fails.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    file_bytes = path.read_bytes()
    if len(file_bytes) == 0:
        return []

    file_hash = compute_file_hash(file_bytes)

    try:
        text = file_bytes.decode(encoding).replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError as err:
        raise ValueError(
            f"Failed to decode Markdown file '{path.name}' with encoding '{encoding}': {err}"
        ) from err

    return [
        DocumentPage(
            text=text,
            filename=path.name,
            page_number=1,
            file_hash=file_hash,
        )
    ]


def load_document(file_path: Path | str) -> list[DocumentPage]:
    """Dispatch file loading to the appropriate loader based on file extension.

    Supports:
        - PDF files (.pdf)
        - Text files (.txt)
        - Markdown files (.md, .markdown)

    Args:
        file_path: Filesystem path to the document.

    Returns:
        A list of DocumentPage objects extracted from the document.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file extension is unsupported or corrupted.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".txt":
        return load_txt(path)
    if suffix in (".md", ".markdown"):
        return load_markdown(path)

    raise ValueError(
        f"Unsupported file format '{suffix}' for file '{path.name}'. "
        "Supported formats: .pdf, .txt, .md, .markdown"
    )
