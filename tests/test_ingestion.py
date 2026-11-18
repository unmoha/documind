"""Unit tests for document loading and recursive text chunking."""

from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from documind.ingestion.chunker import (
    RecursiveCharacterChunker,
    chunk_document,
)
from documind.ingestion.loaders import (
    compute_file_hash,
    load_document,
    load_markdown,
    load_pdf,
    load_txt,
)


def _generate_pdf(file_path: Path, pages_text: list[str]) -> Path:
    """Helper to generate a valid multi-page PDF with text using pypdf."""
    writer = PdfWriter()
    for text in pages_text:
        page = writer.add_blank_page(width=612, height=792)
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 700 Td ({text}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = stream

        font_dict = DictionaryObject()
        font_dict[NameObject("/Type")] = NameObject("/Font")
        font_dict[NameObject("/Subtype")] = NameObject("/Type1")
        font_dict[NameObject("/BaseFont")] = NameObject("/Helvetica")

        resources = DictionaryObject()
        fonts = DictionaryObject()
        fonts[NameObject("/F1")] = font_dict
        resources[NameObject("/Font")] = fonts
        page[NameObject("/Resources")] = resources

    with open(file_path, "wb") as f:
        writer.write(f)
    return file_path


def test_empty_files(tmp_path: Path) -> None:
    """Verify loading and chunking empty (0-byte and whitespace) files."""
    # 0-byte TXT file
    empty_txt = tmp_path / "empty.txt"
    empty_txt.write_bytes(b"")
    pages_txt = load_document(empty_txt)
    assert pages_txt == []
    chunks_txt = chunk_document(pages_txt)
    assert chunks_txt == []

    # 0-byte Markdown file
    empty_md = tmp_path / "empty.md"
    empty_md.write_bytes(b"")
    pages_md = load_document(empty_md)
    assert pages_md == []
    chunks_md = chunk_document(pages_md)
    assert chunks_md == []

    # 0-byte PDF file
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")
    pages_pdf = load_document(empty_pdf)
    assert pages_pdf == []
    chunks_pdf = chunk_document(pages_pdf)
    assert chunks_pdf == []

    # Whitespace-only file
    ws_file = tmp_path / "whitespace.txt"
    ws_file.write_text("   \n\n\t  \n  ")
    pages_ws = load_document(ws_file)
    assert len(pages_ws) == 1
    chunks_ws = chunk_document(pages_ws)
    assert chunks_ws == []


def test_very_short_file(tmp_path: Path) -> None:
    """Verify loading and chunking a very short file (under chunk_size)."""
    short_file = tmp_path / "short_doc.txt"
    content = "DocuMind is an open-source, production-grade RAG system."
    short_file.write_text(content, encoding="utf-8")

    expected_hash = compute_file_hash(short_file.read_bytes())
    pages = load_document(short_file)

    assert len(pages) == 1
    page = pages[0]
    assert page.filename == "short_doc.txt"
    assert page.page_number == 1
    assert page.text == content
    assert page.file_hash == expected_hash

    chunks = chunk_document(pages)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.filename == "short_doc.txt"
    assert chunk.page_number == 1
    assert chunk.chunk_index == 0
    assert chunk.file_hash == expected_hash
    assert chunk.text == content


def test_very_long_file(tmp_path: Path) -> None:
    """Verify chunking a very long file into multiple chunks with overlap."""
    long_file = tmp_path / "long_doc.txt"
    # Generate ~3500 characters across multiple paragraphs
    paragraphs = [
        f"Paragraph {p}: " + " ".join([f"term_{p}_{w}" for w in range(40)])
        for p in range(12)
    ]
    full_text = "\n\n".join(paragraphs)
    long_file.write_text(full_text, encoding="utf-8")

    expected_hash = compute_file_hash(long_file.read_bytes())
    pages = load_document(long_file)
    assert len(pages) == 1

    chunker = RecursiveCharacterChunker(chunk_size=800, chunk_overlap=120)
    chunks = chunk_document(pages, chunker=chunker)

    # Must produce multiple chunks
    assert len(chunks) > 1

    # Check boundaries and metadata
    for i, chunk in enumerate(chunks):
        assert chunk.filename == "long_doc.txt"
        assert chunk.page_number == 1
        assert chunk.chunk_index == i
        assert chunk.file_hash == expected_hash
        assert len(chunk.text) <= 800

    # Verify overlap exists between consecutive chunks
    for i in range(len(chunks) - 1):
        curr_chunk = chunks[i].text
        next_chunk = chunks[i + 1].text
        # Overlap means some trailing tokens from curr_chunk exist in next_chunk
        curr_tail_words = curr_chunk.split()[-3:]
        tail_sub = " ".join(curr_tail_words)
        assert tail_sub in next_chunk, (
            f"Expected overlap between chunk {i} and {i + 1}. "
            f"Tail '{tail_sub}' not found in next chunk."
        )


def test_unicode_text(tmp_path: Path) -> None:
    """Verify loading and chunking documents with Unicode and multilingual characters."""
    unicode_file = tmp_path / "unicode_doc.md"
    content = (
        "# Multilingual Benchmark\n\n"
        "Café, résumé, façade, and naïve.\n"
        "Greek: αβγδε, Ωμέγα, Σύστημα.\n"
        "Japanese: 日本語のテキストと検索拡張生成。\n"
        "Chinese: 检索增强生成与文档分块。\n"
        "Arabic: نظام توليد معزز بالاسترجاع عالي الدقة.\n"
        "Emojis: 🚀 🤖 📚 🔍 💡 🛡️\n"
        "Math: ∑_{i=1}^n x_i ≈ ∫_0^∞ f(x) dx ≠ 0."
    )
    unicode_file.write_text(content, encoding="utf-8")

    pages = load_document(unicode_file)
    assert len(pages) == 1
    assert "日本語" in pages[0].text
    assert "🚀" in pages[0].text
    assert "نظام" in pages[0].text

    chunks = chunk_document(pages)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].filename == "unicode_doc.md"
    assert "日本語" in chunks[0].text
    assert "🚀" in chunks[0].text
    assert chunks[0].file_hash == compute_file_hash(unicode_file.read_bytes())


def test_multipage_pdf(tmp_path: Path) -> None:
    """Verify loading a multi-page PDF, preserving page numbers and chunk sequencing."""
    pdf_path = tmp_path / "multipage_doc.pdf"
    page_texts = [
        "Executive Summary and Introduction to DocuMind architecture.",
        "System Design and Ingestion Pipeline Specification.",
        "Retrieval Evaluation Hit Rate and Latency Metrics.",
        "Conclusion and Production Deployment Readiness.",
    ]
    _generate_pdf(pdf_path, page_texts)

    expected_hash = compute_file_hash(pdf_path.read_bytes())
    pages = load_document(pdf_path)

    # 4 pages extracted
    assert len(pages) == 4
    for idx, (page, expected_text) in enumerate(
        zip(pages, page_texts, strict=True), start=1
    ):
        assert page.page_number == idx
        assert page.filename == "multipage_doc.pdf"
        assert page.file_hash == expected_hash
        assert expected_text in page.text

    chunks = chunk_document(pages)
    assert len(chunks) == 4

    for idx, (chunk, expected_text) in enumerate(zip(chunks, page_texts, strict=True)):
        assert chunk.chunk_index == idx
        assert chunk.page_number == idx + 1
        assert chunk.filename == "multipage_doc.pdf"
        assert chunk.file_hash == expected_hash
        assert expected_text in chunk.text


def test_chunker_validation() -> None:
    """Verify that invalid chunker configurations raise ValueErrors."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        RecursiveCharacterChunker(chunk_size=0)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        RecursiveCharacterChunker(chunk_size=-50)

    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        RecursiveCharacterChunker(chunk_size=500, chunk_overlap=-10)

    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        RecursiveCharacterChunker(chunk_size=500, chunk_overlap=500)

    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        RecursiveCharacterChunker(chunk_size=500, chunk_overlap=600)


def test_loader_error_handling(tmp_path: Path) -> None:
    """Verify error conditions on nonexistent, unsupported, and corrupted files."""
    # Nonexistent file
    nonexistent = tmp_path / "does_not_exist.txt"
    with pytest.raises(FileNotFoundError):
        load_document(nonexistent)

    # Unsupported format
    unsupported = tmp_path / "image.png"
    unsupported.write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_document(unsupported)

    # Corrupted PDF
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"Not a valid PDF header content")
    with pytest.raises(ValueError, match="Failed to parse PDF"):
        load_pdf(corrupt_pdf)


def test_explicit_loaders(tmp_path: Path) -> None:
    """Verify direct usage of load_txt and load_markdown."""
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Hello Text", encoding="utf-8")
    assert len(load_txt(txt_file)) == 1

    md_file = tmp_path / "sample.md"
    md_file.write_text("# Hello Markdown", encoding="utf-8")
    assert len(load_markdown(md_file)) == 1
