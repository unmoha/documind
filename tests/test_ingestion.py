"""Unit tests for document loading, chunking, embedding, and vector storage."""

import math
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from documind.ingestion.chunker import (
    RecursiveCharacterChunker,
    chunk_document,
)
from documind.ingestion.embedder import FakeEmbedder
from documind.ingestion.errors import EmptyPDFError
from documind.ingestion.loaders import (
    compute_file_hash,
    load_document,
    load_markdown,
    load_pdf,
    load_txt,
)
from documind.ingestion.models import DocumentChunk
from documind.ingestion.pipeline import ingest_file
from documind.ingestion.storage import ChromaVectorStore


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


def _generate_blank_pdf(file_path: Path) -> Path:
    """Helper to generate a valid PDF with blank pages (no extractable text)."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
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

    assert len(chunks) > 1

    for i, chunk in enumerate(chunks):
        assert chunk.filename == "long_doc.txt"
        assert chunk.page_number == 1
        assert chunk.chunk_index == i
        assert chunk.file_hash == expected_hash
        assert len(chunk.text) <= 800

    for i in range(len(chunks) - 1):
        curr_chunk = chunks[i].text
        next_chunk = chunks[i + 1].text
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


# --- New Tests for Embeddings, Storage, Idempotency, and PDF validation ---


def test_no_chunk_exceeds_chunk_size() -> None:
    """Assert that no chunk produced by RecursiveCharacterChunker exceeds chunk_size characters.

    Tests against varying chunk sizes, overlaps, continuous text strings without spaces,
    dense technical text, code blocks, and irregular punctuation.
    """
    test_cases = [
        # (chunk_size, chunk_overlap, text)
        (100, 20, "Short sentence that easily fits in one chunk."),
        (
            50,
            10,
            "A" * 150 + " " + "B" * 120 + "\n\n" + "C" * 80,
        ),
        (
            80,
            15,
            "Dense paragraph with multiple sentences. Each sentence tests boundaries. "
            "Another sentence here with abbreviations like e.g. and i.e. followed by words. "
            * 4,
        ),
        (
            200,
            40,
            "```python\ndef calculate_metric(x, y):\n    return (x ** 2 + y ** 2) ** 0.5\n```\n\n"
            * 10,
        ),
        (
            300,
            50,
            "ContinuousNoWhitespaceTextThatForcesCharacterLevelSplittingAtChunkBoundaries"
            * 15,
        ),
        (
            60,
            25,
            "Mixed whitespace \t\t and newlines \n\n with symbols !@#$%^&*()_+{}[]:;'<>?,./ "
            * 6,
        ),
    ]

    for chunk_size, chunk_overlap, text in test_cases:
        chunker = RecursiveCharacterChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = chunker.split_text(text)
        assert len(chunks) > 0, "Expected at least one chunk to be produced"
        for idx, chunk in enumerate(chunks):
            assert len(chunk) <= chunk_size, (
                f"Chunk {idx} exceeded chunk_size: len={len(chunk)} > {chunk_size}. "
                f"Content: {chunk!r}"
            )


def test_pdf_no_extractable_text_fails(tmp_path: Path) -> None:
    """Verify that a PDF yielding no extractable text fails with a clear EmptyPDFError."""
    # 1. Valid PDF file structure, but pages contain zero text streams (e.g. image or blank)
    blank_pdf = tmp_path / "blank_page.pdf"
    _generate_blank_pdf(blank_pdf)

    # Loader must fail directly with EmptyPDFError
    with pytest.raises(EmptyPDFError, match="contains no extractable text"):
        load_pdf(blank_pdf)

    with pytest.raises(EmptyPDFError, match="contains no extractable text"):
        load_document(blank_pdf)

    # Ingestion pipeline must fail with EmptyPDFError
    fake_embedder = FakeEmbedder()
    store = ChromaVectorStore(persist_dir=tmp_path / "chroma_err_test")
    with pytest.raises(EmptyPDFError, match="contains no extractable text"):
        ingest_file(blank_pdf, embedder=fake_embedder, store=store)

    # 2. 0-byte PDF in ingestion pipeline
    zero_byte_pdf = tmp_path / "zero_bytes.pdf"
    zero_byte_pdf.write_bytes(b"")
    with pytest.raises(EmptyPDFError, match="yielded no extractable text"):
        ingest_file(zero_byte_pdf, embedder=fake_embedder, store=store)


def test_fake_embedder_unit_normalized() -> None:
    """Verify that FakeEmbedder produces deterministic, unit-normalized vectors without network access."""
    embedder = FakeEmbedder(dimension=384)

    # 1. Determinism
    vec1 = embedder.embed_query("DocuMind RAG system")
    vec2 = embedder.embed_query("DocuMind RAG system")
    assert vec1 == vec2, "FakeEmbedder must be strictly deterministic"

    # 2. Different text yields different vectors
    vec3 = embedder.embed_query("Completely different search query")
    assert vec1 != vec3

    # 3. Dimensionality
    assert len(vec1) == 384

    # 4. Unit norm check (L2 norm should be ~ 1.0)
    norm = math.sqrt(sum(x * x for x in vec1))
    assert pytest.approx(norm, rel=1e-3) == 1.0

    # 5. Batched embedding of multiple documents
    docs = ["Doc 1 text", "Doc 2 text", "Doc 3 text"]
    doc_vecs = embedder.embed_documents(docs)
    assert len(doc_vecs) == 3
    assert all(len(v) == 384 for v in doc_vecs)


def test_chroma_vector_store_operations(
    tmp_path: Path,
    fake_embedder: FakeEmbedder,
) -> None:
    """Verify ChromaVectorStore insertion, metadata preservation, query, and deletion."""
    persist_dir = tmp_path / "chroma_store_test"
    store = ChromaVectorStore(
        persist_dir=persist_dir,
        collection_name="test_ops",
    )

    chunks = [
        DocumentChunk(
            text="First chunk of knowledge base.",
            filename="guide.txt",
            page_number=1,
            chunk_index=0,
            file_hash="hash_abc123",
        ),
        DocumentChunk(
            text="Second chunk of knowledge base.",
            filename="guide.txt",
            page_number=1,
            chunk_index=1,
            file_hash="hash_abc123",
        ),
    ]
    embeddings = fake_embedder.embed_documents([c.text for c in chunks])

    # Insert chunks
    added_count = store.add_chunks(chunks, embeddings)
    assert added_count == 2
    assert store.count() == 2
    assert store.has_document("hash_abc123") is True
    assert store.has_document("hash_unknown") is False

    # Retrieve chunks by hash
    retrieved = store.get_document_chunks("hash_abc123")
    assert len(retrieved) == 2
    assert retrieved[0]["metadata"]["filename"] == "guide.txt"
    assert retrieved[0]["metadata"]["chunk_index"] == 0

    # List documents
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0]["document_id"] == "hash_abc123"
    assert docs[0]["chunk_count"] == 2

    # Delete document
    deleted = store.delete_document("hash_abc123")
    assert deleted == 2
    assert store.count() == 0
    assert store.has_document("hash_abc123") is False


def test_idempotent_ingestion(
    tmp_path: Path,
    fake_embedder: FakeEmbedder,
    test_vector_store: ChromaVectorStore,
) -> None:
    """Verify that re-ingesting the exact same file creates no duplicates and skips recomputation."""
    sample_doc = tmp_path / "architecture.md"
    sample_doc.write_text(
        "# DocuMind Architecture\n\n"
        "DocuMind uses ChromaDB for persistent vector storage and local BGE embeddings.\n"
        "It enforces strict groundedness and transparent citations.\n\n"
        "## Ingestion Pipeline\n"
        "The ingestion pipeline extracts text, chunks recursively, and stores vectors.",
        encoding="utf-8",
    )

    # Initial ingestion
    res1 = ingest_file(
        sample_doc,
        embedder=fake_embedder,
        store=test_vector_store,
    )
    assert res1.status == "indexed"
    assert res1.chunks_indexed > 0
    initial_chunks = res1.chunks_indexed
    assert test_vector_store.count() == initial_chunks

    # Second ingestion of the same file (must be idempotent)
    res2 = ingest_file(
        sample_doc,
        embedder=fake_embedder,
        store=test_vector_store,
    )
    assert res2.status == "already_indexed"
    assert res2.chunks_indexed == initial_chunks
    assert res2.document_id == res1.document_id

    # Crucial assertion: Total vector store count must NOT have increased
    assert test_vector_store.count() == initial_chunks

    # Third ingestion with a copied/renamed file containing identical content
    renamed_doc = tmp_path / "architecture_copy.md"
    renamed_doc.write_bytes(sample_doc.read_bytes())
    res3 = ingest_file(
        renamed_doc,
        embedder=fake_embedder,
        store=test_vector_store,
    )
    assert res3.status == "already_indexed"
    assert test_vector_store.count() == initial_chunks

    # Fourth ingestion with force=True (must replace chunks in-place without duplicating)
    res4 = ingest_file(
        sample_doc,
        embedder=fake_embedder,
        store=test_vector_store,
        force=True,
    )
    assert res4.status == "indexed"
    assert res4.chunks_indexed == initial_chunks
    assert test_vector_store.count() == initial_chunks


def test_ingest_multipage_pdf_pipeline(
    tmp_path: Path,
    fake_embedder: FakeEmbedder,
    test_vector_store: ChromaVectorStore,
) -> None:
    """Verify end-to-end ingestion of a valid multi-page PDF into ChromaDB."""
    pdf_path = tmp_path / "report.pdf"
    pages_text = [
        "First section: Introduction to DocuMind retrieval benchmarks.",
        "Second section: ChromaDB vector search and embedding indexing.",
        "Third section: Faithfulness evaluation and citation metrics.",
    ]
    _generate_pdf(pdf_path, pages_text)

    result = ingest_file(
        pdf_path,
        embedder=fake_embedder,
        store=test_vector_store,
    )
    assert result.status == "indexed"
    assert result.chunks_indexed == 3
    assert result.filename == "report.pdf"
    assert test_vector_store.count() == 3

    # Check that retrieved chunks have correct page metadata
    chunks = test_vector_store.get_document_chunks(result.file_hash)
    assert len(chunks) == 3
    page_nums = [c["metadata"]["page_number"] for c in chunks]
    assert page_nums == [1, 2, 3]


def test_reingest_modified_file_purges_stale_chunks(
    tmp_path: Path,
    fake_embedder: FakeEmbedder,
    test_vector_store: ChromaVectorStore,
) -> None:
    """Verify that re-ingesting a modified file replaces stale chunks without duplicates."""
    doc_path = tmp_path / "notes.md"
    doc_path.write_text(
        "# Heading 1\n\nContent for section 1.\n\n"
        "# Heading 2\n\nContent for section 2.\n\n"
        "# Heading 3\n\nContent for section 3.",
        encoding="utf-8",
    )

    custom_chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)

    # Initial ingestion
    res1 = ingest_file(
        doc_path,
        embedder=fake_embedder,
        store=test_vector_store,
        chunker=custom_chunker,
    )
    assert res1.status == "indexed"
    assert res1.chunks_indexed == 3
    assert res1.chunks_replaced == 0
    assert test_vector_store.count() == 3

    # Edit file content (shortened to a single chunk with a new content hash)
    doc_path.write_text(
        "# Updated Notes\n\nShort single chunk.",
        encoding="utf-8",
    )

    # Re-ingest the modified file
    res2 = ingest_file(
        doc_path,
        embedder=fake_embedder,
        store=test_vector_store,
        chunker=custom_chunker,
    )
    assert res2.status == "indexed"
    assert res2.chunks_indexed == 1
    assert res2.chunks_replaced == 3
    assert res2.file_hash != res1.file_hash

    # Assert only the new version's chunks remain in ChromaDB
    assert test_vector_store.count() == 1
    assert test_vector_store.has_document(res1.file_hash) is False
    assert test_vector_store.has_document(res2.file_hash) is True

    remaining_chunks = test_vector_store.get_chunks_by_filename("notes.md")
    assert len(remaining_chunks) == 1
    assert remaining_chunks[0]["metadata"]["file_hash"] == res2.file_hash
    assert "Short single chunk" in remaining_chunks[0]["text"]


def test_delete_by_filename_in_vector_store(
    tmp_path: Path,
    fake_embedder: FakeEmbedder,
    test_vector_store: ChromaVectorStore,
) -> None:
    """Verify delete_by_filename removes only targeted document chunks."""
    file_a = tmp_path / "file_a.txt"
    file_a.write_text("Unique text for file A.", encoding="utf-8")
    file_b = tmp_path / "file_b.txt"
    file_b.write_text("Unique text for file B.", encoding="utf-8")

    res_a = ingest_file(file_a, embedder=fake_embedder, store=test_vector_store)
    res_b = ingest_file(file_b, embedder=fake_embedder, store=test_vector_store)
    assert test_vector_store.count() == res_a.chunks_indexed + res_b.chunks_indexed

    # Delete file A
    removed = test_vector_store.delete_by_filename("file_a.txt")
    assert removed == res_a.chunks_indexed
    assert test_vector_store.count() == res_b.chunks_indexed
    assert len(test_vector_store.get_chunks_by_filename("file_a.txt")) == 0
    assert (
        len(test_vector_store.get_chunks_by_filename("file_b.txt"))
        == res_b.chunks_indexed
    )

    # Deleting non-existent file returns 0
    assert test_vector_store.delete_by_filename("non_existent.txt") == 0
