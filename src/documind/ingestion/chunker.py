"""Recursive character text chunker for document ingestion.

Splits document text into manageable, semantically coherent chunks using
hierarchical separators while enforcing maximum chunk size and overlap boundaries.
"""

from documind.config.settings import get_settings
from documind.ingestion.models import DocumentChunk, DocumentPage

DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", " ", ""]


class RecursiveCharacterChunker:
    """Splits text into chunks using a hierarchical list of separators.

    Separators are evaluated in order of preference (e.g. double newline,
    single newline, space, empty string) to keep semantic units such as
    paragraphs and sentences intact whenever possible.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ) -> None:
        """Initialize the chunker with size, overlap, and separator preferences.

        Args:
            chunk_size: Maximum character count per chunk. Defaults to Settings.chunk_size.
            chunk_overlap: Character overlap between consecutive chunks. Defaults to Settings.chunk_overlap.
            separators: Hierarchical separator strings. Defaults to ["\\n\\n", "\\n", " ", ""].

        Raises:
            ValueError: If chunk_size <= 0, chunk_overlap < 0, or chunk_overlap >= chunk_size.
        """
        settings = get_settings()
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        )

        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(
                f"chunk_overlap must be non-negative, got {self.chunk_overlap}"
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less "
                f"than chunk_size ({self.chunk_size})"
            )

        self.separators = (
            list(separators) if separators is not None else DEFAULT_SEPARATORS
        )

    def _get_overlap_prefix(self, prev_text: str) -> str:
        """Extract a clean tail portion of the previous text for overlap prefixing.

        Attempts to align the overlap boundary with whitespace to avoid clipping words.

        Args:
            prev_text: Full text of the preceding chunk.

        Returns:
            Overlap prefix string of at most `self.chunk_overlap` characters.
        """
        if not prev_text or self.chunk_overlap <= 0:
            return ""

        tail = prev_text[-self.chunk_overlap :]
        # If there is whitespace in the first half of the tail, snap to it for clean word boundaries
        first_space = tail.find(" ")
        if first_space != -1 and first_space < len(tail) // 2:
            return tail[first_space + 1 :].strip()
        return tail.strip()

    def _split_text_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Internal recursive helper to decompose text across separators.

        Args:
            text: Text to split.
            separators: Remaining candidate separators.

        Returns:
            List of text segments each <= self.chunk_size (or character splits).
        """
        if not text or not text.strip():
            return []

        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text]

        # Identify the first matching separator
        chosen_sep = ""
        for sep in separators:
            if sep == "":
                chosen_sep = ""
                break
            if sep in text:
                chosen_sep = sep
                break

        if chosen_sep != "":
            splits = [s for s in text.split(chosen_sep) if s]
        else:
            splits = list(text)

        next_separators = (
            separators[separators.index(chosen_sep) + 1 :] if chosen_sep != "" else []
        )

        processed_splits: list[str] = []
        for s in splits:
            if len(s) <= self.chunk_size:
                processed_splits.append(s)
            else:
                if next_separators:
                    processed_splits.extend(
                        self._split_text_recursive(s, next_separators)
                    )
                else:
                    # Fallback character splitting
                    processed_splits.append(s)

        # Merge splits into chunks respecting chunk_size and chunk_overlap
        chunks: list[str] = []
        current_pieces: list[str] = []
        current_len = 0

        for piece in processed_splits:
            sep_len = len(chosen_sep) if current_pieces else 0
            if current_len + sep_len + len(piece) <= self.chunk_size:
                current_pieces.append(piece)
                current_len += sep_len + len(piece)
            else:
                if current_pieces:
                    chunk_str = chosen_sep.join(current_pieces).strip()
                    if chunk_str:
                        chunks.append(chunk_str)

                    # Build overlap from recent pieces
                    overlap_pieces: list[str] = []
                    overlap_len = 0
                    for p in reversed(current_pieces):
                        p_sep = len(chosen_sep) if overlap_pieces else 0
                        if overlap_len + p_sep + len(p) <= self.chunk_overlap:
                            overlap_pieces.insert(0, p)
                            overlap_len += p_sep + len(p)
                        else:
                            break

                    if overlap_pieces:
                        current_pieces = overlap_pieces
                        current_len = overlap_len
                    else:
                        tail_overlap = self._get_overlap_prefix(chunk_str)
                        if (
                            tail_overlap
                            and len(tail_overlap) + len(chosen_sep) + len(piece)
                            <= self.chunk_size
                        ):
                            current_pieces = [tail_overlap]
                            current_len = len(tail_overlap)
                        else:
                            current_pieces = []
                            current_len = 0

                sep_len = len(chosen_sep) if current_pieces else 0
                if current_len + sep_len + len(piece) <= self.chunk_size:
                    current_pieces.append(piece)
                    current_len += sep_len + len(piece)
                else:
                    if current_pieces:
                        flushed = chosen_sep.join(current_pieces).strip()
                        if flushed:
                            chunks.append(flushed)
                    current_pieces = [piece]
                    current_len = len(piece)

        if current_pieces:
            final_chunk = chosen_sep.join(current_pieces).strip()
            if final_chunk:
                chunks.append(final_chunk)

        return chunks

    def split_text(self, text: str) -> list[str]:
        """Split a string of text into chunks of at most `self.chunk_size`.

        Args:
            text: Raw input text.

        Returns:
            List of string chunks. Returns empty list if text is empty or whitespace.
        """
        if not text or not text.strip():
            return []
        normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
        return self._split_text_recursive(normalized_text, self.separators)

    def chunk_document(self, pages: list[DocumentPage]) -> list[DocumentChunk]:
        """Chunk a list of DocumentPages into DocumentChunks with sequential indices.

        Args:
            pages: List of DocumentPage objects from document loaders.

        Returns:
            List of DocumentChunk objects with sequential chunk indices and source metadata.
        """
        chunks: list[DocumentChunk] = []
        chunk_index = 0

        for page in pages:
            if not page.text or not page.text.strip():
                continue

            page_chunks = self.split_text(page.text)
            for text_chunk in page_chunks:
                chunks.append(
                    DocumentChunk(
                        text=text_chunk,
                        filename=page.filename,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        file_hash=page.file_hash,
                    )
                )
                chunk_index += 1

        return chunks


def chunk_document(
    pages: list[DocumentPage],
    chunker: RecursiveCharacterChunker | None = None,
) -> list[DocumentChunk]:
    """Convenience function to chunk document pages using a default or custom chunker.

    Args:
        pages: List of DocumentPage objects.
        chunker: Optional custom RecursiveCharacterChunker instance.

    Returns:
        List of DocumentChunk objects.
    """
    if chunker is None:
        chunker = RecursiveCharacterChunker()
    return chunker.chunk_document(pages)
