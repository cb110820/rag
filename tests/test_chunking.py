from pathlib import Path

import pytest

from src.rag.chunking import chunk_document
from src.rag.models import SourceDocument


def test_chunking_preserves_source_and_overlap() -> None:
    document = SourceDocument("x.md", "一" * 1200, Path("x.md"))
    chunks = chunk_document(document, chunk_size=500, overlap=50)
    assert len(chunks) == 3
    assert all(chunk.source_id == "x.md" for chunk in chunks)
    assert chunks[1].start == chunks[0].end - 50


def test_chunking_rejects_invalid_overlap() -> None:
    document = SourceDocument("x.md", "content", Path("x.md"))
    with pytest.raises(ValueError):
        chunk_document(document, chunk_size=100, overlap=100)
