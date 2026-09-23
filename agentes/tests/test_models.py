import pytest
from agentes.rag.models import Chunk, Document, SearchResult


def test_document_id_success():
    doc = Document(text="hola", metadata={"document_id": "DOC-123"})
    assert doc.document_id == "DOC-123"

    chunk = Chunk(id="CH-1", text="chunk", metadata={"document_id": "DOC-123"})
    assert chunk.document_id == "DOC-123"

    res = SearchResult(chunk_id="CH-1", text="chunk", score=0.9, metadata={"document_id": "DOC-123"})
    assert res.document_id == "DOC-123"


def test_document_id_fails_explicitly_when_missing():
    doc = Document(text="hola", metadata={"source": "manual.pdf"})
    with pytest.raises(KeyError):
        _ = doc.document_id

    chunk = Chunk(id="CH-1", text="chunk", metadata={"source": "manual.pdf"})
    with pytest.raises(KeyError):
        _ = chunk.document_id

    res = SearchResult(chunk_id="CH-1", text="chunk", score=0.9, metadata={"source": "manual.pdf"})
    with pytest.raises(KeyError):
        _ = res.document_id


def test_document_id_no_fallback_to_source():
    res = SearchResult(chunk_id="CH-1", text="chunk", score=0.9, metadata={"source": "archivo.txt"})
    # No debe retornar "archivo.txt" ni "" silenciosamente
    with pytest.raises(KeyError):
        _ = res.document_id
