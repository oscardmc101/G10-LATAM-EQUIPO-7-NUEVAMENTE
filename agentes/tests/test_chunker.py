import pytest
from agentes.rag.chunker import create_chunks
from agentes.rag.config import DEFAULT_RAG_CONFIG
from agentes.rag.models import Document


def test_create_chunks_success():
    doc = Document(
        text="Este es un texto de prueba suficientemente largo para ser dividido. " * 30,
        metadata={"document_id": "AI-ES-001", "page": 1, "source": "test.txt"}
    )
    chunks = create_chunks([doc])

    assert len(chunks) > 0
    for i, chunk in enumerate(chunks):
        assert chunk.id == f"AI-ES-001_1_{i}"
        assert chunk.metadata["document_id"] == "AI-ES-001"
        assert chunk.metadata["chunk_index"] == i
        assert chunk.document_id == "AI-ES-001"


def test_create_chunks_missing_document_id():
    doc = Document(
        text="Texto de prueba",
        metadata={"source": "test.txt"}
    )
    with pytest.raises(ValueError, match="document_id"):
        create_chunks([doc])


def test_create_chunks_empty_document_id():
    doc = Document(
        text="Texto de prueba",
        metadata={"document_id": "   "}
    )
    with pytest.raises(ValueError, match="document_id"):
        create_chunks([doc])


def test_create_chunks_invalid_overlap():
    doc = Document(
        text="Texto",
        metadata={"document_id": "DOC-1"}
    )
    with pytest.raises(ValueError, match="chunk_overlap"):
        create_chunks([doc], chunk_size=100, chunk_overlap=150)


def test_create_chunks_uses_default_rag_config():
    # Verifica que los defaults coincidan con DEFAULT_RAG_CONFIG
    import inspect
    sig = inspect.signature(create_chunks)
    assert sig.parameters["chunk_size"].default == DEFAULT_RAG_CONFIG.chunk_size
    assert sig.parameters["chunk_overlap"].default == DEFAULT_RAG_CONFIG.chunk_overlap
