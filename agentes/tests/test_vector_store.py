import pytest
from agentes.rag.models import Chunk


def test_vector_store_cosine_space_and_search(temp_vector_store):
    # Verificar que el espacio sea efectivamente coseno
    metadata = temp_vector_store.collection.metadata
    assert metadata.get("hnsw:space") == "cosine"

    chunks = [
        Chunk(
            id="AI-01_1",
            text="Inteligencia artificial y aprendizaje profundo",
            metadata={"document_id": "AI-01", "page": 1}
        ),
        Chunk(
            id="CLD-01_1",
            text="Infraestructura de nube y Kubernetes",
            metadata={"document_id": "CLD-01", "page": 1}
        )
    ]
    temp_vector_store.add_chunks(chunks)

    results = temp_vector_store.search(query="Inteligencia artificial", top_k=2)
    assert len(results) == 2
    for r in results:
        # Cosine similarity debe estar razonablemente acotada entre -1.0 y 1.0 (generalmente > 0 para textos similares)
        assert -1.0 <= r.score <= 1.0
        assert r.document_id in ("AI-01", "CLD-01")


def test_vector_store_filter_by_document_id(temp_vector_store):
    chunks = [
        Chunk(
            id="AI-01_1",
            text="Redes neuronales convolucionales",
            metadata={"document_id": "AI-01"}
        ),
        Chunk(
            id="AI-02_1",
            text="Redes neuronales recurrentes",
            metadata={"document_id": "AI-02"}
        )
    ]
    temp_vector_store.add_chunks(chunks)

    # Filtrar solo por AI-01
    results = temp_vector_store.search(
        query="redes neuronales",
        top_k=5,
        document_id="AI-01"
    )
    assert len(results) == 1
    assert results[0].document_id == "AI-01"
    assert results[0].chunk_id == "AI-01_1"


def test_vector_store_empty_collection(temp_vector_store):
    results = temp_vector_store.search(query="algo inexistente", top_k=5)
    assert results == []


def test_vector_store_rejects_chunk_without_document_id(temp_vector_store):
    chunk_sin_id = Chunk(id="CH-1", text="Texto", metadata={"source": "test.txt"})
    with pytest.raises(ValueError, match="document_id"):
        temp_vector_store.add_chunks([chunk_sin_id])


def test_vector_store_rejects_empty_filter_document_id(temp_vector_store):
    with pytest.raises(ValueError, match="document_id"):
        temp_vector_store.search(query="hola", top_k=5, document_id="   ")
