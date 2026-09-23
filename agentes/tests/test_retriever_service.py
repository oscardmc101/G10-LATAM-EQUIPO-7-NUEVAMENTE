from unittest.mock import MagicMock
from agentes.rag.models import Chunk
from agentes.rag.retriever import RetrieverService


def test_retriever_service_success(temp_vector_store):
    chunks = [
        Chunk(
            id="AI-ES-001_CH_001",
            text="Conceptos de inteligencia artificial",
            metadata={"document_id": "AI-ES-001", "categoria": "IA"}
        ),
        Chunk(
            id="AI-ES-001_CH_002",
            text="Agentes generativos y LLMs",
            metadata={"document_id": "AI-ES-001", "categoria": "IA"}
        )
    ]
    temp_vector_store.add_chunks(chunks)

    service = RetrieverService(temp_vector_store)
    response = service.retrieve_for_evaluation(
        case_id="AI-ES-001-Q01",
        query="inteligencia artificial",
        top_k=2
    )

    assert response["contract_version"] == "1.0"
    assert response["case_id"] == "AI-ES-001-Q01"
    assert response["query"] == "inteligencia artificial"
    assert response["top_k"] == 2
    assert response["score_type"] == "cosine_similarity"
    assert response["status"] == "success"
    assert response["error"] is None
    assert len(response["results"]) == 2

    first = response["results"][0]
    assert first["rank"] == 1
    assert first["chunk_id"].startswith("AI-ES-001_CH_")
    assert first["document_id"] == "AI-ES-001"
    assert isinstance(first["score"], float)
    assert first["metadata"]["categoria"] == "IA"


def test_retriever_service_no_results(temp_vector_store):
    service = RetrieverService(temp_vector_store)
    response = service.retrieve_for_evaluation(
        case_id="CASE-01",
        query="algo que no existe",
        top_k=3
    )

    assert response["status"] == "no_results"
    assert response["results"] == []
    assert response["error"] is None


def test_retriever_service_error_handling(temp_vector_store):
    service = RetrieverService(temp_vector_store)
    # Simular una falla en el vector_store
    service.vector_store.search = MagicMock(side_effect=RuntimeError("Chroma DB desconectado"))

    response = service.retrieve_for_evaluation(
        case_id="CASE-01",
        query="inteligencia",
        top_k=3
    )

    assert response["status"] == "error"
    assert response["results"] == []
    assert response["error"]["code"] == "RETRIEVAL_FAILED"
    assert "Chroma DB desconectado" in response["error"]["message"]


def test_retriever_service_validation_empty_query(temp_vector_store):
    service = RetrieverService(temp_vector_store)

    resp1 = service.retrieve(query="")
    assert resp1["status"] == "error"
    assert resp1["error"]["code"] == "INVALID_QUERY"

    resp2 = service.retrieve(query="    ")
    assert resp2["status"] == "error"
    assert resp2["error"]["code"] == "INVALID_QUERY"


def test_retriever_service_validation_empty_case_id_in_evaluation(temp_vector_store):
    service = RetrieverService(temp_vector_store)

    resp1 = service.retrieve_for_evaluation(case_id="", query="hola")
    assert resp1["status"] == "error"
    assert resp1["error"]["code"] == "INVALID_CASE_ID"

    resp2 = service.retrieve_for_evaluation(case_id="   ", query="hola")
    assert resp2["status"] == "error"
    assert resp2["error"]["code"] == "INVALID_CASE_ID"


def test_retriever_service_validation_top_k_less_equal_zero(temp_vector_store):
    service = RetrieverService(temp_vector_store)

    resp1 = service.retrieve(query="test", top_k=0)
    assert resp1["status"] == "error"
    assert resp1["error"]["code"] == "INVALID_TOP_K"

    resp2 = service.retrieve(query="test", top_k=-5)
    assert resp2["status"] == "error"
    assert resp2["error"]["code"] == "INVALID_TOP_K"


def test_retriever_service_filter_by_document_id(temp_vector_store):
    chunks = [
        Chunk(id="DOC1_1", text="Texto A", metadata={"document_id": "DOC1"}),
        Chunk(id="DOC2_1", text="Texto B", metadata={"document_id": "DOC2"})
    ]
    temp_vector_store.add_chunks(chunks)

    service = RetrieverService(temp_vector_store)
    response = service.retrieve(
        query="Texto",
        top_k=5,
        document_id="DOC2"
    )

    assert response["status"] == "success"
    assert len(response["results"]) == 1
    assert response["results"][0]["document_id"] == "DOC2"
