from unittest.mock import MagicMock
from agentes.rag.contract import (
    CONTRACT_VERSION,
    SCORE_TYPE,
    build_error_response,
    build_no_results_response,
    build_success_response,
)
from agentes.rag.models import Chunk
from agentes.rag.retriever import RetrieverService


def test_build_success_response():
    """Valida que build_success_response cumpla estrictamente con el contrato."""
    sample_results = [
        {
            "rank": 1,
            "chunk_id": "CH-01",
            "document_id": "DOC-01",
            "score": 0.95,
            "text": "Texto relevante",
            "metadata": {"categoria": "IA"}
        }
    ]
    resp = build_success_response(
        case_id="CASE-01",
        query="¿Qué es IA?",
        top_k=5,
        results=sample_results
    )

    assert resp["contract_version"] == CONTRACT_VERSION
    assert resp["case_id"] == "CASE-01"
    assert resp["query"] == "¿Qué es IA?"
    assert resp["top_k"] == 5
    assert resp["score_type"] == SCORE_TYPE
    assert resp["status"] == "success"
    assert resp["results"] == sample_results
    assert resp["error"] is None


def test_build_no_results_response():
    """Valida que build_no_results_response genere status=no_results, results=[] y error=None."""
    resp = build_no_results_response(
        case_id="CASE-02",
        query="consulta sin matches",
        top_k=3
    )

    assert resp["contract_version"] == CONTRACT_VERSION
    assert resp["case_id"] == "CASE-02"
    assert resp["query"] == "consulta sin matches"
    assert resp["top_k"] == 3
    assert resp["score_type"] == SCORE_TYPE
    assert resp["status"] == "no_results"
    assert resp["results"] == []
    assert resp["error"] is None


def test_build_error_response():
    """Valida que build_error_response genere status=error, results=[] y objeto error con code y message."""
    resp = build_error_response(
        case_id="CASE-03",
        query="consulta inválida",
        top_k=5,
        error_code="INVALID_QUERY",
        error_message="El query no puede estar vacío."
    )

    assert resp["contract_version"] == CONTRACT_VERSION
    assert resp["case_id"] == "CASE-03"
    assert resp["query"] == "consulta inválida"
    assert resp["top_k"] == 5
    assert resp["score_type"] == SCORE_TYPE
    assert resp["status"] == "error"
    assert resp["results"] == []
    assert resp["error"] == {
        "code": "INVALID_QUERY",
        "message": "El query no puede estar vacío."
    }


def test_retriever_service_retrieve_for_evaluation_contract_compliance(temp_vector_store):
    """
    Valida que RetrieverService.retrieve_for_evaluation genere respuestas que cumplan
    las reglas de error=None en success y no_results, y objeto error en error.
    """
    chunks = [
        Chunk(
            id="AI-01_1",
            text="Texto de prueba sobre algoritmos de búsqueda",
            metadata={"document_id": "AI-01"}
        )
    ]
    temp_vector_store.add_chunks(chunks)
    service = RetrieverService(temp_vector_store)

    # 1. Success -> error is None
    success_resp = service.retrieve_for_evaluation(
        case_id="AI-CASE-01",
        query="algoritmos de búsqueda",
        top_k=1
    )
    assert success_resp["status"] == "success"
    assert len(success_resp["results"]) == 1
    assert success_resp["error"] is None

    # 2. No results -> error is None
    no_results_resp = service.retrieve_for_evaluation(
        case_id="AI-CASE-02",
        query="algoritmos",
        top_k=1,
        document_id="DOC_INEXISTENTE"
    )
    assert no_results_resp["status"] == "no_results"
    assert no_results_resp["results"] == []
    assert no_results_resp["error"] is None

    # 3. Error de validación -> objeto error
    val_err_resp = service.retrieve_for_evaluation(
        case_id="",
        query="algoritmos",
        top_k=1
    )
    assert val_err_resp["status"] == "error"
    assert val_err_resp["results"] == []
    assert val_err_resp["error"]["code"] == "INVALID_CASE_ID"

    # 4. Error técnico de ejecución -> objeto error
    service.vector_store.search = MagicMock(side_effect=RuntimeError("Fallo de conexión"))
    tech_err_resp = service.retrieve_for_evaluation(
        case_id="AI-CASE-04",
        query="algoritmos",
        top_k=1
    )
    assert tech_err_resp["status"] == "error"
    assert tech_err_resp["results"] == []
    assert tech_err_resp["error"]["code"] == "RETRIEVAL_FAILED"
    assert "Fallo de conexión" in tech_err_resp["error"]["message"]
