"""
Módulo puro del contrato Retrieval v1 acordado con el equipo de Data/IA.
Define la versión, métrica de score y builders estandarizados para las respuestas.
"""
from typing import Any, Optional

CONTRACT_VERSION = "1.0"
SCORE_TYPE = "cosine_similarity"


def build_success_response(
    query: str,
    top_k: int,
    results: list[dict[str, Any]],
    case_id: Optional[str] = None,
    score_type: str = SCORE_TYPE,
    contract_version: str = CONTRACT_VERSION
) -> dict[str, Any]:
    """
    Construye la respuesta contractual para una búsqueda exitosa con resultados.
    Garantiza status='success', results con los chunks ordenados y error=None.
    """
    return {
        "contract_version": contract_version,
        "case_id": case_id,
        "query": query,
        "top_k": top_k,
        "score_type": score_type,
        "status": "success",
        "results": results,
        "error": None
    }


def build_no_results_response(
    query: str,
    top_k: int,
    case_id: Optional[str] = None,
    score_type: str = SCORE_TYPE,
    contract_version: str = CONTRACT_VERSION
) -> dict[str, Any]:
    """
    Construye la respuesta contractual para una búsqueda sin coincidencias.
    Garantiza status='no_results', results=[] y error=None.
    """
    return {
        "contract_version": contract_version,
        "case_id": case_id,
        "query": query,
        "top_k": top_k,
        "score_type": score_type,
        "status": "no_results",
        "results": [],
        "error": None
    }


def build_error_response(
    query: str,
    top_k: int,
    error_code: str,
    error_message: str,
    case_id: Optional[str] = None,
    score_type: str = SCORE_TYPE,
    contract_version: str = CONTRACT_VERSION
) -> dict[str, Any]:
    """
    Construye la respuesta contractual para un fallo de validación o error técnico.
    Garantiza status='error', results=[] y un objeto de error con code y message.
    """
    return {
        "contract_version": contract_version,
        "case_id": case_id,
        "query": query,
        "top_k": top_k,
        "score_type": score_type,
        "status": "error",
        "results": [],
        "error": {
            "code": error_code,
            "message": error_message
        }
    }
