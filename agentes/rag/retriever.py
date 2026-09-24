from typing import Optional

from .config import DEFAULT_RAG_CONFIG
from .contract import (
    build_error_response,
    build_no_results_response,
    build_success_response
)
from .vector_store import VectorStore


class RetrieverService:
    """
    Servicio de recuperación desacoplado.
    Ejecuta la búsqueda semántica sobre VectorStore y delega la construcción
    del payload JSON al módulo contractual contract.py.
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_RAG_CONFIG.default_top_k,
        document_id: Optional[str] = None
    ) -> dict:
        """
        Recuperación para el flujo operativo normal.
        case_id no es requerido en este flujo (None).
        Permite filtrado opcional por document_id.
        """
        return self._execute_retrieval(
            case_id=None,
            query=query,
            top_k=top_k,
            document_id=document_id,
            require_case_id=False
        )

    def retrieve_for_evaluation(
        self,
        case_id: str,
        query: str,
        top_k: int = DEFAULT_RAG_CONFIG.default_top_k,
        document_id: Optional[str] = None
    ) -> dict:
        """
        Recuperación para el flujo de evaluación con Data/IA.
        case_id es obligatorio y no puede estar vacío.
        Devuelve un payload 100% compatible con Retrieval Contract v1.
        """
        return self._execute_retrieval(
            case_id=case_id,
            query=query,
            top_k=top_k,
            document_id=document_id,
            require_case_id=True
        )

    def _execute_retrieval(
        self,
        case_id: Optional[str],
        query: str,
        top_k: int,
        document_id: Optional[str],
        require_case_id: bool
    ) -> dict:
        # Validaciones de entrada
        if require_case_id:
            if not case_id or not str(case_id).strip():
                return build_error_response(
                    query=query,
                    top_k=top_k,
                    error_code="INVALID_CASE_ID",
                    error_message="case_id es obligatorio y no puede estar vacío para evaluación.",
                    case_id=case_id
                )

        if not query or not str(query).strip():
            return build_error_response(
                query=query,
                top_k=top_k,
                error_code="INVALID_QUERY",
                error_message="query no puede estar vacío o contener únicamente espacios.",
                case_id=case_id
            )

        if top_k is None or top_k <= 0:
            return build_error_response(
                query=query,
                top_k=top_k,
                error_code="INVALID_TOP_K",
                error_message="top_k debe ser un entero mayor a 0.",
                case_id=case_id
            )

        if document_id is not None and not str(document_id).strip():
            return build_error_response(
                query=query,
                top_k=top_k,
                error_code="INVALID_DOCUMENT_ID",
                error_message="document_id no puede ser una cadena vacía cuando se proporciona.",
                case_id=case_id
            )

        try:
            raw_results = self.vector_store.search(
                query=query,
                top_k=top_k,
                document_id=document_id
            )

            if not raw_results:
                return build_no_results_response(
                    query=query,
                    top_k=top_k,
                    case_id=case_id
                )

            formatted_results = []
            for rank, res in enumerate(raw_results, start=1):
                # document_id canónico estricto (falla si el chunk carece de document_id)
                canonical_doc_id = res.document_id

                formatted_results.append({
                    "rank": rank,
                    "chunk_id": res.chunk_id,
                    "document_id": canonical_doc_id,
                    "score": round(res.score, 4),
                    "text": res.text,
                    "metadata": res.metadata
                })

            return build_success_response(
                query=query,
                top_k=top_k,
                results=formatted_results,
                case_id=case_id
            )

        except Exception as e:
            return build_error_response(
                query=query,
                top_k=top_k,
                error_code="RETRIEVAL_FAILED",
                error_message=str(e),
                case_id=case_id
            )