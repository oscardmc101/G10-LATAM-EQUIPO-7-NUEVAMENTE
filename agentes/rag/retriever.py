from typing import Optional

from .config import DEFAULT_RAG_CONFIG
from .vector_store import VectorStore


class RetrieverService:
    """
    Servicio de recuperación desacoplado que implementa el contrato JSON
    acordado con el equipo de Data/IA y prepara la integración con Backend.
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
        case_id no es requerido en este flujo.
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
        base_payload = {
            "contract_version": "1.0",
            "case_id": case_id,
            "query": query,
            "top_k": top_k,
            "score_type": DEFAULT_RAG_CONFIG.score_type,
            "status": "error",
            "results": [],
            "error": None
        }

        # Validaciones de entrada
        if require_case_id:
            if not case_id or not str(case_id).strip():
                base_payload["error"] = {
                    "code": "INVALID_CASE_ID",
                    "message": "case_id es obligatorio y no puede estar vacío para evaluación."
                }
                return base_payload

        if not query or not str(query).strip():
            base_payload["error"] = {
                "code": "INVALID_QUERY",
                "message": "query no puede estar vacío o contener únicamente espacios."
            }
            return base_payload

        if top_k is None or top_k <= 0:
            base_payload["error"] = {
                "code": "INVALID_TOP_K",
                "message": "top_k debe ser un entero mayor a 0."
            }
            return base_payload

        if document_id is not None and not str(document_id).strip():
            base_payload["error"] = {
                "code": "INVALID_DOCUMENT_ID",
                "message": "document_id no puede ser una cadena vacía cuando se proporciona."
            }
            return base_payload

        try:
            raw_results = self.vector_store.search(
                query=query,
                top_k=top_k,
                document_id=document_id
            )

            if not raw_results:
                base_payload["status"] = "no_results"
                base_payload["results"] = []
                base_payload["error"] = None
                return base_payload

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

            base_payload["status"] = "success"
            base_payload["results"] = formatted_results
            base_payload["error"] = None
            return base_payload

        except Exception as e:
            base_payload["status"] = "error"
            base_payload["results"] = []
            base_payload["error"] = {
                "code": "RETRIEVAL_FAILED",
                "message": str(e)
            }
            return base_payload