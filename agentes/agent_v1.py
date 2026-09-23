from typing import Optional, Union

from .rag.config import DEFAULT_RAG_CONFIG
from .rag.retriever import RetrieverService
from .rag.vector_store import VectorStore


class AgentV1:
    """
    Agente V1 que coordina el acceso al conocimiento a través de RetrieverService,
    manteniéndose desacoplado de los detalles internos de VectorStore o modelos de embedding.
    """

    def __init__(self, retriever_or_store: Union[RetrieverService, VectorStore]):
        if isinstance(retriever_or_store, RetrieverService):
            self.retriever = retriever_or_store
        else:
            self.retriever = RetrieverService(vector_store=retriever_or_store)

    def answer(
        self,
        query: str,
        top_k: int = DEFAULT_RAG_CONFIG.default_top_k,
        document_id: Optional[str] = None
    ) -> dict:
        """
        Consulta operativa del agente.
        """
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
            document_id=document_id
        )

    def answer_for_evaluation(
        self,
        case_id: str,
        query: str,
        top_k: int = DEFAULT_RAG_CONFIG.default_top_k,
        document_id: Optional[str] = None
    ) -> dict:
        """
        Consulta para el flujo de evaluación con Data/IA.
        case_id es obligatorio.
        """
        return self.retriever.retrieve_for_evaluation(
            case_id=case_id,
            query=query,
            top_k=top_k,
            document_id=document_id
        )