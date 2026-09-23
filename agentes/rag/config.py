from dataclasses import dataclass


@dataclass(frozen=True)
class RAGConfig:
    """Configuración centralizada para el módulo RAG."""
    chunk_size: int = 900
    chunk_overlap: int = 150
    default_top_k: int = 5
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    vector_store_path: str = "./chroma_db"
    collection_name: str = "knowledge_base"
    score_type: str = "cosine_similarity"
    hnsw_space: str = "cosine"


DEFAULT_RAG_CONFIG = RAGConfig()
