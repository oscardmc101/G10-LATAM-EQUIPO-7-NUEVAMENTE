import chromadb
from typing import Optional

from .config import DEFAULT_RAG_CONFIG
from .models import Chunk, SearchResult


class VectorStore:
    """
    Única implementación de VectorStore para el módulo Agentes.
    Almacena chunks con embeddings y metadatos usando ChromaDB con métrica coseno.
    """

    def __init__(
        self,
        path: str = DEFAULT_RAG_CONFIG.vector_store_path,
        collection_name: str = DEFAULT_RAG_CONFIG.collection_name,
        embedding_service=None
    ):
        if embedding_service is None:
            from .embeddings import MultilingualEmbedding
            self.embedding_service = MultilingualEmbedding(
                DEFAULT_RAG_CONFIG.embedding_model_name
            )
        else:
            self.embedding_service = embedding_service

        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": DEFAULT_RAG_CONFIG.hnsw_space}
        )

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 500):
        """
        Inserta o actualiza chunks en la base vectorial por lotes.
        Verifica el invariante de document_id en cada chunk.
        """
        if not chunks:
            return

        for chunk in chunks:
            doc_id = chunk.metadata.get("document_id")
            if not doc_id or not str(doc_id).strip():
                raise ValueError(
                    f"El chunk '{chunk.id}' no contiene un 'document_id' válido en su metadata."
                )

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.text for c in batch]
            ids = [c.id for c in batch]
            metadatas = [c.metadata for c in batch]
            embeddings = self.embedding_service.embed_documents(texts)

            self.collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_RAG_CONFIG.default_top_k,
        document_id: Optional[str] = None
    ) -> list[SearchResult]:
        """
        Ejecuta búsqueda semántica por similitud coseno.
        Permite filtrado opcional por document_id canónico.
        """
        if document_id is not None:
            if not str(document_id).strip():
                raise ValueError("document_id no puede ser una cadena vacía cuando se proporciona.")
            where_clause = {"document_id": document_id}
        else:
            where_clause = None

        embedding = self.embedding_service.embed_query(query)

        query_kwargs = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if where_clause:
            query_kwargs["where"] = where_clause

        results = self.collection.query(**query_kwargs)

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        output = []
        for i, chunk_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            # Con hnsw:space="cosine", distance = 1 - cos_sim, por lo que score = 1 - distance es cosine similarity
            score = 1.0 - distance
            doc_text = results["documents"][0][i]
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}

            output.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=doc_text,
                    score=score,
                    metadata=metadata
                )
            )

        return output