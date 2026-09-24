import hashlib
import math
import pytest
from agentes.rag.vector_store import VectorStore

# Mitigación de compatibilidad en Windows para enlaces simbólicos temporales
try:
    import _pytest.pathlib
    _orig_cleanup = _pytest.pathlib.cleanup_dead_symlinks

    def _safe_cleanup_dead_symlinks(root):
        try:
            _orig_cleanup(root)
        except (PermissionError, OSError):
            pass

    _pytest.pathlib.cleanup_dead_symlinks = _safe_cleanup_dead_symlinks
except Exception:
    pass


class FakeEmbeddingService:
    """
    Servicio de embeddings mock determinista (16 dimensiones, normalizado L2).
    Permite calcular distancias y similitudes coseno exactas sin conexión a internet
    ni descarga de modelos externos.
    """
    def __init__(self, dim: int = 16):
        self.dim = dim

    def _hash_to_vector(self, text: str) -> list[float]:
        vec = []
        for i in range(self.dim):
            h = hashlib.sha256(f"{i}_{text}".encode("utf-8")).digest()
            val = int.from_bytes(h[:4], "little", signed=True) / (2**31)
            vec.append(val)
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._hash_to_vector(query)


@pytest.fixture(scope="session")
def fake_embedding():
    return FakeEmbeddingService()


@pytest.fixture
def temp_vector_store(tmp_path, fake_embedding):
    store = VectorStore(
        path=str(tmp_path / "test_chroma_db"),
        collection_name="test_collection",
        embedding_service=fake_embedding
    )
    return store
