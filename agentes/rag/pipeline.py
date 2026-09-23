from .chunker import create_chunks
from .cleaner import clean_text
from .extractor import extract_document
from .vector_store import VectorStore


def ingest_file(
    path: str,
    vector_store: VectorStore,
    document_id: str
) -> dict:
    """
    Coordina el pipeline de ingestión para un archivo físico:
    Extracción con document_id -> Limpieza conservadora -> Chunking -> VectorStore.
    """
    if not document_id or not str(document_id).strip():
        raise ValueError("document_id es obligatorio para la ingestión y no puede estar vacío.")

    documents = extract_document(path=path, document_id=document_id)

    for document in documents:
        document.text = clean_text(document.text)

    chunks = create_chunks(documents)

    vector_store.add_chunks(chunks)

    return {
        "document_id": document_id,
        "documents": len(documents),
        "chunks": len(chunks)
    }