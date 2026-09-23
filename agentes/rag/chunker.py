from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import DEFAULT_RAG_CONFIG
from .models import Chunk, Document


def create_chunks(
    documents: list[Document],
    chunk_size: int = DEFAULT_RAG_CONFIG.chunk_size,
    chunk_overlap: int = DEFAULT_RAG_CONFIG.chunk_overlap
) -> list[Chunk]:
    """
    Divide los documentos en chunks conservando estrictamente el document_id canónico en metadatos y en el id del chunk.
    Defaults tomados directamente de DEFAULT_RAG_CONFIG.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe ser menor que chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            ""
        ]
    )

    chunks = []

    for document in documents:
        doc_id = document.metadata.get("document_id")
        if not doc_id or not str(doc_id).strip():
            raise ValueError(
                "Invariante violado: El documento no contiene un 'document_id' válido en sus metadatos."
            )

        cleaned_text = splitter.split_text(document.text)

        for index, text in enumerate(cleaned_text):
            metadata = dict(document.metadata)
            metadata["chunk_index"] = index
            metadata["document_id"] = str(doc_id)

            page = metadata.get("page", 1)
            chunk_id = f"{doc_id}_{page}_{index}"

            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    metadata=metadata
                )
            )

    return chunks