import csv
from pathlib import Path
from typing import Union

from .models import Chunk
from .vector_store import VectorStore


def load_evaluation_chunks(
    csv_path: Union[str, Path],
    vector_store: VectorStore,
    batch_size: int = 500
) -> int:
    """
    Carga chunks precalculados desde el Ground Truth CSV (chunks_v1.csv) directamente a VectorStore.
    Reglas estrictas:
    - NO vuelve a chunkear (no se usa splitter).
    - NO modifica ni limpia destructivamente chunk_text.
    - NO regenera chunk_id ni altera document_id.
    - Usa encoding='utf-8-sig' para evitar desajustes por BOM en encabezados.
    - Soporta campos multilínea mediante csv.DictReader estándar (RFC 4180).
    - Retorna la cantidad dinámica total de chunks leídos e indexados.
    """
    file_path = Path(csv_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo de evaluación no encontrado: {file_path}")

    chunks: list[Chunk] = []

    with open(file_path, mode="r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            chunk_id = row.get("chunk_id")
            if not chunk_id:
                raise ValueError("Fila del CSV carece del campo obligatorio 'chunk_id'.")

            doc_id = row.get("document_id")
            if not doc_id:
                raise ValueError(f"El chunk '{chunk_id}' carece de 'document_id'.")

            chunk_text = row.get("chunk_text", "")

            # Preservar metadatos relevantes respetando tipos primitivos compatibles con ChromaDB
            metadata = {
                "document_id": str(doc_id),
                "categoria": str(row.get("categoria", "")),
                "titulo_documento": str(row.get("titulo_documento", "")),
                "chunk_index": int(row.get("chunk_index", 0)),
                "char_count": int(row.get("char_count", len(chunk_text)))
            }

            chunks.append(
                Chunk(
                    id=str(chunk_id),
                    text=chunk_text,
                    metadata=metadata
                )
            )

    if chunks:
        vector_store.add_chunks(chunks, batch_size=batch_size)

    return len(chunks)
