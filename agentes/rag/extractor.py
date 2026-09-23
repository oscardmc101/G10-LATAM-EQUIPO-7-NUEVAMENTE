from pathlib import Path

from .models import Document

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


def extract_document(path: str, document_id: str) -> list[Document]:
    """
    Extrae contenido de un archivo (.pdf, .md, .txt) asociando explícitamente el document_id canónico.
    """
    if not document_id or not str(document_id).strip():
        raise ValueError("document_id es obligatorio y no puede estar vacío.")

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(file_path, document_id)
    elif suffix in (".md", ".txt"):
        return _extract_plain_text(file_path, suffix, document_id)
    else:
        raise ValueError(f"Formato no soportado: {suffix}")


def _extract_pdf(file_path: Path, document_id: str) -> list[Document]:
    if PdfReader is None:
        raise ImportError(
            "Falta la librería 'pypdf'. Instálala con: pip install pypdf"
        )

    reader = PdfReader(str(file_path))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        documents.append(
            Document(
                text=text,
                metadata={
                    "document_id": document_id,
                    "source": file_path.name,
                    "page": page_number,
                    "file_type": "pdf"
                }
            )
        )

    return documents


def _extract_plain_text(
    file_path: Path,
    suffix: str,
    document_id: str
) -> list[Document]:
    text = file_path.read_text(encoding="utf-8")
    file_type = "markdown" if suffix == ".md" else "text"

    return [
        Document(
            text=text,
            metadata={
                "document_id": document_id,
                "source": file_path.name,
                "page": 1,
                "file_type": file_type
            }
        )
    ]