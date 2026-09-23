import pytest
from agentes.rag.extractor import extract_document


def test_extract_plain_text(tmp_path):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("Línea de prueba con contenido relevante.", encoding="utf-8")

    docs = extract_document(str(sample_file), document_id="DOC-TEST-01")
    assert len(docs) == 1
    assert docs[0].text == "Línea de prueba con contenido relevante."
    assert docs[0].metadata["document_id"] == "DOC-TEST-01"
    assert docs[0].document_id == "DOC-TEST-01"
    assert docs[0].metadata["source"] == "sample.txt"


def test_extract_markdown(tmp_path):
    sample_file = tmp_path / "sample.md"
    sample_file.write_text("# Encabezado Markdown\n\nTexto descriptivo.", encoding="utf-8")

    docs = extract_document(str(sample_file), document_id="DOC-MD-01")
    assert len(docs) == 1
    assert docs[0].metadata["file_type"] == "markdown"
    assert docs[0].document_id == "DOC-MD-01"


def test_extract_document_missing_document_id(tmp_path):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("Contenido", encoding="utf-8")

    with pytest.raises(ValueError, match="document_id"):
        extract_document(str(sample_file), document_id="")

    with pytest.raises(ValueError, match="document_id"):
        extract_document(str(sample_file), document_id="   ")


def test_extract_document_file_not_found():
    with pytest.raises(FileNotFoundError):
        extract_document("archivo_inexistente.txt", document_id="DOC-404")
