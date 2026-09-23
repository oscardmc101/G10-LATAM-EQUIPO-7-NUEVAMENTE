import pytest
from agentes.rag.pipeline import ingest_file
from agentes.rag.retriever import RetrieverService


def test_pipeline_ingest_and_retrieve(tmp_path, temp_vector_store):
    sample_file = tmp_path / "guia_nube.md"
    sample_file.write_text(
        "# Arquitectura Cloud\n\n"
        "Una Virtual Cloud Network (VCN) es una red virtual personalizable y privada.\n"
        "Permite aislar instancias informáticas en subredes seguras.\n",
        encoding="utf-8"
    )

    result = ingest_file(
        path=str(sample_file),
        vector_store=temp_vector_store,
        document_id="CLD-ES-001"
    )

    assert result["document_id"] == "CLD-ES-001"
    assert result["documents"] == 1
    assert result["chunks"] >= 1

    # Recuperar mediante RetrieverService
    service = RetrieverService(temp_vector_store)
    response = service.retrieve(
        query="Virtual Cloud Network VCN",
        top_k=3,
        document_id="CLD-ES-001"
    )

    assert response["status"] == "success"
    assert len(response["results"]) >= 1
    assert response["results"][0]["document_id"] == "CLD-ES-001"
    assert "VCN" in response["results"][0]["text"]


def test_pipeline_ingest_missing_document_id(tmp_path, temp_vector_store):
    sample_file = tmp_path / "test.txt"
    sample_file.write_text("Texto", encoding="utf-8")

    with pytest.raises(ValueError, match="document_id"):
        ingest_file(str(sample_file), temp_vector_store, document_id="")
