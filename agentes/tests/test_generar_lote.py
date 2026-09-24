import json
from pathlib import Path
import pytest

from agentes.generar_lote import generar_lote_resultados


def test_generar_lote_con_fake_embedding(tmp_path, fake_embedding):
    """
    Valida la ejecución del generador de lote usando las rutas oficiales de Data_IA,
    FakeEmbedding determinista y un output temporal en tmp_path.
    """
    output_file = tmp_path / "lote_evaluacion_test.json"

    # Ejecutar para los primeros 2 casos reales del Ground Truth
    results = generar_lote_resultados(
        output_path=output_file,
        embedding_service=fake_embedding,
        limit=2,
        top_k=5
    )

    assert len(results) == 2
    assert output_file.exists(), "El archivo de salida debe haberse generado en tmp_path."

    # Verificar estructura contractual de cada resultado en memoria
    for item in results:
        assert item["contract_version"] == "1.0"
        assert item["score_type"] == "cosine_similarity"
        assert item["top_k"] == 5
        assert item["status"] in ("success", "no_results")
        assert item["error"] is None
        assert isinstance(item["case_id"], str) and len(item["case_id"]) > 0
        assert isinstance(item["query"], str) and len(item["query"]) > 0
        assert isinstance(item["results"], list)

        if item["status"] == "success":
            assert len(item["results"]) > 0
            first = item["results"][0]
            assert first["rank"] == 1
            assert "chunk_id" in first
            assert "document_id" in first
            assert "score" in first
            assert "text" in first

    # Verificar que el archivo JSON escrito en disco coincide con los resultados
    with open(output_file, mode="r", encoding="utf-8") as f:
        loaded_json = json.load(f)

    assert len(loaded_json) == 2
    assert loaded_json[0]["case_id"] == results[0]["case_id"]
    assert loaded_json[0]["error"] is None


def test_generar_lote_missing_csv_raises():
    """Valida que falle explícitamente si alguna ruta oficial no existe."""
    with pytest.raises(FileNotFoundError):
        generar_lote_resultados(chunks_csv_path="ruta/inexistente/chunks.csv")

    with pytest.raises(FileNotFoundError):
        generar_lote_resultados(ground_truth_csv_path="ruta/inexistente/gt.csv")
