import csv
from pathlib import Path
import pytest
from agentes.rag.chunks_loader import load_evaluation_chunks


def test_load_evaluation_chunks_from_real_ground_truth_csv(temp_vector_store):
    # Localizar el archivo Ground Truth acordado con Data/IA
    repo_root = Path(__file__).resolve().parent.parent.parent
    csv_path = repo_root / "Data_IA" / "data" / "evaluation" / "chunks_v1.csv"

    assert csv_path.exists(), f"El archivo debe existir en: {csv_path}"

    # Derivar dinámicamente la cantidad de filas sin hardcodear
    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        sample_rows = list(reader)

    expected_total = len(sample_rows)
    assert expected_total > 0

    # Cargar chunks en el VectorStore temporal
    loaded_count = load_evaluation_chunks(csv_path=csv_path, vector_store=temp_vector_store)
    assert loaded_count == expected_total

    # Verificar preservación EXACTA del primer chunk
    first_expected = sample_rows[0]
    res_first = temp_vector_store.collection.get(ids=[first_expected["chunk_id"]])
    assert res_first["ids"][0] == first_expected["chunk_id"]
    assert res_first["documents"][0] == first_expected["chunk_text"]
    assert res_first["metadatas"][0]["document_id"] == first_expected["document_id"]
    assert res_first["metadatas"][0]["chunk_index"] == int(first_expected["chunk_index"])

    # Verificar preservación EXACTA del último chunk
    last_expected = sample_rows[-1]
    res_last = temp_vector_store.collection.get(ids=[last_expected["chunk_id"]])
    assert res_last["ids"][0] == last_expected["chunk_id"]
    assert res_last["documents"][0] == last_expected["chunk_text"]
    assert res_last["metadatas"][0]["document_id"] == last_expected["document_id"]


def test_load_evaluation_chunks_missing_file(temp_vector_store):
    with pytest.raises(FileNotFoundError):
        load_evaluation_chunks("ruta/inexistente.csv", vector_store=temp_vector_store)


def test_load_evaluation_chunks_bom_handling(tmp_path, temp_vector_store):
    # Crear un CSV con BOM explícito (\ufeff)
    csv_with_bom = tmp_path / "bom_test.csv"
    content = (
        "\ufeffchunk_id,document_id,categoria,titulo_documento,chunk_index,char_count,chunk_text\n"
        "TEST_01,DOC_BOM,Test,Doc Test,1,10,Hola mundo\n"
    )
    csv_with_bom.write_text(content, encoding="utf-8")

    count = load_evaluation_chunks(csv_with_bom, temp_vector_store)
    assert count == 1

    stored = temp_vector_store.collection.get(ids=["TEST_01"])
    assert stored["ids"][0] == "TEST_01"
    assert stored["metadatas"][0]["document_id"] == "DOC_BOM"
