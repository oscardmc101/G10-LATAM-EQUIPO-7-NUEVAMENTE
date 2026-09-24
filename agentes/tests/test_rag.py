import csv
from pathlib import Path
import pytest

from agentes import AgentV1
from agentes.rag import (
    RetrieverService,
    VectorStore,
    load_evaluation_chunks
)


import shutil
import tempfile

@pytest.fixture(scope="module")
def loaded_rag_agent(fake_embedding):
    """
    Fixture de nivel módulo que realiza la carga real de chunks_v1.csv una sola vez
    para optimizar la velocidad de la suite de pruebas. Usa tempfile.mkdtemp para
    compatibilidad total con Windows y evitar errores de enlaces simbólicos.
    """
    temp_dir = tempfile.mkdtemp(prefix="rag_e2e_chroma_")
    vector_store = VectorStore(
        path=temp_dir,
        collection_name="evaluation_kb",
        embedding_service=fake_embedding
    )

    repo_root = Path(__file__).resolve().parent.parent.parent
    csv_path = repo_root / "Data_IA" / "data" / "evaluation" / "chunks_v1.csv"
    assert csv_path.exists(), f"El archivo Ground Truth debe existir en: {csv_path}"

    loaded_count = load_evaluation_chunks(csv_path=csv_path, vector_store=vector_store)
    assert loaded_count > 0

    retriever_service = RetrieverService(vector_store)
    agent = AgentV1(retriever_service)

    yield {
        "agent": agent,
        "vector_store": vector_store,
        "csv_path": csv_path,
        "loaded_count": loaded_count
    }

    # Limpieza al finalizar el módulo
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_rag_e2e_instantiation_and_ground_truth_loading(loaded_rag_agent):
    """
    A & B: Valida importación pública, instanciación en 3 capas
    (AgentV1 -> RetrieverService -> VectorStore) y carga real de chunks_v1.csv.
    """
    agent = loaded_rag_agent["agent"]
    assert isinstance(agent, AgentV1)
    assert isinstance(agent.retriever, RetrieverService)
    assert isinstance(agent.retriever.vector_store, VectorStore)

    # Verificar que el total de chunks leídos coincide con las filas del CSV sin duplicados ni pérdida
    with open(loaded_rag_agent["csv_path"], mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert loaded_rag_agent["loaded_count"] == len(rows)

    # Verificar preservación exacta de IDs y textos en el VectorStore
    first_row = rows[0]
    stored = agent.retriever.vector_store.collection.get(ids=[first_row["chunk_id"]])
    assert stored["ids"][0] == first_row["chunk_id"]
    assert stored["documents"][0] == first_row["chunk_text"]
    assert stored["metadatas"][0]["document_id"] == first_row["document_id"]


def test_rag_e2e_answer_success(loaded_rag_agent):
    """
    C & E: Valida ejecución de agent.answer() produciendo status == 'success'.
    """
    agent = loaded_rag_agent["agent"]
    response = agent.answer(
        query="Introducción a los conceptos de inteligencia artificial",
        top_k=3
    )

    assert response["contract_version"] == "1.0"
    assert response["query"] == "Introducción a los conceptos de inteligencia artificial"
    assert response["top_k"] == 3
    assert response["score_type"] == "cosine_similarity"
    assert response["status"] == "success"
    assert response["error"] is None
    assert len(response["results"]) == 3

    first = response["results"][0]
    assert first["rank"] == 1
    assert "chunk_id" in first and len(first["chunk_id"]) > 0
    assert "document_id" in first and len(first["document_id"]) > 0
    assert isinstance(first["score"], float)
    assert len(first["text"]) > 0
    assert isinstance(first["metadata"], dict)


def test_rag_e2e_answer_for_evaluation_success(loaded_rag_agent):
    """
    D & E: Valida ejecución de agent.answer_for_evaluation() extrayendo dinámicamente
    un caso real desde Data_IA/data/evaluation/ground_truth_v1.csv.
    Produce status == 'success' con el contrato JSON acordado con Data/IA.
    """
    agent = loaded_rag_agent["agent"]
    repo_root = Path(__file__).resolve().parent.parent.parent
    gt_path = repo_root / "Data_IA" / "data" / "evaluation" / "ground_truth_v1.csv"
    assert gt_path.exists(), f"El archivo Ground Truth debe existir en: {gt_path}"

    with open(gt_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        gt_rows = list(reader)

    assert len(gt_rows) > 0
    first_case = gt_rows[0]
    case_id = first_case["case_id"]
    query = first_case["pregunta"]

    response = agent.answer_for_evaluation(
        case_id=case_id,
        query=query,
        top_k=5
    )

    assert response["contract_version"] == "1.0"
    assert response["case_id"] == case_id
    assert response["query"] == query
    assert response["top_k"] == 5
    assert response["score_type"] == "cosine_similarity"
    assert response["status"] == "success"
    assert response["error"] is None
    assert len(response["results"]) == 5

    for item in response["results"]:
        assert item["rank"] >= 1
        assert "chunk_id" in item and len(item["chunk_id"]) > 0
        assert "document_id" in item and len(item["document_id"]) > 0
        assert isinstance(item["score"], float)
        assert len(item["text"]) > 0


def test_rag_e2e_no_results_flow(loaded_rag_agent):
    """
    F: Valida flujo reproducible y controlado que produce status == 'no_results'
    al filtrar por un document_id inexistente.
    """
    agent = loaded_rag_agent["agent"]

    # Flujo operativo normal
    normal_res = agent.answer(
        query="inteligencia artificial",
        top_k=3,
        document_id="DOC_ID_COMPLETAMENTE_INEXISTENTE"
    )
    assert normal_res["status"] == "no_results"
    assert normal_res["results"] == []
    assert normal_res["error"] is None

    # Flujo de evaluación
    eval_res = agent.answer_for_evaluation(
        case_id="CASE-NO-RESULTS-01",
        query="inteligencia artificial",
        top_k=3,
        document_id="DOC_ID_COMPLETAMENTE_INEXISTENTE"
    )
    assert eval_res["status"] == "no_results"
    assert eval_res["results"] == []
    assert eval_res["error"] is None
