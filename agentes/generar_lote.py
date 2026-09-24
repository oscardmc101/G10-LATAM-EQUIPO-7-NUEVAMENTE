"""
Generador de lote de resultados de retrieval para evaluación formal con Data/IA.
Consume las fuentes oficiales:
- Data_IA/data/evaluation/chunks_v1.csv
- Data_IA/data/evaluation/ground_truth_v1.csv
Garantiza el cumplimiento estricto del Retrieval Contract v1 sin mutaciones manuales.
"""
import csv
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Optional, Union

from .agent_v1 import AgentV1
from .rag.chunks_loader import load_evaluation_chunks
from .rag.retriever import RetrieverService
from .rag.vector_store import VectorStore

# Rutas canónicas resueltas desde la ubicación del archivo, independientes del cwd
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS_PATH = REPO_ROOT / "Data_IA" / "data" / "evaluation" / "chunks_v1.csv"
DEFAULT_GROUND_TRUTH_PATH = REPO_ROOT / "Data_IA" / "data" / "evaluation" / "ground_truth_v1.csv"


def generar_lote_resultados(
    output_path: Optional[Union[str, Path]] = None,
    embedding_service: Optional[Any] = None,
    vector_store_path: Optional[Union[str, Path]] = None,
    limit: Optional[int] = None,
    top_k: int = 5,
    chunks_csv_path: Optional[Union[str, Path]] = None,
    ground_truth_csv_path: Optional[Union[str, Path]] = None
) -> list[dict[str, Any]]:
    """
    Ejecuta el pipeline de evaluación por lote sobre el Ground Truth oficial:
    1. Carga chunks_v1.csv en VectorStore usando load_evaluation_chunks.
    2. Lee casos de prueba desde ground_truth_v1.csv.
    3. Ejecuta AgentV1.answer_for_evaluation() para cada caso.
    4. El payload cumple con Retrieval Contract v1 directamente desde RetrieverService / contract.py.
    5. Retorna la lista de resultados y opcionalmente la guarda en output_path.
    """
    chunks_file = Path(chunks_csv_path) if chunks_csv_path else DEFAULT_CHUNKS_PATH
    gt_file = Path(ground_truth_csv_path) if ground_truth_csv_path else DEFAULT_GROUND_TRUTH_PATH

    if not chunks_file.exists():
        raise FileNotFoundError(f"Archivo de chunks no encontrado en: {chunks_file}")
    if not gt_file.exists():
        raise FileNotFoundError(f"Archivo de Ground Truth no encontrado en: {gt_file}")

    # Si no se proporciona ruta de almacenamiento vectorial, se utiliza un directorio temporal
    is_temp_dir = False
    if vector_store_path is None:
        actual_vs_path = tempfile.mkdtemp(prefix="chroma_lote_")
        is_temp_dir = True
    else:
        actual_vs_path = str(vector_store_path)

    try:
        # Inicializar VectorStore e indexar los chunks oficiales
        vector_store = VectorStore(
            path=actual_vs_path,
            collection_name="batch_evaluation_kb",
            embedding_service=embedding_service
        )
        load_evaluation_chunks(csv_path=chunks_file, vector_store=vector_store)

        # Configurar Agente con RetrieverService
        retriever_service = RetrieverService(vector_store)
        agent = AgentV1(retriever_service)

        batch_results: list[dict[str, Any]] = []

        with open(gt_file, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if limit is not None and i >= limit:
                    break

                case_id = row.get("case_id")
                query = row.get("pregunta")

                if not case_id or not query:
                    continue

                # Ejecutar recuperación para evaluación (el payload ya incluye contract_version, error=None, etc.)
                response = agent.answer_for_evaluation(
                    case_id=case_id,
                    query=query,
                    top_k=top_k
                )
                batch_results.append(response)

        # Si se especificó output_path, exportar el JSON
        if output_path is not None:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, mode="w", encoding="utf-8") as f:
                json.dump(batch_results, f, ensure_ascii=False, indent=2)

        return batch_results

    finally:
        if is_temp_dir:
            shutil.rmtree(actual_vs_path, ignore_errors=True)


if __name__ == "__main__":
    default_output = REPO_ROOT / "agentes" / "retrieval_results_agentes_v1.json"
    print(f"Iniciando generación de lote hacia: {default_output}")
    generar_lote_resultados(output_path=default_output)
    print("Generación de lote finalizada con éxito.")
