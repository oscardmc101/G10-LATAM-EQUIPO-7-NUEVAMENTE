# NuevaMente — Módulo Agentes (Agent V1 + RAG Core)

Módulo de Ingestión, Almacenamiento Vectorial y Recuperación Semántica de Conocimiento para el Agente V1 de NuevaMente.

---

## 1. Arquitectura del Módulo

El flujo de recuperación sigue una arquitectura desacoplada en tres capas con contrato formal de integración:

```text
AgentV1 (agentes/agent_v1.py)
   ↓
RetrieverService (agentes/rag/retriever.py)  ──→  contract.py (Retrieval Contract v1)
   ↓
VectorStore (agentes/rag/vector_store.py) [ChromaDB — Cosine Similarity]
```

- **`AgentV1` (`agentes/agent_v1.py`)**: Fachada de alto nivel del agente. Expone los métodos `answer(...)` (operativo) y `answer_for_evaluation(...)` (evaluación formal). Desacoplado de la base vectorial y de los modelos de embedding.
- **`contract.py` (`agentes/rag/contract.py`)**: Módulo puro con la fuente única de verdad del contrato Retrieval v1 acordado con **Data/IA**. Define `CONTRACT_VERSION = "1.0"`, `SCORE_TYPE = "cosine_similarity"` y los builders `build_success_response`, `build_no_results_response` y `build_error_response`.
- **`RetrieverService` (`agentes/rag/retriever.py`)**: Servicio de recuperación que orquesta la búsqueda en `VectorStore`, valida parámetros y delega la construcción de respuestas a `contract.py`.
- **`VectorStore` (`agentes/rag/vector_store.py`)**: Única implementación de almacenamiento vectorial. Gestiona colecciones de ChromaDB configuradas con métrica coseno (`hnsw:space: cosine`), indexación por lotes y filtrado estricto por `document_id`.
- **`generar_lote.py` (`agentes/generar_lote.py`)**: Generador reproducible del lote de evaluación que consume directamente las fuentes oficiales de Data/IA y produce el JSON de evaluación.

---

## 2. Componentes Principales

```text
agentes/
├── README.md                  # Documentación real del módulo
├── requirements.txt           # Dependencias con versiones fijadas
├── agent_v1.py                # Clase principal AgentV1
├── generar_lote.py            # Generador de lote para evaluación formal
├── rag/
│   ├── __init__.py            # Exportaciones públicas del RAG
│   ├── config.py              # Configuración centralizada (RAGConfig)
│   ├── contract.py            # Contrato Retrieval v1 y builders de respuesta
│   ├── models.py              # Modelos: Document, Chunk, SearchResult
│   ├── cleaner.py             # Limpiador conservador (NFC, preserva indentación)
│   ├── chunker.py             # Recursive splitter con document_id canónico
│   ├── extractor.py           # Extractor de PDF, Markdown y texto plano
│   ├── embeddings.py          # Embeddings multilingües con SentenceTransformers
│   ├── vector_store.py        # VectorStore unificado en ChromaDB
│   ├── retriever.py           # RetrieverService coordinado con contract.py
│   ├── pipeline.py            # Pipeline de ingestión normal (ingest_file)
│   └── chunks_loader.py       # Cargador exacto de Ground Truth (chunks_v1.csv)
└── tests/                     # Suite de pruebas automatizadas (incluye test_rag.py)
```

---

## 3. Identificador Canónico `document_id`

`document_id` es un identificador canónico inmutable obligatorio provisto por el sistema (o entregado por Backend).

- **Sin fallbacks:** Se eliminó cualquier fallback a `source` o nombres de archivo.
- **Sin retornos vacíos engañosos:** Si un objeto o chunk carece de `document_id`, el sistema falla explícitamente (`KeyError` o `ValueError`).
- **Trazabilidad:** Cada chunk generado o indexado conserva `metadata["document_id"]` y genera su id como `{document_id}_{page}_{chunk_index}`.
- **Filtrado en Retrieval:** Tanto `VectorStore.search` como `RetrieverService.retrieve` permiten filtrar opcionalmente por `document_id`.

---

## 4. Fuentes Oficiales e Ingestión de Conocimiento

Las fuentes de datos oficiales provistas por el equipo de Data/IA residen en:
- `Data_IA/data/evaluation/chunks_v1.csv`
- `Data_IA/data/evaluation/ground_truth_v1.csv`

**No es necesario copiar estos archivos dentro de `agentes/`**; el código y los tests resuelven las rutas relativas a la raíz del repositorio de manera determinista.

### A. Ingestión Normal (`pipeline.py`)

Para documentos nuevos entregados por Backend:

```python
from agentes.rag import VectorStore, ingest_file

store = VectorStore()
resumen = ingest_file(
    path="ruta/al/documento.pdf",
    vector_store=store,
    document_id="CLD-ES-001"
)
# Retorna: {"document_id": "CLD-ES-001", "documents": 1, "chunks": 5}
```

El pipeline aplica:
1. `extractor.py`: Extrae el texto y valida la presencia de `document_id`.
2. `cleaner.py`: Limpieza conservadora con normalización Unicode **NFC**, saltos de línea LF y eliminación de caracteres de control inválidos, **preservando la indentación de código Python/YAML y dobles espacios de Markdown**.
3. `chunker.py`: Fragmenta mediante `RecursiveCharacterTextSplitter` asociando el `document_id` canónico.
4. `vector_store.py`: Indexa los chunks en lotes calculando embeddings.

### B. Ingestión para Evaluación (`chunks_loader.py`)

Para las pruebas de evaluación contra el Ground Truth consensuado con Data/IA (`Data_IA/data/evaluation/chunks_v1.csv`):

- **No se vuelve a chunkear** (no se utiliza splitter).
- **No se limpia destructivamente** el texto de ground truth.
- **Se preservan con total exactitud** `chunk_id`, `document_id`, `chunk_text`, `chunk_index` y metadatos.
- Se lee con `encoding="utf-8-sig"` para evitar problemas de BOM y con soporte RFC 4180 para campos multilínea.

```python
from agentes.rag import VectorStore, load_evaluation_chunks

store = VectorStore()
total_chunks = load_evaluation_chunks(
    csv_path="Data_IA/data/evaluation/chunks_v1.csv",
    vector_store=store
)
```

---

## 5. Similitud Coseno y Retrieval Contract v1

La colección de ChromaDB se inicializa explícitamente con:
```python
metadata={"hnsw:space": "cosine"}
```

En ChromaDB, la distancia devuelta para el espacio coseno es `distance = 1 - cos_sim`. Por ello, el score se calcula como:
```python
score = 1.0 - distance
```
lo que garantiza una similitud coseno matemáticamente válida.

### Respuestas JSON Construidas por `contract.py`

Las respuestas de `RetrieverService.retrieve_for_evaluation(case_id, query, top_k)` se construyen exclusivamente mediante `agentes.rag.contract` y cumplen con:

```json
{
  "contract_version": "1.0",
  "case_id": "CLD-ES-001-Q01",
  "query": "¿Qué es una VCN?",
  "top_k": 5,
  "score_type": "cosine_similarity",
  "status": "success",
  "results": [
    {
      "rank": 1,
      "chunk_id": "CLD-ES-001_CH_001",
      "document_id": "CLD-ES-001",
      "score": 0.8921,
      "text": "Una Virtual Cloud Network (VCN)...",
      "metadata": {
        "categoria": "Cloud/DevOps"
      }
    }
  ],
  "error": null
}
```

- En `status == "success"`: `error` es estrictamente `null` (None en Python).
- En `status == "no_results"`: `results` es `[]` y `error` es estrictamente `null` (None en Python).
- En `status == "error"`: `results` es `[]` y `error` contiene el objeto `{"code": "...", "message": "..."}`.

---

## 6. Generador de Lote para Evaluación (`generar_lote.py`)

Para ejecutar la evaluación batch completa de forma desacoplada y reproducible:

```python
from agentes.generar_lote import generar_lote_resultados

resultados = generar_lote_resultados(
    output_path="agentes/retrieval_results_agentes_v1.json"
)
```

El generador soporta inyección de `embedding_service`, `vector_store_path` y `limit` para pruebas deterministas sin descargas de modelos.

---

## 7. Configuración Centralizada

Toda la configuración se encuentra centralizada en `agentes/rag/config.py`:

```python
from agentes.rag.config import DEFAULT_RAG_CONFIG, RAGConfig

# Valores por defecto:
# chunk_size: 900
# chunk_overlap: 150
# default_top_k: 5
# embedding_model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# vector_store_path: "./chroma_db"
# collection_name: "knowledge_base"
# score_type: "cosine_similarity"
# hnsw_space: "cosine"
```

Los componentes consumen estos valores como defaults, evitando números mágicos dispersos.

---

## 8. Instalación y Dependencias

Instalar las dependencias fijadas del módulo:

```bash
pip install -r agentes/requirements.txt
```

---

## 9. Ejecución de Tests y Prueba Funcional End-to-End

La suite de pruebas automatizadas está basada en `pytest` y utiliza embeddings mock deterministas (no requiere descargar modelos ni conexión a internet).

Incluye pruebas unitarias para cada componente, validación de builders en `contract.py`, prueba del generador de lote y la prueba funcional completa del agente (`agentes/tests/test_rag.py`), que valida la instanciación de `AgentV1`, la carga real de `chunks_v1.csv` en `VectorStore`, la lectura de casos de `ground_truth_v1.csv` y los estados `success` y `no_results`.

Comando oficial de ejecución desde la raíz del repositorio:

```bash
python -m pytest agentes/tests -v
```

---

## 10. Contrato de Integración con Backend

- **Responsabilidad de Backend:** Obtener y descargar el archivo desde OCI Object Storage y llamar a Agentes entregando la ruta local del archivo junto con su `document_id` canónico.
- **Responsabilidad de Agentes:** Ingestar y buscar sobre el documento asociando el `document_id`. Agentes **no accede directamente a OCI** ni gestiona credenciales de infraestructura cloud.