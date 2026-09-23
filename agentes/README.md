# NuevaMente — Módulo Agentes (Agent V1 + RAG Core)

Módulo de Ingestión, Almacenamiento Vectorial y Recuperación Semántica de Conocimiento para el Agente V1 de NuevaMente.

---

## 1. Arquitectura del Módulo

El flujo de recuperación sigue una arquitectura desacoplada en tres capas:

```text
AgentV1
   ↓
RetrieverService
   ↓
VectorStore (ChromaDB — Cosine Similarity)
```

- **`AgentV1` (`agentes/agent_v1.py`)**: Fachada de alto nivel del agente. Expone los métodos `answer(...)` (operativo) y `answer_for_evaluation(...)` (evaluación formal). No conoce los detalles internos del Vector Store ni de los embeddings.
- **`RetrieverService` (`agentes/rag/retriever.py`)**: Servicio intermedio que ejecuta la búsqueda, valida los parámetros de entrada y formatea la salida siguiendo el contrato JSON acordado con el equipo de **Data/IA**.
- **`VectorStore` (`agentes/rag/vector_store.py`)**: Única implementación de almacenamiento vectorial. Gestiona colecciones de ChromaDB configuradas con métrica coseno (`hnsw:space: cosine`), indexación por lotes y filtrado estricto por `document_id`.

---

## 2. Componentes Principales

```text
agentes/
├── README.md                  # Documentación real del módulo
├── requirements.txt           # Dependencias con versiones fijadas
├── agent_v1.py                # Clase principal AgentV1
├── rag/
│   ├── __init__.py            # Exportaciones públicas del RAG
│   ├── config.py              # Configuración centralizada (RAGConfig)
│   ├── models.py              # Document, Chunk, SearchResult
│   ├── cleaner.py             # Limpiador conservador (NFC, preserva indentación)
│   ├── chunker.py             # Recursive splitter con document_id canónico
│   ├── extractor.py           # Extractor de PDF, Markdown y texto plano
│   ├── embeddings.py          # Embeddings multilingües con SentenceTransformers
│   ├── vector_store.py        # VectorStore unificado en ChromaDB
│   ├── retriever.py           # RetrieverService con contrato Data/IA
│   ├── pipeline.py            # Pipeline de ingestión normal (ingest_file)
│   └── evaluation_loader.py   # Cargador exacto de Ground Truth (chunks_v1.csv)
└── tests/                     # Suite de pruebas automatizadas con pytest
```

---

## 3. Identificador Canónico `document_id`

`document_id` es un identificador canónico inmutable obligatorio provisto por el sistema (o entregado por Backend).

- **Sin fallbacks:** Se eliminó cualquier fallback a `source` o nombres de archivo.
- **Sin retornos vacíos engañosos:** Si un objeto o chunk carece de `document_id`, el sistema falla explícitamente (`KeyError` o `ValueError`).
- **Trazabilidad:** Cada chunk generado o indexado conserva `metadata["document_id"]` y genera su id como `{document_id}_{page}_{chunk_index}`.
- **Filtrado en Retrieval:** Tanto `VectorStore.search` como `RetrieverService.retrieve` permiten filtrar opcionalmente por `document_id`.

---

## 4. Ingestión de Conocimiento

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

### B. Ingestión para Evaluación (`evaluation_loader.py`)

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

## 5. Similitud Coseno y Contrato con Data/IA

La colección de ChromaDB se inicializa explícitamente con:
```python
metadata={"hnsw:space": "cosine"}
```

En ChromaDB, la distancia devuelta para el espacio coseno es `distance = 1 - cos_sim`. Por ello, el score se calcula como:
```python
score = 1.0 - distance
```
lo que garantiza una similitud coseno matemáticamente válida.

### Respuestas JSON de `RetrieverService`

`RetrieverService.retrieve_for_evaluation(case_id, query, top_k)` retorna el JSON estructurado:

```json
{
  "contract_version": "1.0",
  "case_id": "CLD-ES-001-Q01",
  "query": "¿Qué es una VCN?",
  "top_k": 3,
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

En caso de no haber resultados, `status` es `"no_results"` con lista vacía.
En caso de error técnico o validación fallida, `status` es `"error"` con detalles en el objeto `error` (`code` y `message`).

---

## 6. Configuración Centralizada

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

## 7. Instalación y Dependencias

Instalar las dependencias fijadas del módulo:

```bash
pip install -r agentes/requirements.txt
```

---

## 8. Ejecución de Tests

La suite de pruebas automatizadas está basada en `pytest` y utiliza embeddings mock deterministas (no requiere descargar modelos ni conexión a internet):

```bash
pytest agentes/tests -v
```

---

## 9. Contrato de Integración con Backend

- **Responsabilidad de Backend:** Obtener y descargar el archivo desde OCI Object Storage y llamar a Agentes entregando la ruta local del archivo junto con su `document_id` canónico.
- **Responsabilidad de Agentes:** Ingestar y buscar sobre el documento asociando el `document_id`. Agentes **no accede directamente a OCI** ni gestiona credenciales de infraestructura cloud.