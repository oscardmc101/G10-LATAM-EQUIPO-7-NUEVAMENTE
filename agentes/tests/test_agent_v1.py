from agentes.agent_v1 import AgentV1
from agentes.rag.models import Chunk
from agentes.rag.retriever import RetrieverService


def test_agent_v1_initialization_with_vector_store(temp_vector_store):
    agent = AgentV1(temp_vector_store)
    assert isinstance(agent.retriever, RetrieverService)
    assert agent.retriever.vector_store is temp_vector_store


def test_agent_v1_initialization_with_retriever_service(temp_vector_store):
    service = RetrieverService(temp_vector_store)
    agent = AgentV1(service)
    assert agent.retriever is service


def test_agent_v1_answer_flow(temp_vector_store):
    chunks = [
        Chunk(id="AI-01_1", text="Contenido sobre aprendizaje automático", metadata={"document_id": "AI-01"})
    ]
    temp_vector_store.add_chunks(chunks)

    agent = AgentV1(temp_vector_store)
    response = agent.answer(query="aprendizaje automático", top_k=1)

    assert response["status"] == "success"
    assert len(response["results"]) == 1
    assert response["results"][0]["document_id"] == "AI-01"


def test_agent_v1_answer_for_evaluation_flow(temp_vector_store):
    chunks = [
        Chunk(id="AI-01_1", text="Contenido sobre agentes y LLMs", metadata={"document_id": "AI-01"})
    ]
    temp_vector_store.add_chunks(chunks)

    agent = AgentV1(temp_vector_store)
    response = agent.answer_for_evaluation(
        case_id="EVAL-CASE-01",
        query="agentes y LLMs",
        top_k=1
    )

    assert response["contract_version"] == "1.0"
    assert response["case_id"] == "EVAL-CASE-01"
    assert response["status"] == "success"
    assert len(response["results"]) == 1
