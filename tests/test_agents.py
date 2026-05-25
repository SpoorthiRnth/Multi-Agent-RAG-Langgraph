"""
Unit tests for agent logic.
Uses mocked LLM to avoid real API calls in CI.
"""

import pytest
from unittest.mock import MagicMock, patch


def make_state(query="test query", extra=None):
    state = {
        "messages": [],
        "query": query,
        "route": "",
        "retrieval_results": [],
        "table_results": [],
        "metadata_results": {},
        "final_answer": "",
        "agent_trace": [],
        "_vector_store": None,
        "_tables": [],
    }
    if extra:
        state.update(extra)
    return state


def mock_llm_response(text: str):
    llm = MagicMock()
    response = MagicMock()
    response.content = text
    llm.invoke.return_value = response
    return llm


class TestSupervisorNode:
    @patch("agents.supervisor.get_llm")
    def test_routes_to_retrieval(self, mock_get_llm):
        mock_get_llm.return_value = mock_llm_response("retrieval")
        from agents.supervisor import supervisor_node
        state = make_state(query="Summarize the maintenance log for M-204")
        result = supervisor_node(state)
        assert result["route"] == "retrieval"
        assert len(result["agent_trace"]) == 1
        assert result["agent_trace"][0]["agent"] == "supervisor"

    @patch("agents.supervisor.get_llm")
    def test_routes_to_table(self, mock_get_llm):
        mock_get_llm.return_value = mock_llm_response("table")
        from agents.supervisor import supervisor_node
        state = make_state(query="What is the average downtime per machine?")
        result = supervisor_node(state)
        assert result["route"] == "table"

    @patch("agents.supervisor.get_llm")
    def test_invalid_route_defaults_to_retrieval(self, mock_get_llm):
        mock_get_llm.return_value = mock_llm_response("something_unexpected")
        from agents.supervisor import supervisor_node
        state = make_state(query="What happened?")
        result = supervisor_node(state)
        assert result["route"] == "retrieval"

    def test_router_mapping(self):
        from agents.supervisor import router
        for route, expected in [
            ("retrieval", "retrieval_agent"),
            ("table", "table_agent"),
            ("metadata", "metadata_agent"),
            ("multi", "multi_path"),
        ]:
            state = make_state()
            state["route"] = route
            assert router(state) == expected


class TestRetrievalAgent:
    def test_empty_vector_store_returns_gracefully(self):
        from agents.retrieval_agent import retrieval_agent_node
        mock_store = MagicMock()
        mock_store.is_empty.return_value = True
        state = make_state("What is the MTBF?", {"_vector_store": mock_store})
        result = retrieval_agent_node(state)
        assert result["retrieval_results"] == []
        assert any(t["agent"] == "retrieval_agent" for t in result["agent_trace"])

    def test_no_vector_store_in_state(self):
        from agents.retrieval_agent import retrieval_agent_node
        state = make_state("test")
        result = retrieval_agent_node(state)
        assert result["retrieval_results"] == []

    @patch("agents.retrieval_agent.get_llm")
    def test_successful_retrieval(self, mock_get_llm):
        mock_get_llm.return_value = mock_llm_response("M-204 had 6 incidents.")
        from agents.retrieval_agent import retrieval_agent_node

        mock_store = MagicMock()
        mock_store.is_empty.return_value = False
        mock_store.search.return_value = [
            {"content": "M-204 hydraulic press had 6 incidents", "source": "report.txt",
             "page": 1, "doc_type": "txt", "chunk_id": "rep_1", "score": 0.2}
        ]
        state = make_state("How many incidents did M-204 have?", {"_vector_store": mock_store})
        result = retrieval_agent_node(state)
        assert len(result["retrieval_results"]) == 1
        assert result["retrieval_results"][0]["source"] == "report.txt"


class TestTableAgent:
    def test_no_tables_returns_gracefully(self):
        from agents.table_agent import table_agent_node
        state = make_state("Average downtime?")
        result = table_agent_node(state)
        assert result["table_results"] == []

    @patch("agents.table_agent.get_llm")
    def test_with_tables(self, mock_get_llm):
        import json, pandas as pd
        mock_get_llm.return_value = mock_llm_response("M-204 has highest downtime: 64h")
        from agents.table_agent import table_agent_node

        df = pd.DataFrame({"machine": ["M-101", "M-204"], "downtime": [8, 64]})
        tables = [{
            "source": "sensor_data.csv",
            "table_index": 0,
            "dataframe_json": df.to_json(orient="records"),
            "description": "Downtime summary",
        }]
        state = make_state("Which machine had the highest downtime?", {"_tables": tables})
        result = table_agent_node(state)
        assert len(result["table_results"]) == 1


class TestSynthesisAgent:
    @patch("agents.synthesis_agent.get_llm")
    def test_synthesises_from_agent_messages(self, mock_get_llm):
        from langchain_core.messages import HumanMessage
        mock_get_llm.return_value = mock_llm_response("Final: M-204 had 6 incidents and 64h downtime.")
        from agents.synthesis_agent import synthesis_agent_node

        state = make_state("Tell me about M-204")
        state["messages"] = [
            HumanMessage(content="M-204 had 6 incidents", name="retrieval_agent"),
            HumanMessage(content="Total downtime: 64 hours", name="table_agent"),
        ]
        result = synthesis_agent_node(state)
        assert "M-204" in result["final_answer"]
        assert result["final_answer"] != ""

    def test_no_agent_messages_returns_fallback(self):
        from agents.synthesis_agent import synthesis_agent_node
        state = make_state("test")
        result = synthesis_agent_node(state)
        assert "unable" in result["final_answer"].lower() or result["final_answer"] != ""


class TestDocumentLoader:
    def test_validates_short_chunks(self):
        from core.document_loader import DocumentLoader, DocumentChunk
        loader = DocumentLoader()
        short_chunk = DocumentChunk(
            content="too short",
            source="test.txt", page=None, doc_type="txt",
            chunk_id="test_c0", metadata={}
        )
        valid_chunk = DocumentChunk(
            content="This is a long enough chunk with more than ten words total here",
            source="test.txt", page=None, doc_type="txt",
            chunk_id="test_c1", metadata={}
        )
        result = loader._validate_chunks([short_chunk, valid_chunk])
        assert len(result) == 1
        assert result[0].chunk_id == "test_c1"

    def test_split_text_overlap(self):
        from core.document_loader import DocumentLoader
        loader = DocumentLoader(chunk_size=10, chunk_overlap=2)
        text = " ".join([f"word{i}" for i in range(30)])
        chunks = loader._split_text(text)
        assert len(chunks) > 1
        # Verify overlap: last words of chunk N appear in chunk N+1
        if len(chunks) >= 2:
            chunk0_words = set(chunks[0].split())
            chunk1_words = set(chunks[1].split())
            assert len(chunk0_words & chunk1_words) > 0
