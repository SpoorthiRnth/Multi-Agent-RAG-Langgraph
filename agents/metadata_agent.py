"""
This agent is a document slevel metadata query agent.

Answers questions about what documents are available, their date ranges,
machine IDs covered, and document types without doing RAG.
"""

import json
import logging
from pathlib import Path
from langchain_core.messages import HumanMessage

from core.llm_factory import get_llm
from core.config import config

logger = logging.getLogger(__name__)

METADATA_PROMPT = """You are a document registry assistant for a manufacturing document system.

Here is a summary of all indexed documents:
{registry}

User question: {query}

Answer based only on the document registry above. Be concise and precise.
"""


def _load_registry() -> dict:
    path = Path(config.ingest.metadata_registry_path)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"documents": [], "note": "No documents indexed yet. Run ingestion first."}


def metadata_agent_node(state: dict) -> dict:
    query = state["query"]
    logger.info("Metadata agent: answering document-level query")

    registry = _load_registry()
    registry_str = json.dumps(registry, indent=2)

    llm = get_llm()
    prompt = METADATA_PROMPT.format(registry=registry_str, query=query)
    response = llm.invoke([HumanMessage(content=prompt)])
    agent_answer = response.content.strip()

    trace = {
        "agent": "metadata_agent",
        "status": "success",
        "documents_in_registry": len(registry.get("documents", [])),
        "agent_answer_preview": agent_answer[:200],
    }

    return {
        **state,
        "metadata_results": registry,
        "messages": state["messages"] + [HumanMessage(content=agent_answer, name="metadata_agent")],
        "agent_trace": state.get("agent_trace", []) + [trace],
    }
