"""
This is a RAG-based retrieval agent.

Performs dense vector search over the indexed document corpus
and returns relevant chunks for synthesis.
"""

import logging
from langchain_core.messages import HumanMessage

from core.llm_factory import get_llm

logger = logging.getLogger(__name__)

RETRIEVAL_PROMPT = """You are a document analysis assistant for manufacturing maintenance reports.

Using the following retrieved document excerpts, answer the user's question as accurately and concisely as possible.
If the excerpts don't contain the answer, say so clearly.

Retrieved excerpts:
{context}

User question: {query}

Provide a clear, structured answer based only on the excerpts above.
"""


def retrieval_agent_node(state: dict) -> dict:
    #retreives relevant chunks from the vextor store and geberated an answer for LLM
    
    query = state["query"]
    vector_store = state.get("_vector_store")

    if vector_store is None:
        logger.error("No vector store found in state")
        trace = {"agent": "retrieval_agent", "status": "error", "reason": "no vector store"}
        return {**state, "retrieval_results": [], "agent_trace": state.get("agent_trace", []) + [trace]}

    if vector_store.is_empty():
        logger.warning("Vector store is empty no documents indexed yet")
        trace = {"agent": "retrieval_agent", "status": "empty_store", "chunks_found": 0}
        return {**state, "retrieval_results": [], "agent_trace": state.get("agent_trace", []) + [trace]}

    logger.info(f"Retrieval agent: searching for '{query}'")
    results = vector_store.search(query, k=6)

    # Build context string with source citations
    context_parts = []
    for i, r in enumerate(results, 1):
        page_info = f", page {r['page']}" if r.get("page") else ""
        context_parts.append(
            f"[{i}] Source: {r['source']}{page_info}\n{r['content']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # LLM call
    llm = get_llm()
    prompt = RETRIEVAL_PROMPT.format(context=context, query=query)
    response = llm.invoke([HumanMessage(content=prompt)])
    agent_answer = response.content.strip()

    trace = {
        "agent": "retrieval_agent",
        "status": "success",
        "chunks_found": len(results),
        "sources": list({r["source"] for r in results}),
        "agent_answer_preview": agent_answer[:200],
    }
    logger.info(f"Retrieval agent: found {len(results)} chunks")

    return {
        **state,
        "retrieval_results": results,
        "messages": state["messages"] + [HumanMessage(content=agent_answer, name="retrieval_agent")],
        "agent_trace": state.get("agent_trace", []) + [trace],
    }
