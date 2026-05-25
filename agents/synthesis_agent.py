"""
This is an answer synthesis and citation agent.

Merges outputs from all activated agents into a coherent,
well-structured final answer with source citations.
"""

import logging
from langchain_core.messages import HumanMessage, AIMessage

from core.llm_factory import get_llm

logger = logging.getLogger(__name__)

SYNTHESIS_PROMPT = """You are the final synthesis agent in a manufacturing document intelligence system.

You have received outputs from one or more specialist agents. Your job is to:
1. Combine their findings into a single, coherent, well-structured answer
2. Cite sources where relevant (mention document names, page numbers if available)
3. Be direct and precise — avoid repetition
4. If agent outputs conflict, note the discrepancy

Original user question: {query}

Agent outputs:
{agent_outputs}

Retrieved source chunks (for citation):
{source_chunks}

Write the final answer now:
"""


def synthesis_agent_node(state: dict) -> dict:
    
    # Collects all agent answers from the message history and synthesises them into a single final response.
    
    query = state["query"]
    messages = state.get("messages", [])

    # Collect agent outputs (messages with name attribute = from agents)
    agent_outputs = []
    for msg in messages:
        if hasattr(msg, "name") and msg.name and msg.name.endswith("_agent"):
            agent_outputs.append(f"[{msg.name}]: {msg.content}")

    # Build source chunk citations
    retrieval_results = state.get("retrieval_results", [])
    source_chunks = "\n\n".join(
        f"- {r['source']} (page {r.get('page', 'N/A')}): {r['content'][:300]}..."
        for r in retrieval_results[:4]
    ) or "No text chunks retrieved."

    if not agent_outputs:
        logger.warning("Synthesis agent: no agent outputs found")
        final_answer = "I was unable to find relevant information for your question. Please ensure documents are indexed."
    else:
        agent_outputs_str = "\n\n".join(agent_outputs)
        llm = get_llm()
        prompt = SYNTHESIS_PROMPT.format(
            query=query,
            agent_outputs=agent_outputs_str,
            source_chunks=source_chunks,
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        final_answer = response.content.strip()

    trace = {
        "agent": "synthesis_agent",
        "status": "success",
        "agents_combined": len(agent_outputs),
        "answer_length": len(final_answer),
    }
    logger.info(f"Synthesis complete: {len(final_answer)} chars from {len(agent_outputs)} agent output(s)")

    return {
        **state,
        "final_answer": final_answer,
        "messages": state["messages"] + [AIMessage(content=final_answer)],
        "agent_trace": state.get("agent_trace", []) + [trace],
    }
