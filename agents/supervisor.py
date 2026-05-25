"""
This is aLangGraph supervisor agent.

Defines the multi-agent graph with a supervisor that classifies
incoming queries and routes them to specialized agents.
The supervisor uses an LLM to decide which agent(s) to activate based on the query type.
"""

import logging
from typing import Annotated, Literal, TypedDict
import operator

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from core.llm_factory import get_llm
from agents.retrieval_agent import retrieval_agent_node
from agents.table_agent import table_agent_node
from agents.metadata_agent import metadata_agent_node
from agents.synthesis_agent import synthesis_agent_node

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    # state passed between all graph nodes.
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    route: str # Which agent(s) to activate
    retrieval_results: list[dict] # Output from retrieval agent
    table_results: list[dict] # Output from table agent
    metadata_results: dict # Output from metadata agent
    final_answer: str # Synthesis agent output
    agent_trace: list[dict] # Trace for UI visualization


SUPERVISOR_PROMPT = """You are the supervisor of a document intelligence system for manufacturing maintenance reports.

Given a user's question, classify it into ONE of these routing categories:

"retrieval": Narrative/text questions: summaries, descriptions, maintenance logs, repair history, explanations
"table": Structured data questions: numbers, counts, averages, trends, anomalies, comparisons across machines
"metadata":  Document-level questions: what documents exist, date ranges, machine IDs, file names
"multi": Complex questions requiring both text and structured data (e.g. "Which machine had the most failures and what were the reported causes?")

Respond with ONLY the routing category. No explanation.

User question: {query}
"""

def supervisor_node(state: AgentState) -> AgentState:
    #Classifies the query and sets the routing decision.

    llm = get_llm()
    query = state["query"]
    logger.info(f"Supervisor routing query: {query!r}")

    prompt = SUPERVISOR_PROMPT.format(query=query)
    response = llm.invoke([HumanMessage(content=prompt)])
    route = response.content.strip().lower()

    # Normalise response
    valid_routes = {"retrieval", "table", "metadata", "multi"}
    if route not in valid_routes:
        logger.warning(f"Unexpected route '{route}' — defaulting to 'retrieval'")
        route = "retrieval"

    logger.info(f"Supervisor route: {route}")
    trace_entry = {"agent": "supervisor", "decision": route, "query": query}

    return {
        **state,
        "route": route,
        "agent_trace": state.get("agent_trace", []) + [trace_entry],
    }


def router(state: AgentState) -> Literal["retrieval_agent", "table_agent", "metadata_agent", "multi_path"]:
    # maps route to next node.

    route = state.get("route", "retrieval")
    mapping = {
        "retrieval": "retrieval_agent",
        "table": "table_agent",
        "metadata": "metadata_agent",
        "multi": "multi_path",
    }
    return mapping.get(route, "retrieval_agent")


def multi_path_node(state: AgentState) -> AgentState:
    """
    For 'multi' queries: calls retrieval + table agents sequentially,
    then passes combined results to synthesis.
    """
    logger.info("Multi-path: activating retrieval + table agents")
    state = retrieval_agent_node(state)
    state = table_agent_node(state)
    trace_entry = {"agent": "multi_path", "decision": "retrieval + table combined"}
    return {**state, "agent_trace": state.get("agent_trace", []) + [trace_entry]}


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("retrieval_agent", retrieval_agent_node)
    graph.add_node("table_agent", table_agent_node)
    graph.add_node("metadata_agent", metadata_agent_node)
    graph.add_node("multi_path", multi_path_node)
    graph.add_node("synthesis_agent", synthesis_agent_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Conditional routing from supervisor
    graph.add_conditional_edges(
        "supervisor",
        router,
        {
            "retrieval_agent": "retrieval_agent",
            "table_agent": "table_agent",
            "metadata_agent": "metadata_agent",
            "multi_path": "multi_path",
        },
    )

    # All specialist agents flow to synthesis
    for node in ["retrieval_agent", "table_agent", "metadata_agent", "multi_path"]:
        graph.add_edge(node, "synthesis_agent")

    graph.add_edge("synthesis_agent", END)

    return graph.compile()


# Singleton compiled graph
_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_query(query: str, vector_store, tables: list[dict] | None = None) -> dict:
    """
    Main entrypoint for running a user query through the agent graph.

    Returns:
        {
          "answer": str,
          "agent_trace": list[dict],
          "retrieval_results": list[dict],
          "table_results": list[dict],
        }
    """
    graph = get_graph()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "route": "",
        "retrieval_results": [],
        "table_results": [],
        "metadata_results": {},
        "final_answer": "",
        "agent_trace": [],
        # Inject dependencies via state (agents pick these up)
        "_vector_store": vector_store,
        "_tables": tables or [],
    }

    final_state = graph.invoke(initial_state)

    return {
        "answer": final_state.get("final_answer", "No answer generated."),
        "agent_trace": final_state.get("agent_trace", []),
        "retrieval_results": final_state.get("retrieval_results", []),
        "table_results": final_state.get("table_results", []),
    }
