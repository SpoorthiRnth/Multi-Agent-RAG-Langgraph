"""
This is a structured data and table reasoning agent.

Handles numeric/statistical questions over extracted tables and CSV data.
Uses pandas for computation + LLM for interpretation.
"""

import json
import logging
import pandas as pd
from langchain_core.messages import HumanMessage

from core.llm_factory import get_llm

logger = logging.getLogger(__name__)

TABLE_PROMPT = """You are a data analyst assistant for manufacturing operations.

The following structured data tables were extracted from maintenance reports and sensor datasets.
Answer the user's question using the data below. Be precise with numbers and cite the source table.

Available tables:
{tables_summary}

User question: {query}

Provide a data-driven answer. If you compute statistics, show your reasoning.
"""


def _summarise_tables(tables: list[dict]) -> str:
    # Convert table records into a readable summary for the LLM.
    summaries = []
    for t in tables:
        try:
            df = pd.read_json(t["dataframe_json"])
            # Include description + first rows + basic stats
            desc = t.get("description", "Unnamed table")
            shape = f"{len(df)} rows × {len(df.columns)} columns"
            cols = ", ".join(df.columns.tolist())
            preview = df.head(5).to_string(index=False)
            try:
                stats = df.describe(include="all").to_string()
            except Exception:
                stats = ""
            summaries.append(
                f"TABLE: {desc}\n"
                f"Shape: {shape}\n"
                f"Columns: {cols}\n"
                f"Preview (first 5 rows):\n{preview}\n"
                f"Statistics:\n{stats}"
            )
        except Exception as e:
            summaries.append(f"TABLE: {t.get('description', '?')} — could not parse: {e}")
    return "\n\n" + ("=" * 60 + "\n\n").join(summaries)


def table_agent_node(state: dict) -> dict:
    """
    Reasons over extracted tables and CSV data to answer
    structured/numeric questions.
    """
    query = state["query"]
    tables: list[dict] = state.get("_tables", [])

    if not tables:
        logger.info("Table agent: no tables available")
        trace = {"agent": "table_agent", "status": "no_tables", "tables_found": 0}
        return {**state, "table_results": [], "agent_trace": state.get("agent_trace", []) + [trace]}

    logger.info(f"Table agent: reasoning over {len(tables)} table(s)")
    tables_summary = _summarise_tables(tables)

    llm = get_llm()
    prompt = TABLE_PROMPT.format(tables_summary=tables_summary, query=query)
    response = llm.invoke([HumanMessage(content=prompt)])
    agent_answer = response.content.strip()

    table_results = [
        {"source": t["source"], "description": t["description"]}
        for t in tables
    ]

    trace = {
        "agent": "table_agent",
        "status": "success",
        "tables_used": len(tables),
        "sources": list({t["source"] for t in tables}),
        "agent_answer_preview": agent_answer[:200],
    }
    logger.info("Table agent: completed analysis")

    return {
        **state,
        "table_results": table_results,
        "messages": state["messages"] + [HumanMessage(content=agent_answer, name="table_agent")],
        "agent_trace": state.get("agent_trace", []) + [trace],
    }
