"""
Streamlit Application.

Features:
  Document upload and ingestion
  Multi-agent chat interface
  Live agent trace visualization
  Source citation panel
  LLM provider status
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import streamlit as st


sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import config
from core.llm_factory import get_provider_name
from core.vectorstore import get_vector_store
from core.document_loader import DocumentLoader
from agents.supervisor import run_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="Doclyst",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Main theme */
    .main { background-color: #0f1117; }
    
    .stApp { background-color: #0f1117; color: #e2e8f0; }
    
    /* Agent trace card */
    .agent-card {
        background: #1e2433;
        border-left: 3px solid #4f8ef7;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 13px;
        font-family: 'JetBrains Mono', monospace;
    }
    .agent-card.supervisor { border-left-color: #f59e0b; }
    .agent-card.retrieval  { border-left-color: #10b981; }
    .agent-card.table      { border-left-color: #6366f1; }
    .agent-card.metadata   { border-left-color: #ec4899; }
    .agent-card.synthesis  { border-left-color: #14b8a6; }
    
    /* Source chunk */
    .source-chunk {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 4px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 12px;
        color: #94a3b8;
    }
    
    /* Provider badge */
    .provider-badge {
        background: #1e3a5f;
        color: #60a5fa;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Chat messages */
    .user-msg {
        background: #1e2433;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 3px solid #4f8ef7;
    }
    .assistant-msg {
        background: #162032;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 3px solid #10b981;
    }
    
    h1, h2, h3 { color: #f1f5f9; }
    
    .stButton > button {
        background: #4f8ef7;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: #3b7de8;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "tables" not in st.session_state:
    st.session_state.tables = []
if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "last_trace" not in st.session_state:
    st.session_state.last_trace = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

with st.sidebar:
    st.markdown("## Doclyst")
    st.markdown(
        f'<span class="provider-badge">⚡ {get_provider_name()}</span>',
        unsafe_allow_html=True,
    )

    st.markdown("### Document Ingestion")
    st.markdown("Upload manufacturing documents or use the demo dataset.")

    use_demo = st.button("Load Dataset", use_container_width=True)
    st.markdown("*Includes Q3 2024 maintenance report + sensor CSV*")

    st.markdown("**— or upload your own —**")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "csv", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("Ingest Uploaded Files", use_container_width=True):
            _ingest_uploads(uploaded_files)

    st.markdown("### Settings")
    show_trace = st.toggle("Show Agent Trace", value=True)
    show_sources = st.toggle("Show Source Chunks", value=True)

    st.markdown("### Sample Questions")
    sample_questions = [
        "Which machine had the most failures in Q3?",
        "Summarize the maintenance events for M-204",
        "Which sensors exceeded their thresholds?",
        "What was the total maintenance cost in Q3?",
        "What are the recommendations for Q4?",
        "Show average sensor values by machine",
    ]
    for q in sample_questions:
        if st.button(q, key=f"sample_{q[:20]}", use_container_width=True):
            st.session_state.pending_query = q


def _load_demo_dataset():
    with st.spinner("Indexing demo documents..."):
        loader = DocumentLoader(
            chunk_size=config.ingest.chunk_size,
            chunk_overlap=config.ingest.chunk_overlap,
        )
        demo_path = "data/"
        try:
            chunks, tables = loader.load_directory(demo_path)
            store = get_vector_store()
            store.add_chunks(chunks)
            st.session_state.vector_store = store
            st.session_state.tables = [
                {
                    "source": t.source,
                    "table_index": t.table_index,
                    "dataframe_json": t.dataframe_json,
                    "description": t.description,
                }
                for t in tables
            ]
            st.session_state.ingested = True
            st.success(f"Indexed {len(chunks)} chunks from {len(set(c.source for c in chunks))} documents")
        except Exception as e:
            st.error(f"Ingestion failed: {e}")


def _ingest_uploads(files):
    import tempfile
    with st.spinner("Ingesting uploaded documents..."):
        loader = DocumentLoader(
            chunk_size=config.ingest.chunk_size,
            chunk_overlap=config.ingest.chunk_overlap,
        )
        all_chunks, all_tables = [], []
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in files:
                dest = Path(tmpdir) / f.name
                dest.write_bytes(f.read())
            chunks, tables = loader.load_directory(tmpdir)
            all_chunks.extend(chunks)
            all_tables.extend(tables)

        if all_chunks:
            store = get_vector_store()
            store.add_chunks(all_chunks)
            st.session_state.vector_store = store
            st.session_state.tables = [
                {"source": t.source, "table_index": t.table_index,
                 "dataframe_json": t.dataframe_json, "description": t.description}
                for t in all_tables
            ]
            st.session_state.ingested = True
            st.success(f"Indexed {len(all_chunks)} chunks, {len(all_tables)} tables")
        else:
            st.error("No valid content extracted from uploaded files.")


if use_demo:
    _load_demo_dataset()


st.markdown("# Doclyst")
st.markdown("*Multi-agent RAG system for manufacturing document intelligence*")

if not st.session_state.ingested:
    st.info(" Load the demo dataset or upload documents to get started.")

col_chat, col_trace = st.columns([3, 2])


with col_chat:
    st.markdown("### Chat")

    # Display message history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-msg"> 🧑 <strong>You:</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-msg"> 🤖 <strong>Agent:</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

    # Handle sample question injection
    default_query = st.session_state.pop("pending_query", "")

    query = st.chat_input(
        "Ask a question about your documents...",
        disabled=not st.session_state.ingested,
    )
    if not query and default_query:
        query = default_query

    if query and st.session_state.ingested:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Agents working..."):
            try:
                t0 = time.time()
                result = run_query(
                    query=query,
                    vector_store=st.session_state.vector_store,
                    tables=st.session_state.tables,
                )
                elapsed = time.time() - t0

                answer = result["answer"]
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.last_trace = result["agent_trace"]
                st.session_state.last_sources = result["retrieval_results"]

                st.rerun()

            except Exception as e:
                st.error(f"Query failed: {e}")
                logger.exception("Query error")


with col_trace:
    if show_trace and st.session_state.last_trace:
        st.markdown("### Agent Trace")

        AGENT_COLORS = {
            "supervisor": ("supervisor"),
            "retrieval_agent": ("retrieval"),
            "table_agent": ("table"),
            "metadata_agent": ("metadata"),
            "multi_path": ("retrieval),
            "synthesis_agent": ("synthesis"),
        }

        for step in st.session_state.last_trace:
            agent = step.get("agent", "unknown")
            css_class, icon = AGENT_COLORS.get(agent, ("", "🤖"))
            details = {k: v for k, v in step.items() if k != "agent"}
            details_str = " | ".join(f"{k}: {v}" for k, v in details.items() if v)
            st.markdown(
                f'<div class="agent-card {css_class}">'
                f'{icon} <strong>{agent}</strong><br>'
                f'<span style="color:#94a3b8">{details_str}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if show_sources and st.session_state.last_sources:
        st.markdown("### Retrieved Sources")
        for r in st.session_state.last_sources[:4]:
            page_str = f" · p{r['page']}" if r.get("page") else ""
            st.markdown(
                f'<div class="source-chunk">'
                f'<strong>{r["source"]}{page_str}</strong><br>'
                f'{r["content"][:250]}...'
                f'</div>',
                unsafe_allow_html=True,
            )

    if not st.session_state.last_trace and st.session_state.ingested:
        st.markdown("### Agent Trace")
        st.markdown("*Ask a question to see the agent decision trace here.*")


st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#475569; font-size:12px;">'
    "Multi Agent · LangGraph + RAG + Streamlit · "
    f"Provider: {get_provider_name()}"
    "</p>",
    unsafe_allow_html=True,
)
