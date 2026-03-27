"""
Streamlit frontend for the RAG Chatbot with RBAC.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when running via `streamlit run`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from config.settings import settings
from src.data_ingestion.chunking import chunk_documents
from src.data_ingestion.ingestion import load_documents
from src.llm.llm_client import LLMClient
from src.monitoring.evaluator import RAGEvaluator
from src.monitoring.token_tracker import TokenTracker
from src.rbac.access_control import RBACManager
from src.rag.pipeline import RAGPipeline
from src.guardrails.pii_masker import PIIMasker
from src.vectorstore.vector_db import VectorStore

# ---------------------------------------------------------------------------
# Demo user credentials  (username → {password, role})
# In production, replace with a secure auth backend.
# ---------------------------------------------------------------------------
DEMO_USERS = {
    "alice": {"password": "finance123", "role": "finance", "display": "Alice (Finance)"},
    "bob": {"password": "hr123", "role": "hr", "display": "Bob (HR)"},
    "carol": {"password": "ceo123", "role": "csuite", "display": "Carol (C-Suite)"},
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Internal RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_session() -> None:
    defaults = {
        "authenticated": False,
        "username": "",
        "role": "",
        "display_name": "",
        "chat_history": [],
        "pipeline": None,
        "index_ready": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _login(username: str, password: str) -> bool:
    user = DEMO_USERS.get(username.lower())
    if user and user["password"] == password:
        st.session_state["authenticated"] = True
        st.session_state["username"] = username.lower()
        st.session_state["role"] = user["role"]
        st.session_state["display_name"] = user["display"]
        return True
    return False


def _logout() -> None:
    for key in ["authenticated", "username", "role", "display_name",
                "chat_history", "pipeline", "index_ready"]:
        st.session_state[key] = None if key == "pipeline" else (
            False if key in ("authenticated", "index_ready") else
            ([] if key == "chat_history" else "")
        )


# ---------------------------------------------------------------------------
# Pipeline builder (cached per session to avoid reinitialising on every rerun)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading AI pipeline…")
def _build_pipeline(role: str):
    rbac = RBACManager(config_path=settings.rbac_config_path)
    llm = LLMClient(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
    )
    pii = PIIMasker()
    tracker = TokenTracker(
        model=settings.groq_model,
        alert_threshold_usd=settings.token_cost_alert_threshold,
    )
    vs = VectorStore(
        persist_dir=settings.chroma_persist_dir,
        embedding_model=settings.embedding_model,
    )
    pipeline = RAGPipeline(
        vector_store=vs,
        llm_client=llm,
        rbac_manager=rbac,
        pii_masker=pii,
        token_tracker=tracker,
    )
    return pipeline, vs, rbac


@st.cache_resource(show_spinner="Indexing documents…")
def _build_index(persist_dir: str, data_dir: str, embedding_model: str):
    vs = VectorStore(persist_dir=persist_dir, embedding_model=embedding_model)
    if vs.collection_count() == 0:
        docs = load_documents(data_dir)
        chunks = chunk_documents(docs)
        if chunks:
            vs.add_documents(chunks)
    return vs.collection_count()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    with st.sidebar:
        st.title("🤖 RAG Chatbot")
        st.markdown("---")

        if st.session_state["authenticated"]:
            st.success(f"Logged in as **{st.session_state['display_name']}**")
            role = st.session_state["role"]
            st.info(f"Role: `{role.upper()}`")

            # Document index status
            st.markdown("### 📚 Knowledge Base")
            if st.button("🔄 Index / Refresh Documents"):
                # clear cache and reindex
                _build_index.clear()
                count = _build_index(
                    settings.chroma_persist_dir,
                    settings.data_dir,
                    settings.embedding_model,
                )
                st.session_state["index_ready"] = count > 0
                st.success(f"Indexed {count} chunk(s).")

            # Session cost summary
            st.markdown("### 💰 Session Cost")
            pipeline = st.session_state.get("pipeline")
            if pipeline:
                summary = pipeline.token_tracker.session_summary
                st.metric("Tokens used", summary["total_tokens"])
                st.metric(
                    "Est. cost (USD)",
                    f"${summary['total_cost_usd']:.4f}",
                )
                threshold = summary["alert_threshold_usd"]
                if summary["total_cost_usd"] >= threshold:
                    st.warning(
                        f"⚠ Cost threshold ${threshold:.2f} reached!"
                    )

            st.markdown("---")
            if st.button("🚪 Logout"):
                _logout()
                st.rerun()
        else:
            st.info("Please log in to continue.")


# ---------------------------------------------------------------------------
# Login page
# ---------------------------------------------------------------------------

def _render_login() -> None:
    st.markdown("## 🔐 Login")
    st.markdown(
        "Demo credentials: **alice / finance123**, **bob / hr123**, "
        "**carol / ceo123**"
    )

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if _login(username, password):
            st.success(f"Welcome, {st.session_state['display_name']}!")
            # Pre-build pipeline and index
            pipeline, vs, rbac = _build_pipeline(st.session_state["role"])
            st.session_state["pipeline"] = pipeline

            count = _build_index(
                settings.chroma_persist_dir,
                settings.data_dir,
                settings.embedding_model,
            )
            st.session_state["index_ready"] = count > 0
            st.rerun()
        else:
            st.error("Invalid username or password.")


# ---------------------------------------------------------------------------
# Chat page
# ---------------------------------------------------------------------------

def _render_chat() -> None:
    role = st.session_state["role"]
    pipeline: RAGPipeline = st.session_state["pipeline"]

    st.markdown(f"## 💬 Internal Knowledge Assistant")

    # Warn if no documents indexed
    if not st.session_state.get("index_ready"):
        st.warning(
            "No documents are indexed yet. "
            "Use the sidebar to index the knowledge base."
        )

    # Chat history
    for entry in st.session_state["chat_history"]:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("sources"):
                with st.expander("📎 Sources"):
                    for src in entry["sources"]:
                        st.markdown(
                            f"- **{src['filename']}** ({src['category']})"
                        )
            if entry.get("token_usage"):
                usage = entry["token_usage"]
                st.caption(
                    f"Tokens: {usage.get('total_tokens', 0)} | "
                    f"Cost: ${usage.get('cost_usd', 0):.6f}"
                )
            if entry.get("blocked"):
                st.warning(
                    f"⛔ Blocked: {entry.get('block_reason', 'unknown')}"
                )

    # Input
    user_input = st.chat_input("Ask a question about company data…")
    if user_input:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state["chat_history"].append(
            {"role": "user", "content": user_input}
        )

        # Process via RAG pipeline
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = pipeline.query(question=user_input, role=role)

            st.markdown(result["answer"])

            if result.get("sources"):
                with st.expander("📎 Sources"):
                    for src in result["sources"]:
                        st.markdown(
                            f"- **{src['filename']}** ({src['category']})"
                        )

            if result.get("token_usage"):
                usage = result["token_usage"]
                st.caption(
                    f"Tokens: {usage.get('total_tokens', 0)} | "
                    f"Cost: ${usage.get('cost_usd', 0):.6f}"
                )

            if result.get("blocked"):
                st.warning(f"⛔ Blocked: {result.get('block_reason', 'unknown')}")

        st.session_state["chat_history"].append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
                "token_usage": result.get("token_usage", {}),
                "blocked": result.get("blocked", False),
                "block_reason": result.get("block_reason", ""),
            }
        )


# ---------------------------------------------------------------------------
# Evaluation tab (basic demo)
# ---------------------------------------------------------------------------

def _render_evaluation() -> None:
    st.markdown("## 📊 RAG Evaluation (RAGAS)")
    st.markdown(
        "Evaluate recent chat responses for faithfulness and relevancy. "
        "Requires RAGAS and an OpenAI/Groq key to be configured."
    )

    history = st.session_state.get("chat_history", [])
    qa_pairs = []
    i = 0
    while i < len(history) - 1:
        if history[i]["role"] == "user" and history[i + 1]["role"] == "assistant":
            qa_pairs.append((history[i]["content"], history[i + 1]))
            i += 2
        else:
            i += 1

    if not qa_pairs:
        st.info("No chat history to evaluate yet.")
        return

    if st.button("▶ Run Evaluation"):
        evaluator = RAGEvaluator()
        questions = [q for q, _ in qa_pairs]
        answers = [a["content"] for _, a in qa_pairs]
        contexts = [
            [s["filename"] for s in a.get("sources", [])] or ["no context"]
            for _, a in qa_pairs
        ]
        with st.spinner("Running RAGAS evaluation…"):
            scores = evaluator.evaluate(
                questions=questions,
                answers=answers,
                contexts=contexts,
            )

        st.json(scores)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _init_session()
    _render_sidebar()

    if not st.session_state["authenticated"]:
        _render_login()
        return

    # Ensure pipeline is available (handles page refresh)
    if st.session_state["pipeline"] is None:
        pipeline, vs, rbac = _build_pipeline(st.session_state["role"])
        st.session_state["pipeline"] = pipeline

    tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation"])
    with tab_chat:
        _render_chat()
    with tab_eval:
        _render_evaluation()


if __name__ == "__main__":
    main()
