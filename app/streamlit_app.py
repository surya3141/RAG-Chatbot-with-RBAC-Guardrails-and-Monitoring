"""Streamlit frontend for the Internal RAG Chatbot."""
import logging
import sys
import os

import streamlit as st

# Ensure the project root is on sys.path when running from the app/ directory
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Internal RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Role colour mapping
# ---------------------------------------------------------------------------
ROLE_COLORS = {
    "finance": "#1E88E5",   # Blue
    "hr": "#43A047",        # Green
    "csuite": "#F9A825",    # Gold
}

ROLE_LABELS = {
    "finance": "💰 Finance",
    "hr": "👥 HR",
    "csuite": "🏆 C-Suite",
}


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _init_session() -> None:
    defaults = {
        "authenticated": False,
        "user": None,
        "chat_history": [],
        "pipeline": None,
        "pipeline_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_pipeline(role: str):
    """Lazily initialise the RAGPipeline; cache in session state."""
    if st.session_state.pipeline is not None:
        return st.session_state.pipeline

    try:
        from config.settings import get_settings
        from src.rag.pipeline import RAGPipeline

        settings = get_settings()
        pipeline = RAGPipeline(settings=settings, role=role)
        pipeline.setup()
        st.session_state.pipeline = pipeline
        st.session_state.pipeline_error = None
        return pipeline
    except Exception as exc:
        logger.error("Failed to initialise RAGPipeline: %s", exc)
        st.session_state.pipeline_error = str(exc)
        return None


# ---------------------------------------------------------------------------
# Login page
# ---------------------------------------------------------------------------

def render_login() -> None:
    st.markdown(
        "<h1 style='text-align:center;'>🤖 Internal RAG Chatbot</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#888;'>Secure enterprise knowledge assistant "
        "with role-based access control</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("🔐 Login")
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="e.g. finance_user")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            _handle_login(username, password)

        st.divider()
        st.caption("Demo credentials:")
        st.code(
            "finance_user / finance123\n"
            "hr_user       / hr123\n"
            "ceo           / ceo123",
            language="text",
        )


def _handle_login(username: str, password: str) -> None:
    try:
        from src.rbac.middleware import authenticate_user

        user = authenticate_user(username, password)
        if user:
            st.session_state.authenticated = True
            st.session_state.user = user
            st.session_state.chat_history = []
            st.session_state.pipeline = None
            st.rerun()
        else:
            st.error("❌ Invalid username or password. Please try again.")
    except Exception as exc:
        st.error(f"Authentication error: {exc}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    user = st.session_state.user
    role = user["role"]
    color = ROLE_COLORS.get(role, "#888")
    label = ROLE_LABELS.get(role, role.upper())

    with st.sidebar:
        st.markdown(
            f"<h3 style='color:{color};'>{label}</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Name:** {user['name']}")
        st.markdown(f"**Username:** {user['username']}")

        # Accessible collections
        try:
            from src.rbac.middleware import get_allowed_collections

            collections = get_allowed_collections(role)
            st.markdown("**Accessible Collections:**")
            for c in collections:
                st.markdown(f"  - `{c}`")
        except Exception:
            pass

        st.divider()

        # Token usage
        pipeline = st.session_state.pipeline
        if pipeline:
            cost = pipeline.get_session_cost(user["username"])
            st.metric("Session Cost (est.)", f"${cost:.6f}")
        else:
            st.metric("Session Cost (est.)", "$0.000000")

        st.divider()

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        if st.button("🚪 Logout", use_container_width=True):
            for key in ["authenticated", "user", "chat_history", "pipeline", "pipeline_error"]:
                st.session_state[key] = None if key != "authenticated" else False
            st.session_state.chat_history = []
            st.rerun()


# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------

def render_chat() -> None:
    user = st.session_state.user
    role = user["role"]
    color = ROLE_COLORS.get(role, "#888")
    label = ROLE_LABELS.get(role, role.upper())

    st.markdown(
        f"<h2>🤖 Internal RAG Chatbot "
        f"<span style='font-size:0.6em; background:{color}; color:white; "
        f"padding:3px 10px; border-radius:12px;'>{label}</span></h2>",
        unsafe_allow_html=True,
    )

    # Render chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 Sources"):
                    for src in message["sources"]:
                        st.caption(f"• {src}")
            if message.get("warning"):
                st.warning(f"⚠️ {message['warning']}")
            if message.get("tokens"):
                st.caption(f"Tokens used: {message['tokens']}")

    # Input
    if prompt := st.chat_input("Ask a question about company data…"):
        # Show user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                response = _run_query(prompt, user)

            st.markdown(response["answer"])
            if response.get("sources"):
                with st.expander("📚 Sources"):
                    for src in response["sources"]:
                        st.caption(f"• {src}")
            if response.get("warning"):
                st.warning(f"⚠️ {response['warning']}")
            if response.get("tokens_used"):
                st.caption(f"Tokens used: {response['tokens_used']}")

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": response["answer"],
                "sources": response.get("sources", []),
                "warning": response.get("warning"),
                "tokens": response.get("tokens_used", 0),
            }
        )


def _run_query(question: str, user: dict) -> dict:
    """Execute the RAG query, falling back to a mock response on failure."""
    pipeline = _get_pipeline(user["role"])
    if pipeline is None:
        error_msg = st.session_state.pipeline_error or "Unknown error"
        return {
            "answer": (
                f"⚠️ **Demo Mode** — RAG pipeline unavailable ({error_msg}).\n\n"
                f"*You asked:* {question}\n\n"
                "Configure `GROQ_API_KEY` and ingest documents to enable full functionality."
            ),
            "sources": [],
            "tokens_used": 0,
            "role": user["role"],
        }

    try:
        return pipeline.query(question=question, username=user["username"])
    except Exception as exc:
        logger.error("Pipeline query error: %s", exc)
        return {
            "answer": f"An error occurred while processing your query: {exc}",
            "sources": [],
            "tokens_used": 0,
            "role": user["role"],
        }


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _init_session()

    if not st.session_state.authenticated:
        render_login()
    else:
        render_sidebar()
        render_chat()


if __name__ == "__main__":
    main()
