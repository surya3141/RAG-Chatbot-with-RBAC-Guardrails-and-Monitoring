"""Main RAG pipeline — orchestrates retrieval, guardrails, LLM, and monitoring."""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_TEMPLATE = (
    "You are a secure internal enterprise knowledge assistant.\n"
    "The current user has the role: {role}.\n"
    "You MUST only answer from the provided context and MUST NOT reveal "
    "information outside the user's authorised access level.\n"
    "If the answer is not present in the context, respond with: "
    "'I don't have enough information to answer that question.'\n\n"
    "Context:\n{{context}}"
)


class RAGPipeline:
    """End-to-end RAG pipeline with RBAC, guardrails, and monitoring."""

    def __init__(self, settings: Any, role: str) -> None:
        self.settings = settings
        self.role = role
        self._ready = False

        # Components — set up lazily by setup()
        self._vector_store = None
        self._llm_client = None
        self._pii_masker = None
        self._scope_detector = None
        self._token_tracker = None

    def setup(self) -> None:
        """Initialise all pipeline components.

        Safe to call multiple times — skips reinitialisation if already ready.
        """
        if self._ready:
            return

        from src.guardrails.pii_masker import PIIMasker
        from src.guardrails.scope_detector import ScopeDetector
        from src.llm.groq_client import GroqLLMClient
        from src.monitoring.token_tracker import TokenTracker
        from src.vectorstore.chroma_store import ChromaVectorStore

        self._scope_detector = ScopeDetector()
        self._pii_masker = PIIMasker()
        self._token_tracker = TokenTracker(
            alert_threshold=self.settings.TOKEN_COST_ALERT_THRESHOLD
        )
        self._vector_store = ChromaVectorStore(
            persist_dir=self.settings.CHROMA_PERSIST_DIR,
            embedding_model=self.settings.EMBEDDING_MODEL,
        )
        self._llm_client = GroqLLMClient(
            model_name=self.settings.GROQ_MODEL_NAME,
            api_key=self.settings.GROQ_API_KEY or "",
        )
        self._ready = True
        logger.info("RAGPipeline set up for role '%s'.", self.role)

    # ------------------------------------------------------------------
    # Public query interface
    # ------------------------------------------------------------------

    def query(self, question: str, username: str) -> Dict[str, Any]:
        """Process a user question through the full RAG pipeline.

        Returns a dict with keys:
            answer, sources, tokens_used, role, warning (optional)
        """
        self.setup()

        # 1. Guardrail: scope & blocked-pattern check
        is_allowed, reason = self._scope_detector.check_query(question)
        if not is_allowed:
            return {
                "answer": f"⚠️ {reason}",
                "sources": [],
                "tokens_used": 0,
                "role": self.role,
                "warning": reason,
            }

        # 2. Determine allowed collections for this role
        from src.rbac.middleware import get_allowed_collections

        allowed_collections = get_allowed_collections(self.role)
        if not allowed_collections:
            return {
                "answer": "You do not have access to any knowledge collections.",
                "sources": [],
                "tokens_used": 0,
                "role": self.role,
            }

        # 3. Retrieve relevant documents
        retrieved_docs = self._vector_store.query(
            query_text=question,
            collection_names=allowed_collections,
            n_results=5,
        )

        # 4. Mask PII in retrieved context
        context_parts: List[str] = []
        sources: List[str] = []
        for doc in retrieved_docs:
            masked_content = self._pii_masker.mask_pii(doc.page_content)
            context_parts.append(masked_content)
            src = doc.metadata.get("source", "unknown")
            if src not in sources:
                sources.append(src)

        # 5. Generate answer via LLM
        answer, tokens_used = self._generate_answer(
            question=question,
            context_parts=context_parts,
            retrieved_docs=retrieved_docs,
        )

        # 6. Mask PII in the LLM response
        answer = self._pii_masker.mask_pii(answer)

        # 7. Track token usage
        if tokens_used > 0:
            self._token_tracker.track_usage(
                model=self.settings.GROQ_MODEL_NAME,
                prompt_tokens=tokens_used // 2,
                completion_tokens=tokens_used - tokens_used // 2,
                username=username,
            )
            if self._token_tracker.check_alert(username):
                logger.warning("Cost alert triggered for user '%s'.", username)

        return {
            "answer": answer,
            "sources": sources,
            "tokens_used": tokens_used,
            "role": self.role,
        }

    def get_session_cost(self, username: str) -> float:
        """Return the estimated cost for *username* in this session."""
        if self._token_tracker is None:
            return 0.0
        return self._token_tracker.get_session_cost(username)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_answer(
        self,
        question: str,
        context_parts: List[str],
        retrieved_docs: list,
    ):
        """Call the LLM chain and return (answer_text, tokens_used)."""
        llm = self._llm_client.get_llm() if self._llm_client else None

        if llm is None or not context_parts:
            if not context_parts:
                return (
                    "I couldn't find relevant information in the knowledge base for your query.",
                    0,
                )
            return self._mock_answer(question), 0

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(role=self.role)
        retriever = self._vector_store.get_retriever(
            collection_names=self._get_allowed_collections(),
            k=5,
        )
        chain = self._llm_client.get_chain(retriever=retriever, system_prompt=system_prompt)

        if chain is None:
            return self._mock_answer(question), 0

        try:
            result = chain.invoke({"input": question})
            answer = result.get("answer", "No answer generated.")
            # Attempt to extract token counts from the response metadata
            tokens_used = self._extract_token_count(result)
            return answer, tokens_used
        except Exception as exc:
            logger.error("LLM chain invocation failed: %s", exc)
            return f"An error occurred while generating the answer: {exc}", 0

    def _get_allowed_collections(self) -> List[str]:
        from src.rbac.middleware import get_allowed_collections

        return get_allowed_collections(self.role)

    @staticmethod
    def _extract_token_count(result: Dict) -> int:
        """Best-effort extraction of total token count from a chain result."""
        try:
            usage = result.get("usage_metadata") or {}
            return int(usage.get("total_tokens", 0))
        except Exception:
            return 0

    @staticmethod
    def _mock_answer(question: str) -> str:
        """Return a deterministic demo answer when no LLM is configured."""
        return (
            f"[DEMO MODE — No API key configured]\n\n"
            f"You asked: '{question}'\n\n"
            "In a production deployment with a valid GROQ_API_KEY, this would "
            "return an answer grounded in the retrieved documents."
        )
