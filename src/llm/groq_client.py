"""Groq LLM client wrapper."""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_groq import ChatGroq

    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False
    logger.warning("langchain-groq not installed — GroqLLMClient will be a stub.")


_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful internal enterprise assistant. "
    "Answer questions using ONLY the provided context. "
    "If the answer is not in the context, say you don't know. "
    "Do not reveal confidential information outside the user's access level.\n\n"
    "Context:\n{context}"
)


class GroqLLMClient:
    """Wrapper around ChatGroq that builds LangChain RAG chains."""

    def __init__(self, model_name: str, api_key: str) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self._llm: Optional[object] = None

        if _GROQ_AVAILABLE and api_key:
            try:
                self._llm = ChatGroq(
                    model=model_name,
                    groq_api_key=api_key,
                    temperature=0.1,
                    max_tokens=2048,
                )
                logger.info("ChatGroq initialised with model '%s'.", model_name)
            except Exception as exc:
                logger.error("Failed to initialise ChatGroq: %s", exc)
        else:
            if not api_key:
                logger.warning("GROQ_API_KEY not set — GroqLLMClient is inactive.")

    def get_llm(self):
        """Return the underlying ChatGroq instance (or None if unavailable)."""
        return self._llm

    def get_chain(self, retriever, system_prompt: str = _DEFAULT_SYSTEM_PROMPT):
        """Build a simple retrieval-augmented chain using LCEL.

        Returns None if the LLM or retriever is unavailable.
        """
        if self._llm is None or retriever is None:
            logger.warning("Cannot build chain — LLM or retriever is None.")
            return None

        try:
            from langchain.chains.combine_documents import create_stuff_documents_chain
            from langchain.chains.retrieval import create_retrieval_chain
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )
            combine_chain = create_stuff_documents_chain(self._llm, prompt)
            chain = create_retrieval_chain(retriever, combine_chain)
            logger.debug("RAG chain constructed.")
            return chain
        except Exception as exc:
            logger.error("Failed to build RAG chain: %s", exc)
            return None
