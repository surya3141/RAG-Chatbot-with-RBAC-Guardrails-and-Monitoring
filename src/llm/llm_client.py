"""
LLM client module.

Wraps the Groq Cloud API (Llama / Mixtral) via LangChain.  Falls back to
a simple echo stub when no API key is configured, so the rest of the
application can still be exercised without credentials.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger


class LLMClient:
    """
    Thin wrapper around the Groq LLM.

    Usage::

        client = LLMClient(api_key="gsk_...", model="llama3-8b-8192")
        answer = client.invoke("What is our Q3 revenue?")
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "llama3-8b-8192",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._llm = None

        if api_key:
            self._llm = self._build_groq_llm(api_key, model, temperature, max_tokens)
        else:
            logger.warning(
                "No GROQ_API_KEY provided. LLM calls will use stub responses."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def invoke(self, prompt: str) -> str:
        """Send *prompt* to the LLM and return the response text."""
        if self._llm is None:
            return self._stub_response(prompt)
        try:
            response = self._llm.invoke(prompt)
            # LangChain chat models return an AIMessage; plain LLMs return str
            if hasattr(response, "content"):
                return response.content
            return str(response)
        except Exception as exc:
            logger.error(f"LLM invocation failed: {exc}")
            return f"[LLM error: {exc}]"

    @property
    def model_name(self) -> str:
        return self._model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_groq_llm(
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ):
        try:
            from langchain_groq import ChatGroq

            return ChatGroq(
                groq_api_key=api_key,
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ImportError as exc:
            logger.error(f"langchain-groq not installed: {exc}")
            return None

    @staticmethod
    def _stub_response(prompt: str) -> str:
        return (
            "[STUB] LLM is not configured. "
            "Please set GROQ_API_KEY in your .env file. "
            f"Your query was: {prompt[:200]!r}"
        )
