"""
RAG Pipeline – the central orchestration module.

Flow for each query:
  1. Scope check  → ScopeDetector  (reject if OOS)
  2. RBAC check   → RBACManager    (reject if query is forbidden)
  3. PII masking  → PIIMasker      (sanitise query before sending to LLM)
  4. Retrieval    → VectorStore    (RBAC-filtered similarity search)
  5. Prompt build → assemble context + question
  6. LLM call     → LLMClient
  7. PII masking on response
  8. Token tracking → TokenTracker
"""

from __future__ import annotations

from typing import Dict, List, Optional

from loguru import logger

from config.settings import settings
from src.guardrails.pii_masker import PIIMasker
from src.guardrails.scope_detector import ScopeDetector
from src.llm.llm_client import LLMClient
from src.monitoring.token_tracker import TokenTracker
from src.rbac.access_control import RBACManager
from src.vectorstore.vector_db import VectorStore


_PROMPT_TEMPLATE = """\
You are a helpful internal company assistant.
Use ONLY the context provided below to answer the question.
If the context does not contain enough information, say so honestly.

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    """
    End-to-end RAG pipeline with RBAC, guardrails, and monitoring.

    Args:
        vector_store:   Initialised VectorStore instance.
        llm_client:     Initialised LLMClient instance.
        rbac_manager:   Initialised RBACManager instance.
        pii_masker:     Initialised PIIMasker instance.
        token_tracker:  Initialised TokenTracker instance.
        top_k:          Number of documents to retrieve per query.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: LLMClient,
        rbac_manager: RBACManager,
        pii_masker: Optional[PIIMasker] = None,
        token_tracker: Optional[TokenTracker] = None,
        top_k: int = 5,
    ) -> None:
        self._vs = vector_store
        self._llm = llm_client
        self._rbac = rbac_manager
        self._pii = pii_masker or PIIMasker()
        self._tracker = token_tracker or TokenTracker(model=llm_client.model_name)
        self._top_k = top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str, role: str) -> Dict:
        """
        Process a user question for the given role.

        Returns a dict with keys:
            answer        – LLM-generated (and PII-masked) response
            sources       – list of source metadata dicts
            blocked       – True when the query was rejected
            block_reason  – human-readable reason (if blocked)
            token_usage   – token stats dict
        """
        result: Dict = {
            "answer": "",
            "sources": [],
            "blocked": False,
            "block_reason": "",
            "token_usage": {},
        }

        # 1. Scope check
        role_keywords = self._rbac.get_allowed_keywords(role)
        scope_detector = ScopeDetector(role_keywords=role_keywords)
        if not scope_detector.is_in_scope(question):
            result["blocked"] = True
            result["block_reason"] = "out_of_scope"
            result["answer"] = scope_detector.get_rejection_message()
            logger.info(f"Query blocked [out_of_scope] for role '{role}'.")
            return result

        # 2. RBAC check (keyword-level)
        if not self._rbac.is_query_allowed(question, role):
            result["blocked"] = True
            result["block_reason"] = "rbac_policy"
            result["answer"] = (
                "You do not have permission to access information on that topic."
            )
            logger.info(f"Query blocked [rbac_policy] for role '{role}'.")
            return result

        # 3. Mask PII in the user query before sending to LLM
        sanitised_question = self._pii.mask(question)

        # 4. RBAC-filtered retrieval
        allowed_cats = list(self._rbac.get_allowed_categories(role))
        # C-suite: no filter (allowed_categories contains everything → pass None)
        filter_cats: Optional[List[str]] = allowed_cats if allowed_cats else None

        docs = self._vs.similarity_search(
            sanitised_question,
            allowed_categories=filter_cats,
            k=self._top_k,
        )

        # 5. Assemble prompt
        if docs:
            context = "\n\n---\n\n".join(d.page_content for d in docs)
        else:
            context = "No relevant documents found."

        prompt = _PROMPT_TEMPLATE.format(
            context=context, question=sanitised_question
        )

        # 6. LLM call
        raw_answer = self._llm.invoke(prompt)

        # 7. Mask PII in the LLM response
        answer = self._pii.mask(raw_answer)

        # 8. Token tracking
        input_tokens = self._tracker.estimate_tokens(prompt)
        output_tokens = self._tracker.estimate_tokens(answer)
        usage = self._tracker.record(
            query=question,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            role=role,
        )

        result["answer"] = answer
        result["sources"] = [
            {
                "source": d.metadata.get("source", "unknown"),
                "category": d.metadata.get("category", "unknown"),
                "filename": d.metadata.get("filename", ""),
            }
            for d in docs
        ]
        result["token_usage"] = usage
        return result

    @property
    def token_tracker(self) -> TokenTracker:
        return self._tracker
