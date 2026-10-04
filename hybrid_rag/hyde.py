"""Stage 1 — HyDE: LLM writes a hypothetical answer; we embed that instead of the raw query."""
from __future__ import annotations
from hybrid_rag.llm.base import LLMProvider

HYDE_PROMPT = ("Write a short hypothetical answer passage for this question "
               "(it may contain errors, focus on topical vocabulary):\n{query}")


class HyDEExpander:
    def __init__(self, llm: LLMProvider, enabled: bool = True):
        self.llm = llm
        self.enabled = enabled

    async def expand(self, query: str) -> str:
        """Return hypothetical doc, or raw query on disable/error (fallback)."""
        if not self.enabled:
            return query
        try:
            out = await self.llm.generate(HYDE_PROMPT.format(query=query),
                                          system="hyde")
            return out.strip() or query
        except Exception:
            return query  # ponytail: fallback to raw query, never fail retrieval
