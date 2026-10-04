"""Mock LLM — deterministic, no network. For CI / offline runs."""
from hybrid_rag.llm.base import LLMProvider


class MockLLM(LLMProvider):
    async def generate(self, prompt: str, system: str = "") -> str:
        if "hypothetical" in prompt.lower() or "hyde" in system.lower() or "answer passage" in prompt.lower():
            return "Employees receive paid vacation days annually per company policy."
        return "Mock answer based on the provided passages [1]."
