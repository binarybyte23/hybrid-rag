"""Any OpenAI-compatible endpoint (Ollama / vLLM / LM Studio / Groq / OpenAI)."""
from __future__ import annotations
from hybrid_rag.llm.base import LLMProvider


class OpenAICompatLLM(LLMProvider):
    def __init__(self, base_url: str, model: str, api_key: str = "ollama"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    async def generate(self, prompt: str, system: str = "") -> str:
        import urllib.request, json
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps({"model": self.model, "messages": msgs,
                             "temperature": 0}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        import asyncio
        def _call():
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        return await asyncio.to_thread(_call)


def build_llm(provider: str, base_url: str, model: str, api_key: str = "ollama") -> LLMProvider:
    if provider == "mock":
        from hybrid_rag.llm.mock import MockLLM
        return MockLLM()
    from hybrid_rag.llm.openai_compat import OpenAICompatLLM
    return OpenAICompatLLM(base_url, model, api_key)
