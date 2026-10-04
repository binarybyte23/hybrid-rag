"""HyDE enabled/disabled/concat/fallback paths."""
import pytest
from hybrid_rag.hyde import HyDEExpander
from hybrid_rag.llm.mock import MockLLM


class Boom:
    async def generate(self, prompt, system=""):
        raise RuntimeError("llm down")


@pytest.mark.asyncio
async def test_disabled_returns_raw():
    e = HyDEExpander(MockLLM(), enabled=False)
    assert await e.expand("hello?") == "hello?"


@pytest.mark.asyncio
async def test_enabled_returns_expansion():
    e = HyDEExpander(MockLLM(), enabled=True)
    out = await e.expand("How many vacation days?")
    assert out != "How many vacation days?" and "vacation" in out.lower()


@pytest.mark.asyncio
async def test_fallback_on_error():
    e = HyDEExpander(Boom(), enabled=True)
    assert await e.expand("raw query") == "raw query"


@pytest.mark.asyncio
async def test_fallback_on_empty():
    class Empty:
        async def generate(self, prompt, system=""):
            return "   "
    assert await HyDEExpander(Empty(), True).expand("q") == "q"


@pytest.mark.asyncio
async def test_prompt_contains_query():
    seen = {}
    class Spy:
        async def generate(self, prompt, system=""):
            seen["p"] = prompt
            return "hyp doc"
    await HyDEExpander(Spy(), True).expand("my query")
    assert "my query" in seen["p"]
