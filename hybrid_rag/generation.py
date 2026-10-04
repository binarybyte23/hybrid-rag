"""Stage 5 — grounded generation with [n] citations + source attribution."""
from __future__ import annotations
import re
from dataclasses import dataclass
from hybrid_rag.fusion import FusedDoc
from hybrid_rag.llm.base import LLMProvider

SYSTEM = "Answer ONLY from the passages below. Cite every claim with [n] markers."

CITE_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    text: str
    sources: list[dict]


def _loc(d: FusedDoc) -> str:
    loc = d.source
    if d.page is not None:
        loc += f", p.{d.page}"
    return loc


def build_prompt(query: str, contexts: list[FusedDoc]) -> str:
    passages = "\n\n".join(f"[{i+1}] {d.content}\n(source: {_loc(d)})"
                           for i, d in enumerate(contexts))
    return f"Question: {query}\n\nPassages:\n{passages}\n\nAnswer with [n] citations:"


def parse_citations(text: str, contexts: list[FusedDoc]) -> list[dict]:
    out = []
    for m in CITE_RE.finditer(text):
        i = int(m.group(1)) - 1
        if 0 <= i < len(contexts):
            d = contexts[i]
            out.append({"n": i + 1, "doc_id": d.doc_id, "source": d.source,
                        "page": d.page, "rrf_score": d.rrf_score, "ranks": d.ranks})
    # dedupe by n
    return list({s["n"]: s for s in out}.values())


async def generate_answer(llm: LLMProvider, query: str,
                          contexts: list[FusedDoc]) -> Answer:
    if not contexts:
        return Answer(text="I don't know — no passages retrieved.", sources=[])
    text = await llm.generate(build_prompt(query, contexts), system=SYSTEM)
    return Answer(text=text, sources=parse_citations(text, contexts))
