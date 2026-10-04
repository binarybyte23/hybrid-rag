"""Reciprocal Rank Fusion, from scratch. No deps.

score(d) = Σ_lists 1 / (k + rank), rank is 1-based.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class FusedDoc:
    doc_id: str
    rrf_score: float
    ranks: dict[str, int] = field(default_factory=dict)  # list_name -> 1-based rank
    content: str = ""
    source: str = ""
    page: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None


def rrf_fuse(ranked_lists: dict[str, list], k: int = 60) -> list[FusedDoc]:
    """ranked_lists: name -> ordered list of RetrievalResult (or any obj with doc_id/content/source/score).

    Deterministic tie-break: higher appearance count, then higher rrf_score, then doc_id.
    """
    acc: dict[str, FusedDoc] = {}
    for name, results in ranked_lists.items():
        for rank0, r in enumerate(results, start=1):
            doc_id = r.doc_id
            fused = acc.get(doc_id)
            if fused is None:
                fused = FusedDoc(doc_id=doc_id, rrf_score=0.0,
                                 content=getattr(r, "content", ""),
                                 source=getattr(r, "source", ""),
                                 page=getattr(r, "page", None))
                acc[doc_id] = fused
            fused.rrf_score += 1.0 / (k + rank0)
            fused.ranks[name] = rank0
            score = getattr(r, "score", None)
            if name.startswith("dense") and fused.dense_score is None:
                fused.dense_score = score
            if name.startswith("sparse") and fused.sparse_score is None:
                fused.sparse_score = score
            # keep longest content if duplicates differ
            c = getattr(r, "content", "")
            if c and len(c) > len(fused.content):
                fused.content = c
    return sorted(acc.values(),
                  key=lambda d: (-len(d.ranks), -d.rrf_score, d.doc_id))
