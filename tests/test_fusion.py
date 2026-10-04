"""RRF against hand-computed values."""
from hybrid_rag.fusion import rrf_fuse
from hybrid_rag.retrieval.base import RetrievalResult as R


def mk(doc_id, score=1.0, content="c", source="s"):
    return R(doc_id=doc_id, content=content, source=source, score=score)


def test_both_lists_beats_single_strong():
    a = rrf_fuse({"dense": [mk("strong", 0.99), mk("both", 0.8)],
                  "sparse": [mk("both", 5.0), mk("other", 4.0)]}, k=60)
    ids = [d.doc_id for d in a]
    assert ids[0] == "both"  # appears twice


def test_formula_value():
    out = rrf_fuse({"dense": [mk("a"), mk("b")]}, k=60)
    assert abs(out[0].rrf_score - 1 / 61) < 1e-9
    assert abs(out[1].rrf_score - 1 / 62) < 1e-9


def test_provenance_ranks():
    out = rrf_fuse({"dense": [mk("a"), mk("b")], "sparse": [mk("b"), mk("a")]}, k=60)
    by_id = {d.doc_id: d for d in out}
    assert by_id["a"].ranks == {"dense": 1, "sparse": 2}
    assert by_id["b"].ranks == {"dense": 2, "sparse": 1}


def test_k_sensitivity():
    small = rrf_fuse({"d": [mk("a"), mk("b")]}, k=1)
    big = rrf_fuse({"d": [mk("a"), mk("b")]}, k=60)
    assert (small[0].rrf_score - small[1].rrf_score) > (big[0].rrf_score - big[1].rrf_score)


def test_duplicate_within_list_counts_once():
    out = rrf_fuse({"dense": [mk("a"), mk("a")]})
    assert len([d for d in out if d.doc_id == "a"]) == 1


def test_deterministic_tiebreak():
    out = rrf_fuse({"d": [mk("b")], "s": [mk("a")]})
    # equal scores (1/61 each, 1 appearance) → doc_id order
    assert [d.doc_id for d in out] == ["a", "b"]


def test_empty_lists():
    assert rrf_fuse({}) == []
    assert rrf_fuse({"d": []}) == []


def test_score_preservation():
    out = rrf_fuse({"dense_x": [mk("a", score=0.9)], "sparse_y": [mk("a", score=3.2)]})
    assert out[0].dense_score == 0.9
    assert out[0].sparse_score == 3.2


def test_three_lists():
    out = rrf_fuse({"a": [mk("x")], "b": [mk("x")], "c": [mk("y")]}, k=60)
    assert out[0].doc_id == "x"
