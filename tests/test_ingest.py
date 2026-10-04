from hybrid_rag.ingest import chunk_text


def test_chunk_split():
    chunks = chunk_text("a\n\nb\n\nc", "f.txt", max_chars=4)
    assert len(chunks) >= 2 and all(c[1] for c in chunks)


def test_chunk_ids_unique():
    chunks = chunk_text("hello\n\nworld", "f.txt", max_chars=1000)
    assert len({c[0] for c in chunks}) == len(chunks)
