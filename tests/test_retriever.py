from retriever import Retriever, build_chunks, _split_sentences


def test_split_sentences_splits_on_sentence_boundaries():
    text = "France is in Europe. Its capital is Paris. The currency is the euro."
    sentences = _split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "France is in Europe."


def test_build_chunks_returns_nonempty_list_of_strings():
    chunks = build_chunks()
    assert len(chunks) > 5
    assert all(isinstance(c, str) and c for c in chunks)


def test_retrieve_returns_relevant_chunk_for_france_query():
    retriever = Retriever()
    results = retriever.retrieve("What is the capital of France?", k=3)
    assert any("Paris" in r for r in results)


def test_retrieve_returns_relevant_chunk_for_everest_query():
    retriever = Retriever()
    results = retriever.retrieve("How tall is Mount Everest?", k=3)
    assert any("Everest" in r or "8,849" in r for r in results)


def test_retrieve_respects_k():
    retriever = Retriever()
    results = retriever.retrieve("What is Python?", k=2)
    assert len(results) <= 2
