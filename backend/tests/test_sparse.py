import uuid

import pytest

from app.retrieval.sparse import BM25Index, tokenize


@pytest.fixture
def index() -> BM25Index:
    return BM25Index()


def _add(index: BM25Index, text: str, paper_id=None, **metadata):  # noqa: ANN001, ANN202
    paper_id = paper_id or uuid.uuid4()
    chunk_id = uuid.uuid4()
    index.add(chunk_id, paper_id, text, {"paper_id": str(paper_id), **metadata})
    return chunk_id, paper_id


def test_tokenize_lowercases_and_drops_stopwords() -> None:
    assert tokenize("The Model is Trained on a Corpus") == ["model", "trained", "corpus"]


def test_tokenize_preserves_identifier_shaped_tokens() -> None:
    tokens = tokenize("BERT-base reaches f1_score of 0.89 on SQuAD")

    assert "bert-base" in tokens
    assert "f1_score" in tokens
    assert "0.89" in tokens


def test_tokenize_drops_single_characters() -> None:
    assert tokenize("a b model") == ["model"]


def test_empty_index_returns_nothing(index: BM25Index) -> None:
    assert index.search("anything") == []


def test_query_of_only_stopwords_returns_nothing(index: BM25Index) -> None:
    _add(index, "Some document about retrieval systems")

    assert index.search("the and of") == []


def test_documents_matching_more_query_terms_rank_higher(index: BM25Index) -> None:
    _, both = _add(index, "BERT achieves strong results on SQuAD question answering")
    _add(index, "We evaluate BERT on the GLUE benchmark")

    hits = index.search("BERT SQuAD", limit=2)

    assert hits[0].paper_id == both


def test_exact_lexical_match_is_found_where_paraphrase_would_miss(index: BM25Index) -> None:
    _, target = _add(index, "Results on the WikiSQL dataset using T5-large")
    _add(index, "A study of graph neural networks for molecules")

    hits = index.search("WikiSQL", limit=5)

    assert len(hits) == 1
    assert hits[0].paper_id == target


def test_rare_terms_outweigh_common_ones(index: BM25Index) -> None:
    for _ in range(10):
        _add(index, "transformer model architecture details")
    _, rare = _add(index, "transformer model with rotary positional embeddings")

    hits = index.search("rotary transformer", limit=3)

    assert hits[0].paper_id == rare


def test_search_respects_limit(index: BM25Index) -> None:
    for _ in range(5):
        _add(index, "retrieval augmented generation systems")

    assert len(index.search("retrieval", limit=2)) == 2


def test_filters_restrict_results(index: BM25Index) -> None:
    _add(index, "retrieval augmented generation", kind="text")
    _add(index, "retrieval augmented generation", kind="table")

    hits = index.search("retrieval", limit=10, filters={"kind": "table"})

    assert len(hits) == 1
    assert hits[0].metadata["kind"] == "table"


def test_readding_a_chunk_replaces_it_rather_than_duplicating(index: BM25Index) -> None:
    chunk_id = uuid.uuid4()
    paper_id = uuid.uuid4()
    index.add(chunk_id, paper_id, "original text about retrieval", {})
    index.add(chunk_id, paper_id, "replacement text about retrieval", {})

    assert index.size == 1


def test_replacing_a_chunk_updates_corpus_statistics(index: BM25Index) -> None:
    chunk_id = uuid.uuid4()
    paper_id = uuid.uuid4()
    index.add(chunk_id, paper_id, "quantum annealing optimization", {})
    index.add(chunk_id, paper_id, "classical gradient descent", {})

    # The old vocabulary must be gone, not merely shadowed.
    assert index.search("quantum") == []
    assert index.search("gradient")


def test_removing_a_paper_drops_all_its_chunks(index: BM25Index) -> None:
    paper_id = uuid.uuid4()
    _add(index, "first chunk about retrieval", paper_id=paper_id)
    _add(index, "second chunk about retrieval", paper_id=paper_id)
    _add(index, "unrelated chunk about vision")

    removed = index.remove_paper(paper_id)

    assert removed == 2
    assert index.size == 1


def test_average_length_tracks_the_corpus(index: BM25Index) -> None:
    assert index.average_length == 0.0

    _add(index, "one two three")
    _add(index, "four five six seven nine")

    assert index.average_length == pytest.approx(4.0)


def test_clear_empties_the_index(index: BM25Index) -> None:
    _add(index, "retrieval augmented generation")

    index.clear()

    assert index.size == 0
    assert index.search("retrieval") == []


def test_length_normalization_favors_concise_documents(index: BM25Index) -> None:
    _, concise = _add(index, "sparse retrieval")
    _add(index, "sparse retrieval " + "unrelated filler content ".join(str(i) for i in range(60)))

    hits = index.search("sparse retrieval", limit=2)

    assert hits[0].paper_id == concise
