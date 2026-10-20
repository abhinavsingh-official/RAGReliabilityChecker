import pytest

from hallucination_checker import check_answer, load_detector

CONTEXT = ["France has a population of about 68 million people.", "Its capital is Paris."]
QUESTION = "What is the capital of France and its population?"


@pytest.fixture(scope="module")
def detector():
    return load_detector()


def test_grounded_answer_has_no_flagged_spans(detector):
    answer = "The capital of France is Paris, and the population is about 68 million."
    spans = check_answer(detector, CONTEXT, QUESTION, answer)
    assert spans == []


def test_hallucinated_number_is_flagged(detector):
    answer = "The capital of France is Paris, and the population is about 200 million."
    spans = check_answer(detector, CONTEXT, QUESTION, answer)
    assert len(spans) > 0
    assert any("200 million" in s.text for s in spans)
