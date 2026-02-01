from lm_eval.tasks.longbench1.longbench1_utils import (
    _extract_paragraph_numbers,
    process_results_paragraph_retrieval,
)


def test_extract_paragraph_numbers_matches_label():
    assert _extract_paragraph_numbers("Paragraph 15: hello") == [15]
    assert _extract_paragraph_numbers("paragraph15") == [15]


def test_extract_paragraph_numbers_fallback_first_int():
    assert _extract_paragraph_numbers("Answer: 42") == [42]


def test_process_results_paragraph_retrieval_parses_generation():
    doc = {"answers": ["Paragraph 1"]}
    results = ["Paragraph 1: some text here"]
    out = process_results_paragraph_retrieval(doc, results)
    assert out == {"acc": 1.0}
