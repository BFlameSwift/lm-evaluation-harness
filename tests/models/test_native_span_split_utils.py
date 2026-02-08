import unittest

from lm_eval.models.native_doc_utils import split_tokens_to_spans


class TestNativeSpanSplitUtils(unittest.TestCase):
    def test_empty_tokens_returns_no_spans(self):
        self.assertEqual(split_tokens_to_spans([], 128), [])

    def test_regular_split(self):
        tokens = list(range(10))
        spans = split_tokens_to_spans(tokens, 4)
        self.assertEqual(spans, [list(range(4)), list(range(4, 8)), [8, 9]])

    def test_non_positive_span_len_falls_back_to_one(self):
        tokens = [1, 2, 3]
        self.assertEqual(split_tokens_to_spans(tokens, 0), [[1], [2], [3]])
        self.assertEqual(split_tokens_to_spans(tokens, -5), [[1], [2], [3]])


if __name__ == "__main__":
    unittest.main()
