import unittest

from lm_eval.models.native_doc_utils import should_keep_context_raw


class TestNativeShortCtxPolicy(unittest.TestCase):
    def test_disabled_passthrough(self):
        keep = should_keep_context_raw(
            enable_short_ctx_passthrough=False,
            context_len=100,
            fixed_len=50,
            reserve_gen=10,
            decoder_budget=1024,
            max_seq_len=4096,
            compress_threshold=8192,
        )
        self.assertFalse(keep)

    def test_enabled_and_fits(self):
        keep = should_keep_context_raw(
            enable_short_ctx_passthrough=True,
            context_len=100,
            fixed_len=50,
            reserve_gen=10,
            decoder_budget=1024,
            max_seq_len=4096,
            compress_threshold=8192,
        )
        self.assertTrue(keep)

    def test_reject_when_over_decoder_budget(self):
        keep = should_keep_context_raw(
            enable_short_ctx_passthrough=True,
            context_len=1000,
            fixed_len=50,
            reserve_gen=10,
            decoder_budget=512,
            max_seq_len=4096,
            compress_threshold=8192,
        )
        self.assertFalse(keep)

    def test_reject_when_over_max_seq_len(self):
        keep = should_keep_context_raw(
            enable_short_ctx_passthrough=True,
            context_len=500,
            fixed_len=50,
            reserve_gen=10,
            decoder_budget=4096,
            max_seq_len=400,
            compress_threshold=8192,
        )
        self.assertFalse(keep)

    def test_reject_when_reaching_threshold(self):
        keep = should_keep_context_raw(
            enable_short_ctx_passthrough=True,
            context_len=7000,
            fixed_len=500,
            reserve_gen=500,
            decoder_budget=16384,
            max_seq_len=16384,
            compress_threshold=8000,
        )
        self.assertFalse(keep)

    def test_reject_empty_context(self):
        keep = should_keep_context_raw(
            enable_short_ctx_passthrough=True,
            context_len=0,
            fixed_len=50,
            reserve_gen=10,
            decoder_budget=1024,
            max_seq_len=4096,
            compress_threshold=8192,
        )
        self.assertFalse(keep)

