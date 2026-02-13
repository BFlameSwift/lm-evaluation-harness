"""Tests for the `native` lm-eval model adapter.

Why this test exists
--------------------
Historically, `NativeCausalLM.generate_until()` had a regression where it would
return an empty string for non-trivial prompts because the code incorrectly
"sliced away" generated tokens using the prompt length.

This test constructs a `NativeCausalLM` instance via `__new__` and stubs out the
actual model forward/generation calls so it can run:
- quickly
- on CPU
- without requiring a real checkpoint
"""

from types import SimpleNamespace

import torch

from lm_eval.models.native import NativeCausalLM


class _DummyTokenizer:
    """Tiny deterministic tokenizer stub used by this unit test."""

    eos_id = 0
    pad_id = 999

    def encode(self, text: str, bos: bool = False, eos: bool = False):
        # Keep it deterministic and very cheap; content doesn't matter here.
        return [1, 2, 3]

    def decode(self, tokens):
        return " ".join(str(t) for t in tokens)


def test_generate_until_uses_generated_tokens_only():
    """Regression test: `_model_generate` returns *only* newly generated token ids.

    `generate_until` must NOT slice by the prompt length, or the output becomes
    empty for non-trivial prompts.
    """

    # NOTE: We intentionally bypass `__init__` because it would load checkpoints
    # and require GPU-only deps. For this regression we only need the Python-level
    # plumbing around `_model_generate` and token decoding.
    lm = NativeCausalLM.__new__(NativeCausalLM)

    # Minimal attribute surface that `generate_until` expects.
    lm._mode = "decoder"
    lm._batch_size = 1
    lm._device = torch.device("cpu")
    lm._dtype = torch.float32
    lm._temperature = 0.0
    lm._decoder_budget = 128
    lm._max_seq_length = 128
    lm._tokenizer = _DummyTokenizer()
    lm._chat_use_template = False
    lm._chat_template_version = "v3"
    lm._chat_add_generation_prompt = True
    lm._vllm_manager = None
    # vLLM flags are part of the adapter surface; set them explicitly so the
    # unit test doesn't depend on `__init__` defaults.
    lm._use_vllm_decoder = False
    lm._use_vllm_answer = False
    lm._use_vllm_reconstruct = False
    lm._distributed_args = SimpleNamespace(rank=0)
    lm._gen_do_sample_override = None
    lm._gen_temperature_override = None
    lm._gen_top_p_override = None
    lm._gen_max_gen_toks_override = None

    # Bind token helpers from the real class implementation.
    lm.tok_encode = NativeCausalLM.tok_encode.__get__(lm, NativeCausalLM)
    lm.tok_batch_encode = NativeCausalLM.tok_batch_encode.__get__(lm, NativeCausalLM)
    lm.tok_decode = NativeCausalLM.tok_decode.__get__(lm, NativeCausalLM)
    lm._format_chat = NativeCausalLM._format_chat.__get__(lm, NativeCausalLM)

    # Avoid writing debug JSONL in unit tests.
    lm._append_generate_debug_rows = lambda rows: None

    def _fake_model_generate(self, context, max_length, **kwargs):
        # Return 2 generated tokens + padding.
        return torch.tensor([[10, 11, self.pad_token_id, self.pad_token_id]], dtype=torch.long)

    lm._model_generate = _fake_model_generate.__get__(lm, NativeCausalLM)

    req = SimpleNamespace(
        args=("prompt", {"max_gen_toks": 4, "temperature": 0.0, "top_p": 1.0, "until": []}),
        doc=None,
        task_name="dummy",
        doc_id=0,
    )
    out = NativeCausalLM.generate_until(lm, [req], disable_tqdm=True)
    assert out == ["10 11"]
