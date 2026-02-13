# `native_impl/` (native-rag harness adapter internals)

This folder contains the refactored implementation for the lm-evaluation-harness
model adapter registered as `--model native`.

The stable entrypoint is still:
- `lm_eval/models/native.py`

`native.py` is intentionally kept small; it imports and re-exports
`native_impl.NativeCausalLM` so the `native` adapter can evolve without keeping
everything in a single 7k-line file.

## Call Flow (Where To Start Reading)

The easiest way to understand the refactor is to follow the runtime call path
from lm-eval into this folder.

High-level entrypoints:

1. lm-eval CLI constructs the model:
   - `lm_eval/models/native.py` registers the public name `native`.
   - It imports `lm_eval/models/native_impl/model.py:NativeCausalLM`.

2. `loglikelihood()` (multiple-choice scoring, perplexity-style tasks):
   - `NativeCausalLM.loglikelihood(...)` (in `model.py`) captures `Instance.doc`
     and other per-request metadata into `_active_loglikelihood_*` fields so
     downstream scoring can access structured dataset fields when needed.
   - It delegates scoring into:
     - `native_impl/likelihood.py:_loglikelihood_tokens(...)` for `mode=decoder`
     - `native_impl/likelihood.py:_loglikelihood_tokens_compress_answer(...)`
       for `mode=compress_answer` (compression checkpoints only)
     - `native_impl/likelihood.py:_loglikelihood_tokens_reconstruct_first(...)`
       for `mode=reconstruct_first` (compression checkpoints only)

3. `generate_until()` (generation tasks like NIAH/RULER):
   - `NativeCausalLM.generate_until(...)` delegates into
     `native_impl/generate.py:generate_until(...)`.
   - Some modes require vLLM prompt-embeds:
     - prompt-embeds construction is in `native_impl/reconstruct.py`
     - vLLM safemodel export + engine init is in `native_impl/vllm_backend.py`

4. MCQ verifier scoring (`mcq_score_mode=yes_*`):
   - Verifier prompt construction helpers live in `native_impl/mcq_scoring.py`.
   - Low-level scoring helpers (token-range scoring, chunked projection) live in
     `native_impl/scoring_mixin.py` and are mixed into `NativeCausalLM`.

Notes:
- Avoid absolute paths in docs/comments; always use repo-relative paths.
- There should be exactly one canonical harness checkout under `llm/lm-evaluation-harness/`.

## Quickstart (lm-eval CLI)

Typical pattern:

```bash
python -m lm_eval \
  --model native \
  --tasks mmlu \
  --limit 10 \
  --output_path /tmp/lm_eval_out \
  --model_args checkpoint_dir=/path/to/native_ckpt,tokenizer_path=/path/to/tokenizer,mode=decoder,batch_size=1
```

Common knobs (passed via `--model_args ...`):
- `checkpoint_dir=...`: native checkpoint directory (expects `metadata.json`).
- `pretrain_model_dir=...`: HF/transformers model directory (no `metadata.json`).
- `tokenizer_path=...`: tokenizer path for native checkpoints (HF path for HF models).
- `mode=decoder|compress_answer|reconstruct_first|vllm_decoding_with_compress|niah_generate`
- `batch_size=...`
- `max_seq_length=...`: decoder token budget for torch paths.

## Modes (high level)

All modes share the same `NativeCausalLM` class, but route into different
implementations:

1. `mode=decoder`
   - Vanilla causal LM behavior.
   - Used for standard loglikelihood tasks (e.g. MMLU/ARC/HellaSwag) and basic
     `generate_until`.

2. `mode=compress_answer`
   - For compression checkpoints (`MassiveCompressedMemoryModel`) that expose
     `compression_embeddings`.
   - Intended to *actually compress* the (long) context into memory slots, then
     score/generate conditioned on `<BOM> [mem slots] <EOM> + (suffix/raw query)`.
   - In practice, to avoid degenerate scoring on short MCQ prompts, we support
     (configurable) keeping a small raw suffix.

3. `mode=reconstruct_first`
   - For compression checkpoints.
   - Reconstruct a compressed context first (optionally with vLLM prompt-embeds),
     then compute likelihood on the continuation.

4. `mode=vllm_decoding_with_compress`
   - Hybrid mode used in some long-context suites.
   - vLLM decodes/reconstructs text, then native compression runs on top.

5. `mode=niah_generate`
   - NIAH-focused generation path (needle-in-a-haystack). Usually requires
     compression model + vLLM prompt-embeds backend for performance/stability.

## Glossary (Tokens / Concepts)

The native adapter uses a few "special token" concepts that show up in logs,
debug JSONL, and code comments:

- **BOM / EOM**: "begin/end of memory" markers.
  - Implemented by token ids `BEGIN_OF_MEMORY_INDEX` / `END_OF_MEMORY_INDEX`.
  - Surround the learned memory-slot embeddings in decoder prompt_embeds.
- **BOQ**: "begin of query" marker.
  - Token id: `BEGIN_OF_QUERY_INDEX`.
  - Used in some prompt formats to delimit query text from context/memory.
- **BOR / EOR**: "begin/end of reconstruction" markers.
  - Token ids: `BEGIN_OF_RECONSTRUCTION_INDEX` / `END_OF_RECONSTRUCTION_INDEX`.
  - Used by reconstruct-first and some NIAH variants.
- **n_spans**: number of encoder spans compressed into memory slots.
  - Each span consumes up to `max_mem_span_len` *raw* tokens on the encoder.
  - Each span produces `num_comp` learned memory slots for the decoder.
- **num_comp**: number of compression tokens per span (a.k.a. `num_compression_tokens`).
- **max_mem_span_len**: raw tokens per encoder span (eval-time knob; important for long-context).
- **decoder_budget**: max prompt length that the decoder backend can accept.
  - For vLLM prompt_embeds, this is effectively `vllm_max_model_len` (hard limit).
- **suffix_raw / min_suffix_tokens**: raw prompt tokens kept *uncompressed* near the answer.
  - This mitigates degenerate MCQ scoring when short prompts are fully compressed.

## vLLM integration and safemodel export

For some modes (especially `compress_answer` and `reconstruct_first` on
compression models), we use vLLM with `enable_prompt_embeds=True` so we can pass
pre-computed embeddings instead of massive token prompts.

Native checkpoints are not directly vLLM-loadable, so we export a transformers
style directory to `.../safemodel/` (contains `config.json` and
`model.safetensors`). This is handled by:
- `native_impl/vllm_backend.py:init_vllm_param()`

### Safer I/O on blobfuse / network filesystems (recommended)

Some network filesystems have unreliable atomic rename semantics and can lead to:
- `.tmp -> model.safetensors` rename failures
- partially-written `config.json` being read by vLLM (invalid JSON)

To avoid this, force safemodel generation/copy to a local disk:

```bash
export NATIVE_VLLM_FORCE_LOCAL_SAFEMODEL=1
export NATIVE_VLLM_LOCAL_SAFEMODEL_ROOT=/tmp/native_vllm_safemodel
mkdir -p /tmp/native_vllm_safemodel
```

This makes vLLM read from `/tmp/...` even if your final results are written to
a mounted volume.

## MCQ scoring (MMLU/ARC/HellaSwag)

By default, MCQ tasks in lm-eval use loglikelihood comparison across choices
(`mcq_score_mode=ll`). The native adapter additionally supports a verifier-style
path that converts a multi-choice candidate into a yes/no question.

Relevant `--model_args`:
- `mcq_score_mode=ll|yes_only|yes_minus_no|yes_prob`
- `verifier_score_mode=yes_prob|yes_only|yes_minus_no` (if `mcq_score_mode` is verifier-based)
- `verifier_apply_norm=none|token_avg|...` (normalization on verifier score)
- `mcq_verifier_prompt_style=minimal|...`
- `mcq_verifier_candidate_style=auto|text_only|...`
- `mcq_verifier_tie_break=none|...`

Implementation lives in:
- `native_impl/mcq_scoring.py` (text building and small helpers)
- `native_impl/scoring_mixin.py` (scoring + verifier execution)

## Module map

- `model.py`
  - `NativeCausalLM` class.
  - Parses `--model_args`, initializes torch + optional vLLM backends, routes to
    likelihood/generation functions.

- `likelihood.py`
  - Implements `_loglikelihood_tokens(...)` and compression-aware variants.

- `generate.py`
  - Implements `generate_until(...)` for torch/vLLM and special long-context
    modes (NIAH/RULER).

- `reconstruct.py`
  - Compression and prompt-embeds construction utilities.
  - Tail-span truncation guard rails for vLLM max length.

- `vllm_backend.py`
  - safemodel export, config patching, vLLM engine initialization/teardown.

- `mcq_scoring.py`
  - MCQ verifier prompt assembly and related parsing/normalization helpers.

- `scoring_mixin.py`
  - Shared scoring primitives used by both `likelihood.py` and `generate.py`
    (verifier scoring, chunked projection, fixed-base continuation scoring).

## Debug artifacts

When enabled (`save_loglikelihood_debug=true`), the adapter writes per-request
JSONL rows alongside lm-eval outputs, which can be used to audit:
- token counts / truncation behavior
- per-choice loglikelihoods and verifier scores
- compression span metadata (n_spans/prefix/suffix lengths)

These are derived from `--output_path` via `native_impl/utils.py:derive_lm_eval_output_dir()`.
