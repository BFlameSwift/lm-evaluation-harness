"""Native harness model implementation (`--model native`).

This module defines `NativeCausalLM`, the lm-evaluation-harness model adapter
used by this monorepo's *native-rag* evaluation jobs.

High-level responsibilities:
- Parse `--model_args` coming from the lm-eval CLI (native checkpoints vs HF).
- Initialize tokenizer + torch model, and optionally a vLLM backend.
- Route scoring/generation calls into mode-specific helpers implemented in
  sibling modules (`likelihood.py`, `generate.py`, `reconstruct.py`, ...).
- Provide debug artifacts colocated with lm-eval output directories.

This file intentionally keeps GPU-heavy optional imports *lazy* (wrapped in
try/except and validated at runtime) so `import lm_eval` does not hard-fail in
minimal environments.

See also:
- `lm_eval/models/native.py` (stable entrypoint / registry hook)
- `lm_eval/models/native_impl/README.md` (usage + module map)
"""

import os
import sys
from pathlib import Path


def _maybe_add_native_rag_root_to_syspath() -> None:
    """
    The `native` model backend depends on modules that live outside the
    `lm-evaluation-harness` repo in this monorepo layout:

        <llm_root>/
          arch/
          eval_func/
          data/
          ...
          lm-evaluation-harness/

    When running `python -m lm_eval ...` from the harness directory, that
    `<llm_root>` is not on `sys.path` by default, so importing this module would
    fail and the model would not be registered.

    We keep this auto-discovery lightweight and best-effort so upstream-style
    installs (where these folders don't exist) are unaffected.
    """
    env_root = os.getenv("NATIVE_RAG_EVAL_LLM_ROOT") or os.getenv("NATIVE_RAG_LLM_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "arch").is_dir() and (candidate / "eval_func").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return
        # Allow pointing to the repo root (which contains `llm/`).
        llm_candidate = candidate / "llm"
        if (llm_candidate / "arch").is_dir() and (llm_candidate / "eval_func").is_dir():
            if str(llm_candidate) not in sys.path:
                sys.path.insert(0, str(llm_candidate))
            return

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "arch").is_dir() and (parent / "eval_func").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return
        # When harness is checked out outside `<llm_root>/lm-evaluation-harness` (e.g. git worktrees),
        # the monorepo `llm/` folder may live at a higher-level parent.
        llm_root = parent / "llm"
        if (llm_root / "arch").is_dir() and (llm_root / "eval_func").is_dir():
            if str(llm_root) not in sys.path:
                sys.path.insert(0, str(llm_root))
            return


# This must run before we import any in-tree modules (`arch/`, `eval_func/`, ...)
# so that `python -m lm_eval ...` works even when executed from within
# `lm-evaluation-harness/`.
_maybe_add_native_rag_root_to_syspath()

import json
import re
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval.api.model import TemplateLM

# NOTE: The native backend depends on our in-tree `arch/` code, which in turn
# depends on GPU-specific optional deps (e.g. flash-attn). Import it lazily so
# importing `lm_eval` does not hard-fail in minimal environments.
_ARCH_IMPORT_ERROR: Optional[BaseException] = None
try:
    from arch.model import ModelArgs, create_kv_cache  # type: ignore
    # from arch.comp_mem import CompressedMemoryModel as Model
    from arch.comp_mem import MassiveCompressedMemoryModel as Model  # type: ignore
except Exception as _e:  # pragma: no cover
    # Stash the import error so we can re-raise it with a clearer message when
    # a user actually tries to instantiate the model.
    ModelArgs = None  # type: ignore[assignment]
    create_kv_cache = None  # type: ignore[assignment]
    Model = None  # type: ignore[assignment]
    _ARCH_IMPORT_ERROR = _e
# `config.py` may import `arch/` at import-time; keep it optional for the same
# reason as `arch` above (allow `lm_eval` to import even if GPU deps are absent).
_CONFIG_IMPORT_ERROR: Optional[BaseException] = None
try:
    from config import DistributedArgs  # type: ignore
except Exception as _e:  # pragma: no cover
    DistributedArgs = None  # type: ignore[assignment]
    _CONFIG_IMPORT_ERROR = _e
from data.tokenizer import Tokenizer
from distributed import apply_tp
from datetime import datetime
from torch.distributed.device_mesh import init_device_mesh
from data.ae_loader import (
    BEGIN_OF_MEMORY_INDEX,
    END_OF_MEMORY_INDEX,
    BEGIN_OF_RECONSTRUCTION_INDEX,
    END_OF_RECONSTRUCTION_INDEX,
)
from data.retrieval_loader import BEGIN_OF_QUERY_INDEX
# `eval_func/` also depends on `arch/` (and thus flash-attn). Keep it optional at
# import-time; we validate availability when the model is instantiated.
_EVALFUNC_IMPORT_ERROR: Optional[BaseException] = None
try:
    from eval_func.model2safetensors import convert_checkpoint, safemodel_needs_reconvert  # type: ignore
    from eval_func.vllm_runner import (  # type: ignore
        VLLMEngineWrapper,
        VLLMEngineConfig,
        VLLMDecoderManager,
        VLLMRemoteEngineWrapper,
    )
    from eval_func.utils import load_checkpoint_harness, _build_device_mesh  # type: ignore
except Exception as _e:  # pragma: no cover
    convert_checkpoint = None  # type: ignore[assignment]
    safemodel_needs_reconvert = None  # type: ignore[assignment]
    VLLMEngineWrapper = None  # type: ignore[assignment]
    VLLMEngineConfig = None  # type: ignore[assignment]
    VLLMDecoderManager = None  # type: ignore[assignment]
    VLLMRemoteEngineWrapper = None  # type: ignore[assignment]
    load_checkpoint_harness = None  # type: ignore[assignment]
    _build_device_mesh = None  # type: ignore[assignment]
    _EVALFUNC_IMPORT_ERROR = _e
from lm_eval.models.native_doc_utils import get_doc_query_keys_by_task_name, split_doc_and_query
import math
import gc
import atexit
import weakref

_split_doc_and_query = split_doc_and_query

from .mcq_scoring import (
    _apply_mcq_verifier_tie_break,
    _apply_verifier_score_norm,
    _build_choice_verifier_prompt_text,
    _build_verifier_variant_texts,
    _extract_options_block_from_prompt_text,
    _is_mcq_verifier_mode,
    _map_choice_score_mode_to_mcq_mode,
    _map_choice_score_norm_to_verifier_norm,
    _parse_mcq_score_mode,
    _parse_mcq_verifier_candidate_style,
    _parse_mcq_verifier_prompt_style,
    _parse_mcq_verifier_tie_break,
    _parse_verifier_apply_norm,
    _parse_verifier_score_mode,
    _resolve_verifier_score_mode,
    _verifier_score_from_ll,
)
from .utils import (
    build_rope_scaling_config as _build_rope_scaling_config,
    coerce_bool as _coerce_bool,
    coerce_int as _coerce_int,
    derive_lm_eval_output_dir as _derive_lm_eval_output_dir,
    filter_kwargs_for,
    filter_kwargs_for_callable,
    normalize_optional_text as _normalize_optional_text,
    parse_mode as _parse_mode,
    str_to_dtype as _str_to_dtype,
    token_embed as _token_embed,
)

from .scoring_mixin import ScoringMixin


def _default_distributed_args() -> DistributedArgs:
    """Build a minimal `DistributedArgs` from torchrun-style env vars.

    We don't require `accelerate` for evaluation, but many of our jobs are
    launched under torchrun (or AMLT/azure wrappers that set RANK/WORLD_SIZE).
    This helper provides consistent defaults when those env vars are absent.
    """
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return DistributedArgs(rank=rank, local_rank=local_rank, world_size=world_size)


class NativeCausalLM(ScoringMixin, TemplateLM):
    """
    Minimal lm-evaluation-harness adapter for the native arch.Model checkpoints.

    This adapter supports two "weight sources":
    - Native checkpoints (`checkpoint_dir=...`): expects a `metadata.json`.
    - HF/transformers checkpoints (`pretrain_model_dir=...`): loaded via HF
      model classes, and can be used to sanity-check harness behavior.

    Usage examples:
    - Native checkpoint (decoder):
      `--model native --model_args checkpoint_dir=/path/to/ckpt,tokenizer_path=/path/to/tok,mode=decoder`
    - HF checkpoint (decoder):
      `--model native --model_args pretrain_model_dir=Qwen/Qwen3-4B-Base,mode=decoder`

    mode:
      - decoder (default): vanilla causal scoring on decoder tokens only.
      - compress_answer: compress the context via encoder, then score the answer conditioned on memory.
      - reconstruct_first: reconstruct the context (optionally with vLLM prompt_embeds), then score continuation PPL.
      - vllm_decoding_with_compress: use vLLM to decode the context, then compress the context conditioned on memory.
      - niah_generate: NIAH-focused generate-only path (no likelihood); uses compressed-memory decoding and optional BOR.

    For more details, see `lm_eval/models/native_impl/README.md`.
    """

    backend = "causal"

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        pretrain_model_dir: Optional[str] = None,
        # lm-evaluation-harness CLI may pass `--device` to the model constructor.
        # We accept it for compatibility, but the native model currently expects CUDA.
        device: Optional[str] = None,
        batch_size: int = 1,
        max_seq_length: Optional[int] = None,
        tokenizer_path: Optional[str] = None,
        dtype: str = "bfloat16",
        mode: str = "decoder",
        max_mem_span_len: Optional[int] = None,
        # for vllm related
        use_vllm_decoder: bool = False,
        use_vllm_answer: bool = False,
        use_vllm_reconstruct: bool = False,
        vllm_model_path: Optional[str] = None,
        vllm_max_model_len: Optional[int] = None,
        vllm_tensor_parallel: int = 1,
        vllm_gpu_memory_utilization: float = 0.4,
        vllm_allow_long_max_model_len: bool = False,
        # vLLM stability knob: disable compilation/cudagraph (recommended for prompt_embeds).
        # If unset, defaults to True when prompt_embeds are enabled.
        vllm_enforce_eager: Optional[bool] = None,
        vllm_output_root: Optional[str] = None,
        # Optional remote vLLM server (persistent engine) support
        vllm_server_host: Optional[str] = None,
        vllm_server_port: Optional[int] = None,
        vllm_server_auth: Optional[str] = None,
        vllm_server_authkey: str = "native-rag",
        vllm_server_timeout: Optional[float] = None,
        # for reconstruction related
        vllm_reconstruct_batch_size: int = 40,
        ppl_batch_size: Optional[int] = 8,
        # for compression related
        compress_threshold: int = 8192,
        compress_chunk: int = 2048,
        max_cycles: int = 10,
        rope_scaling_json: Optional[str] = None,
        rope_scaling_type: Optional[str] = None,
        rope_scaling_factor: Optional[float] = None,
        rope_scaling_original_max_position_embeddings: Optional[int] = None,
        vllm_rope_scaling_json: Optional[str] = None,
        vllm_rope_scaling_type: Optional[str] = None,
        vllm_rope_scaling_factor: Optional[float] = None,
        vllm_rope_scaling_original_max_position_embeddings: Optional[int] = None,
        compress_start_tokens: Optional[str] = "<think>",
        compress_answer_min_suffix_tokens: int = 128,
        compress_answer_force_min_span: bool = True,
        temperature: float = 1.0,
        # chat template related， also load from yaml task
        use_chat_template: bool = False, 
        chat_template_version: str = "v3",
        chat_add_generation_prompt: bool = True,
        add_thinking_tokens: bool = False,
        # reconstruct_first (loglikelihood) controls
        reconstruct_add_bor: bool = False,
        reconstruct_max_bor: int = 3,
        add_query_before_likelihood: bool = False,
        likelihood_prefix_reconstruct: Optional[str] = None,
        likelihood_prefix_compress_answer: Optional[str] = None,
        mcq_score_mode: str = "ll",
        # Backward-compat aliases used by eval_longbench-style configs.
        choice_score_mode: Optional[str] = None,
        choice_score_norm: Optional[str] = None,
        verifier_score_mode: str = "yes_prob",
        verifier_yes_token: Optional[str] = " Yes",
        verifier_no_token: Optional[str] = " No",
        verifier_apply_norm: str = "none",
        verifier_prompt_suffix: Optional[str] = None,
        mcq_verifier_prompt_style: str = "minimal",
        mcq_verifier_candidate_style: str = "auto",
        mcq_verifier_tie_break: str = "none",
        # NIAH generate_until defaults
        niah_use_bor: bool = False,
        # NIAH generate_until debugging (write JSONL cases)
        niah_debug_dir: Optional[str] = None,
        # 0 disables, <0 means "unlimited"
        niah_debug_max_cases: int = -1,
        niah_debug_max_prompt_chars: int = 8000,
        # Compression progress (helpful for slow NIAH/RULER runs). If unset:
        #   - enabled by default for `mode=niah_generate`
        #   - disabled otherwise
        show_compress_progress: Optional[bool] = None,
        save_loglikelihood_debug: bool = True,
        loglikelihood_debug_path: Optional[str] = None,
        # generate_until debugging (write JSONL cases for all tasks)
        save_generate_debug: bool = True,
        generate_debug_dir: Optional[str] = None,
        # 0 disables, <0 means "unlimited"
        generate_debug_max_cases: int = -1,
        generate_debug_max_prompt_chars: int = 8000,
        generate_debug_path: Optional[str] = None,

        add_boq_index: bool = False,
        remove_eot_token: bool = True,
        fill_decoder_prefix_embeds: bool = False,

        # Generation overrides (model_args-level). If provided, these take precedence over
        # task YAML `generation_kwargs` (e.g. RULER/NIAH defaults).
        gen_do_sample: Optional[bool] = None,
        gen_temperature: Optional[float] = None,
        gen_top_p: Optional[float] = None,
        gen_max_gen_toks: Optional[int] = None,
        
    ) -> None:
        super().__init__()

        # -----------------------------
        # Basic runtime configuration
        # -----------------------------
        # The native checkpoints are trained/evaluated primarily on CUDA. We still accept
        # `--device` from lm-eval for compatibility, but will prefer CUDA when available.
        self._dtype = _str_to_dtype(dtype)
        resolved_device = None
        if device:
            try:
                resolved_device = torch.device(str(device))
            except Exception:
                resolved_device = None
        if resolved_device is None:
            resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if resolved_device.type != "cuda":
            print(
                f"[native][warn] `--device {resolved_device}` requested, but native model is CUDA-first; "
                f"falling back to cuda if available.",
                file=sys.stderr,
            )
            if torch.cuda.is_available():
                resolved_device = torch.device("cuda")
        self._device = resolved_device

        # -----------------------------
        # Dependency validation (monorepo-only)
        # -----------------------------
        # Fail fast with a clear error if the optional `arch/` dependency chain
        # (e.g. flash-attn) is not available in the current environment.
        if ModelArgs is None or Model is None or DistributedArgs is None or load_checkpoint_harness is None:
            cause = _ARCH_IMPORT_ERROR or _CONFIG_IMPORT_ERROR or _EVALFUNC_IMPORT_ERROR
            raise ImportError(
                "The `native` backend requires in-tree `arch/` modules and their optional "
                "dependencies (e.g. flash-attn). Install the missing deps or run in the "
                "provided GPU container."
            ) from cause

        # -----------------------------
        # High-level mode / backend selection
        # -----------------------------
        self._batch_size = int(batch_size) if isinstance(batch_size, (int, float)) or str(batch_size).isdigit() else 1
        self._mode = _parse_mode(mode)
        self._max_mem_span_len_override = max_mem_span_len
        self._use_vllm_reconstruct = use_vllm_reconstruct
        self._use_vllm_decoder = use_vllm_decoder
        self._use_vllm_answer = use_vllm_answer
        self._vllm_manager = None
        self._vllm_output_root = vllm_output_root
        self._vllm_model_dir = None
        self._last_generate_debug: List[dict] = []

        # vLLM init/runtime knobs
        if vllm_enforce_eager is None:
            # prompt_embeds path is sensitive to vLLM compilation/cudagraph; prefer eager.
            self._vllm_enforce_eager = True
        else:
            self._vllm_enforce_eager = bool(vllm_enforce_eager)

        # Populated by our overridden `loglikelihood()` so `_loglikelihood_tokens_*` can
        # access structured dataset fields via `Instance.doc` when needed.
        self._active_loglikelihood_docs: Optional[List[Optional[dict]]] = None
        self._active_loglikelihood_task_names: Optional[List[Optional[str]]] = None
        self._active_loglikelihood_doc_ids: Optional[List[Optional[int]]] = None
        self._active_loglikelihood_choice_idxs: Optional[List[Optional[int]]] = None
        self._warned_doc_split_fallback: bool = False
        
        self._active_context_key = "context"
        self._active_question_key = "question"
        
        # NOTE: historically we supported a separate vLLM reconstruction batch size; in practice
        # we keep it tied to `batch_size` to avoid surprising OOMs.
        # self._vllm_reconstruct_batch_size = max(1, int(vllm_reconstruct_batch_size))
        self._vllm_reconstruct_batch_size = self._batch_size
        self._ppl_batch_size = max(1, int(ppl_batch_size)) if ppl_batch_size is not None else self._batch_size

        # -----------------------------
        # Compression model knobs
        # -----------------------------
        self._compress_threshold = max(1, int(compress_threshold))
        self._compress_chunk = max(1, int(compress_chunk))
        env_rope_scaling_json = os.environ.get("NATIVE_ROPE_SCALING_JSON")
        env_vllm_rope_scaling_json = os.environ.get("NATIVE_VLLM_ROPE_SCALING_JSON")
        self._rope_scaling_override = _build_rope_scaling_config(
            rope_scaling_json=rope_scaling_json if rope_scaling_json is not None else env_rope_scaling_json,
            rope_scaling_type=rope_scaling_type,
            rope_scaling_factor=rope_scaling_factor,
            rope_scaling_original_max_position_embeddings=rope_scaling_original_max_position_embeddings,
        )
        self._vllm_rope_scaling_override = _build_rope_scaling_config(
            rope_scaling_json=(
                vllm_rope_scaling_json
                if vllm_rope_scaling_json is not None
                else (env_vllm_rope_scaling_json if env_vllm_rope_scaling_json is not None else env_rope_scaling_json)
            ),
            rope_scaling_type=vllm_rope_scaling_type,
            rope_scaling_factor=vllm_rope_scaling_factor,
            rope_scaling_original_max_position_embeddings=vllm_rope_scaling_original_max_position_embeddings,
        )
        self._compress_answer_min_suffix_tokens = max(
            0, _coerce_int(compress_answer_min_suffix_tokens, 128) or 0
        )
        # `compress_answer_min_suffix_tokens`:
        # Keep the last N *raw* prompt tokens uncompressed (suffix near the answer).
        #
        # This is critical for short MCQ prompts (MMLU/ARC/HellaSwag). If you compress
        # the entire prompt into memory slots, the decoder loses direct access to the
        # question/options text and scoring can collapse to a near-random prior.
        #
        # Set this to 0 for "fully compressed" behavior (more aggressive, less stable).
        self._compress_answer_force_min_span = _coerce_bool(compress_answer_force_min_span, default=True)
        # `compress_answer_force_min_span`:
        # If True and the prompt is non-empty, force at least one compressed span even if
        # the context is shorter than `max_mem_span_len`. This ensures `mode=compress_answer`
        # is semantically different from `mode=decoder` (n_spans will not be 0).
        self._max_cycles = max(1, int(max_cycles))
        self._compress_start_tokens = []
        self._add_boq_index = bool(add_boq_index)
        self._fill_decoder_prefix_embeds = bool(fill_decoder_prefix_embeds)
        self._remove_eot_token = bool(remove_eot_token)

        
        if compress_start_tokens:
            for t in compress_start_tokens.split(","):
                t = t.strip()
                if t:
                    self._compress_start_tokens.append(t)

        # -----------------------------
        # Prompt formatting / generation knobs
        # -----------------------------
        self._chat_use_template = bool(use_chat_template)
        self._chat_template_version = chat_template_version
        self._chat_add_generation_prompt = bool(chat_add_generation_prompt)
        self._add_thinking_tokens = bool(add_thinking_tokens)
        self._temperature = temperature
        # Optional generation overrides (apply to generate_until only).
        self._gen_do_sample_override = gen_do_sample if gen_do_sample is not None else None
        self._gen_temperature_override = None if gen_temperature is None else float(gen_temperature)
        self._gen_top_p_override = None if gen_top_p is None else float(gen_top_p)
        self._gen_max_gen_toks_override = _coerce_int(gen_max_gen_toks, None)
        self._reconstruct_add_bor = bool(reconstruct_add_bor)
        self._reconstruct_max_bor = max(0, int(reconstruct_max_bor))
        self._add_query_before_likelihood = bool(add_query_before_likelihood)
        self._likelihood_prefix_reconstruct = _normalize_optional_text(likelihood_prefix_reconstruct)
        self._likelihood_prefix_compress_answer = _normalize_optional_text(likelihood_prefix_compress_answer)
        self._likelihood_prefix_tokens_reconstruct: Optional[List[int]] = None
        self._likelihood_prefix_tokens_compress_answer: Optional[List[int]] = None
        resolved_mcq_score_mode = mcq_score_mode
        mapped_choice_score_mode = _map_choice_score_mode_to_mcq_mode(choice_score_mode)
        if mapped_choice_score_mode is not None:
            explicit_mcq = str(mcq_score_mode or "").strip().lower()
            if explicit_mcq not in {"", "ll"} and mapped_choice_score_mode != explicit_mcq:
                print(
                    f"[native][warn] Both mcq_score_mode={mcq_score_mode!r} and "
                    f"choice_score_mode={choice_score_mode!r} were set; keeping mcq_score_mode.",
                    file=sys.stderr,
                )
            else:
                resolved_mcq_score_mode = mapped_choice_score_mode
        self._mcq_score_mode = _parse_mcq_score_mode(resolved_mcq_score_mode)
        self._verifier_score_mode = _parse_verifier_score_mode(verifier_score_mode)
        resolved_verifier_apply_norm = verifier_apply_norm
        mapped_choice_score_norm = _map_choice_score_norm_to_verifier_norm(choice_score_norm)
        if mapped_choice_score_norm is not None:
            explicit_norm = str(verifier_apply_norm or "").strip().lower()
            if explicit_norm not in {"", "none"} and mapped_choice_score_norm != explicit_norm:
                print(
                    f"[native][warn] Both verifier_apply_norm={verifier_apply_norm!r} and "
                    f"choice_score_norm={choice_score_norm!r} were set; keeping verifier_apply_norm.",
                    file=sys.stderr,
                )
            else:
                resolved_verifier_apply_norm = mapped_choice_score_norm
        self._verifier_apply_norm = _parse_verifier_apply_norm(resolved_verifier_apply_norm)
        self._verifier_yes_variant_texts = _build_verifier_variant_texts(verifier_yes_token, fallback="Yes")
        self._verifier_no_variant_texts = _build_verifier_variant_texts(verifier_no_token, fallback="No")
        self._mcq_verifier_prompt_style = _parse_mcq_verifier_prompt_style(mcq_verifier_prompt_style)
        # `explicit_yesno` prompts already end with a concrete "Answer:" slot.
        # Appending the legacy suffix duplicates the question and hurts discrimination.
        suffix_norm = _normalize_optional_text(verifier_prompt_suffix)
        if suffix_norm is None:
            if self._mcq_verifier_prompt_style == "explicit_yesno":
                suffix_norm = ""
            else:
                suffix_norm = "\nIs this correct? Answer with Yes or No.\n"
        self._verifier_prompt_suffix = suffix_norm
        self._mcq_verifier_candidate_style = _parse_mcq_verifier_candidate_style(mcq_verifier_candidate_style)
        self._mcq_verifier_tie_break = _parse_mcq_verifier_tie_break(mcq_verifier_tie_break)
        self._verifier_prompt_suffix_tokens: Optional[List[int]] = None
        self._verifier_yes_variants: List[Tuple[str, List[int]]] = []
        self._verifier_no_variants: List[Tuple[str, List[int]]] = []
        self._save_loglikelihood_debug = bool(save_loglikelihood_debug)
        self._loglikelihood_debug_path = loglikelihood_debug_path
        self._last_loglikelihood_debug: List[dict] = []
        self._save_generate_debug = bool(save_generate_debug)
        self._generate_debug_path = generate_debug_path
        self._generate_debug_dir = _normalize_optional_text(generate_debug_dir)
        # 0 disables, <0 means "unlimited", >0 caps number of written cases
        self._generate_debug_max_cases = int(generate_debug_max_cases)
        self._generate_debug_max_prompt_chars = max(0, int(generate_debug_max_prompt_chars))
        self._generate_debug_dumped = 0
        self._generate_debug_run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
        self._generate_debug_run_path: Optional[str] = None
        self._niah_use_bor = bool(niah_use_bor)
        # NIAH with BOR is intended to reconstruct relevant context for a given query.
        # Enable BOQ token insertion by default so the query is explicitly marked.
        # if self._mode == "niah_generate" and self._niah_use_bor:
        #     self._add_boq_index = True
        self._niah_debug_dir = _normalize_optional_text(niah_debug_dir)
        if not self._niah_debug_dir:
            # By default, colocate NIAH debug artifacts with lm-eval outputs.
            out_path = _normalize_optional_text(os.environ.get("LM_EVAL_OUTPUT_PATH", ""))
            if out_path:
                ext = os.path.splitext(out_path)[1].lower()
                if ext in (".json", ".jsonl"):
                    self._niah_debug_dir = os.path.dirname(out_path) or "."
                else:
                    # Match lm-eval's default directory layout: output_path/<model_name_sanitized>/...
                    # For native, the model name is derived from checkpoint_dir where possible.
                    model_tag = ""
                    if checkpoint_dir:
                        norm = str(checkpoint_dir).rstrip("/\\")
                        base = os.path.basename(norm)
                        parent = os.path.basename(os.path.dirname(norm))
                        model_tag = f"{parent}/{base}" if parent and parent != norm else base
                    if not model_tag:
                        model_tag = "native"
                    model_dir = re.sub(r"[\"<>:/\|\\?\*\[\]]+", "__", model_tag)
                    self._niah_debug_dir = os.path.join(out_path, model_dir)
        if self._save_generate_debug and not self._generate_debug_dir:
            # Default: colocate generate debug artifacts with lm-eval outputs (model subfolder).
            self._generate_debug_dir = (
                _derive_lm_eval_output_dir(
                    output_path=os.environ.get("LM_EVAL_OUTPUT_PATH", ""),
                    checkpoint_dir=checkpoint_dir,
                    default_model_tag="native",
                )
                or self._niah_debug_dir
            )
        # 0 disables, <0 means "unlimited", >0 caps number of written cases
        self._niah_debug_max_cases = int(niah_debug_max_cases)
        self._niah_debug_max_prompt_chars = max(0, int(niah_debug_max_prompt_chars))
        self._niah_debug_dumped = 0
        # Per-run debug file id (avoid appending different runs into a single JSONL).
        # Keep legacy `niah_debug_cases.jsonl` for backwards compatibility.
        self._niah_debug_run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
        self._niah_debug_run_path: Optional[str] = None
        if show_compress_progress is None:
            self._show_compress_progress = (self._mode == "niah_generate")
        else:
            self._show_compress_progress = bool(show_compress_progress)
        
        
        max_seq_length_int = _coerce_int(max_seq_length, None)
        vllm_max_model_len_int = _coerce_int(vllm_max_model_len, None)
        if max_seq_length_int is not None and max_seq_length_int > 0:
            self._decoder_budget = max(max_seq_length_int, 8192)
        elif vllm_max_model_len_int is not None and vllm_max_model_len_int > 0:
            # If user only sets `vllm_max_model_len`, treat it as the effective decoder budget.
            # This is important for NIAH/RULER where the raw context may exceed 32k but is
            # expected to fit via compressed spans.
            self._decoder_budget = max(vllm_max_model_len_int, 8192)
        else:
            self._decoder_budget = 8192
        
        

        distributed_args = _default_distributed_args()
        # Native supports tensor-parallel only; data-parallel (world_size > model_parallel_size) will duplicate work.
        self._distributed_args = distributed_args
        if self._device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.set_device(distributed_args.local_rank)
            except Exception as e:
                print(
                    f"[native][warn] Failed to set CUDA device (local_rank={distributed_args.local_rank}): {e}",
                    file=sys.stderr,
                )

        # Best-effort vLLM cleanup at process exit.
        #
        # vLLM v1 uses a child "EngineCore" process. If we rely purely on GC at
        # interpreter shutdown, vLLM can emit noisy "EngineCore died unexpectedly"
        # logs. Use an atexit handler so the shutdown happens while the runtime is
        # still mostly intact.
        if not bool(getattr(self, "_atexit_cleanup_registered", False)):
            self._atexit_cleanup_registered = True
            self_ref = weakref.ref(self)

            def _native_atexit_cleanup() -> None:
                obj = self_ref()
                if obj is None:
                    return
                try:
                    obj.shutdown_vllm_manager(verbose=False, terminate_children=True)
                except Exception:
                    pass

            atexit.register(_native_atexit_cleanup)
        
        if checkpoint_dir is None and pretrain_model_dir is None:
            raise ValueError("Provide either checkpoint_dir or pretrain_model_dir for native model.")

        if checkpoint_dir is None:
            model_args = ModelArgs(pretrain_model_dir=pretrain_model_dir, model_parallel_size=1)
            device_mesh = _build_device_mesh(distributed_args.world_size, model_args.model_parallel_size)
            model = Model.from_pretrained(model_args).cuda()
            if device_mesh is not None:
                apply_tp(model, device_mesh)
            tokenizer = Tokenizer(pretrain_model_dir)
        else:
            # Some native checkpoints are trained with a very large `max_seq_len` (e.g., to support
            # many compressed spans on the encoder side). For harness eval we should cap the
            # *decoder* RoPE cache and related buffers to the actual eval budget (vLLM/HF max len),
            # otherwise we may allocate extremely large rotary caches and hit ROCm quirks.
            max_seq_len_override = None
            if max_seq_length_int is not None and max_seq_length_int > 0:
                max_seq_len_override = int(max_seq_length_int)
            elif vllm_max_model_len_int is not None and vllm_max_model_len_int > 0:
                max_seq_len_override = int(vllm_max_model_len_int)
            model, tokenizer, _, device_mesh = load_checkpoint_harness(
                checkpoint_dir,
                distributed_args,
                tokenizer_path,
                max_seq_len_override=max_seq_len_override,
                rope_scaling_override=self._rope_scaling_override,
            )

        self._is_hf_model = hasattr(model, "config") and not hasattr(model, "args")

        # Guard against data-parallel: require world_size <= model_parallel_size (TP only)
        if self._distributed_args.world_size > 1:
            tp_size = device_mesh.mesh.shape[1] if device_mesh is not None else 1
            if self._distributed_args.world_size != tp_size:
                raise ValueError(
                    f"native model only supports tensor-parallel in lm-eval; "
                    f"got world_size={self._distributed_args.world_size}, model_parallel_size={tp_size}. "
                    "Set MODEL_PARALLEL_SIZE or model_args model_parallel_size to match num_processes, "
                    "and ensure checkpoint shards exist (model_state_rank_0..{tp_size-1})."
                )

        # `load_checkpoint_harness()` may already move/cast the model (and may fall back to a safer
        # dtype on ROCm). Avoid an extra `.to()` here, because it can be expensive and can also
        # re-trigger HIP dtype conversion issues.
        loaded_dtype = getattr(model, "_native_loaded_dtype", None)
        if loaded_dtype is not None and loaded_dtype != self._dtype:
            print(
                f"[native][warn] overriding requested dtype={self._dtype} with loaded dtype={loaded_dtype}",
                file=sys.stderr,
            )
            self._dtype = loaded_dtype

        def _same_cuda_device(a: torch.device, b: torch.device) -> bool:
            if a.type != "cuda" or b.type != "cuda":
                return a == b
            try:
                ai = torch.cuda.current_device() if a.index is None else int(a.index)
            except Exception:
                ai = a.index
            try:
                bi = torch.cuda.current_device() if b.index is None else int(b.index)
            except Exception:
                bi = b.index
            return ai == bi

        try:
            any_param = next(model.parameters())
            model_device = any_param.device
            model_dtype = any_param.dtype
        except StopIteration:
            model_device = self._device
            model_dtype = self._dtype

        needs_device = model_device != self._device and not _same_cuda_device(model_device, self._device)
        needs_dtype = (loaded_dtype is None) and (model_dtype != self._dtype)
        if needs_device or needs_dtype:
            self.model = model.to(dtype=self._dtype, device=self._device)
        else:
            self.model = model
        self.model.eval()
        self._tokenizer = tokenizer
        self._verifier_yes_variants = [
            (txt, self._tokenizer.encode(txt, bos=False, eos=False))
            for txt in self._verifier_yes_variant_texts
        ]
        self._verifier_yes_variants = [(txt, toks) for txt, toks in self._verifier_yes_variants if len(toks) > 0]
        self._verifier_no_variants = [
            (txt, self._tokenizer.encode(txt, bos=False, eos=False))
            for txt in self._verifier_no_variant_texts
        ]
        self._verifier_no_variants = [(txt, toks) for txt, toks in self._verifier_no_variants if len(toks) > 0]
        if not self._verifier_yes_variants:
            self._verifier_yes_variants = [(" Yes", self._tokenizer.encode(" Yes", bos=False, eos=False))]
        if not self._verifier_no_variants:
            self._verifier_no_variants = [(" No", self._tokenizer.encode(" No", bos=False, eos=False))]
        self._device_mesh = device_mesh
        self._model_parallel_group = device_mesh.get_group(1) if device_mesh is not None else None

        if max_seq_length_int is not None:
            self._max_seq_length = max_seq_length_int
        else:
            if hasattr(self.model, "args"):
                self._max_seq_length = _coerce_int(getattr(self.model.args, "max_seq_len", None), None)
            else:
                self._max_seq_length = _coerce_int(
                    getattr(getattr(self.model, "config", None), "max_position_embeddings", None), None
                )
                if self._max_seq_length is None:
                    self._max_seq_length = _coerce_int(
                        getattr(getattr(self.model, "config", None), "max_seq_len", None), None
                    )
                if self._max_seq_length is None:
                    self._max_seq_length = _coerce_int(
                        getattr(getattr(self.model, "config", None), "model_max_length", None), None
                    )
            if self._max_seq_length is None or self._max_seq_length <= 0:
                self._max_seq_length = 2048
        # Respect explicit eval-time budgets (`max_seq_length` or `vllm_max_model_len`) instead of
        # always inheriting the checkpoint's `max_seq_len`. Some checkpoints set a very large
        # `max_seq_len` (e.g., encoder-side span capacity), while evaluation may intentionally
        # cap the decoder prompt length to match a vLLM/HF backend configuration.
        user_budget_specified = bool(
            (max_seq_length_int is not None and max_seq_length_int > 0)
            or (vllm_max_model_len_int is not None and vllm_max_model_len_int > 0)
        )
        if not user_budget_specified:
            self._decoder_budget = max(self._decoder_budget, self._max_seq_length)
        if self._max_mem_span_len_override is not None and hasattr(self.model, "args"):
            # `max_mem_span_len` is an eval-time knob; older checkpoints/configs may
            # not declare it on `ModelArgs`, so we attach it dynamically.
            try:
                setattr(self.model.args, "max_mem_span_len", int(self._max_mem_span_len_override))
            except Exception:
                pass

        # `max_mem_span_len` (model_args) should override checkpoint defaults.
        # This is critical for long-context eval (RULER/NIAH/LongBench): if the
        # span length is smaller than intended, the number of spans explodes and
        # can exceed the decoder budget / vLLM max_model_len.
        span_len = _coerce_int(self._max_mem_span_len_override, None)
        if span_len is None and hasattr(self.model, "args"):
            span_len = _coerce_int(getattr(self.model.args, "max_mem_span_len", None), None)
        if span_len is None or span_len <= 0:
            # Default span length for splitting long contexts into memory chunks.
            span_len = 512
        self._max_mem_span_len = int(span_len)
        if hasattr(self.model, "args"):
            # Keep a single source of truth for helper functions that still look at
            # `model.args.max_mem_span_len` via getattr(...).
            try:
                setattr(self.model.args, "max_mem_span_len", int(self._max_mem_span_len))
            except Exception:
                pass
        self._num_compression_tokens = self.model.args.num_compression_tokens if hasattr(self.model, "args") else None

        # -----------------------------
        # vLLM configuration (lazy init)
        # -----------------------------
        # We only *require* vLLM for some compression-aware generation paths
        # (prompt_embeds). For pure loglikelihood MCQ scoring, torch is usually
        # sufficient and avoids extra moving pieces, so we keep vLLM init lazy.
        need_vllm = self._use_vllm_reconstruct or self._use_vllm_decoder or self._use_vllm_answer

        if self._mode in {"vllm_decoding_with_compress", "niah_generate"}:
            need_vllm = True

        self._use_vllm = need_vllm
        self._vllm_model_path = vllm_model_path
        self._vllm_max_model_len = _coerce_int(vllm_max_model_len, None)
        if self._vllm_max_model_len is None or self._vllm_max_model_len <= 0:
            self._vllm_max_model_len = _coerce_int(self._decoder_budget, None) or 2048
        self._vllm_tensor_parallel = vllm_tensor_parallel
        self._vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self._vllm_tokenizer_path = tokenizer_path
        self._vllm_checkpoint_dir = checkpoint_dir
        self._vllm_dtype = dtype
        self._vllm_server_host = _normalize_optional_text(vllm_server_host)
        self._vllm_server_port = _coerce_int(vllm_server_port, None)
        auth_override = _normalize_optional_text(vllm_server_auth)
        self._vllm_server_authkey = auth_override or vllm_server_authkey
        self._vllm_server_timeout = None
        if vllm_server_timeout is not None:
            try:
                self._vllm_server_timeout = float(vllm_server_timeout)
            except Exception:
                self._vllm_server_timeout = None
        self._vllm_allow_long_max_model_len = bool(vllm_allow_long_max_model_len) or (
            str(os.environ.get("NATIVE_VLLM_ALLOW_LONG_MAX_MODEL_LEN", "")).strip().lower()
            in {"1", "true", "yes", "y", "on"}
        )
        if self._vllm_allow_long_max_model_len:
            os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
        self._use_remote_vllm = bool(self._vllm_server_host and self._vllm_server_port)
        
        # NOTE: vLLM initialization is expensive and not needed for pure torch loglikelihood
        # runs (e.g., multiple-choice scoring). Defer vLLM init until first use.
        
    def _init_vllm_param(self) -> None:
        """Prepare/resolve the local vLLM model directory (safemodel export if needed)."""
        from .vllm_backend import init_vllm_param

        return init_vllm_param(self)

    def _init_vllm(self) -> None:
        """Instantiate the vLLM engine wrapper (local or remote) and attach it to `self`."""
        from .vllm_backend import init_vllm

        return init_vllm(self)

    def _ensure_vllm_manager(self, *, caller: str) -> None:
        """Best-effort lazy vLLM init.

        We keep a single vLLM manager per `NativeCausalLM` instance and only
        attempt initialization once. Callers provide `caller=...` so failures
        can be attributed to a specific mode/path in logs.
        """
        from .vllm_backend import ensure_vllm_manager

        return ensure_vllm_manager(self, caller=caller)
    
                
    def _ensure_vllm_config(self, safedir: str) -> None:
        """Patch vLLM config JSONs under `safedir/` to respect max model length."""
        from .vllm_backend import ensure_vllm_config

        return ensure_vllm_config(self, safedir)


    def shutdown_vllm_manager(
        self,
        *,
        empty_cache: bool = True,
        ipc_collect: bool = True,
        synchronize: bool = True,
        verbose: bool = True,
        terminate_children: bool = False,
        terminate_timeout_s: float = 10.0,
    ) -> None:
        from .vllm_backend import shutdown_vllm_manager as _shutdown_vllm_manager

        return _shutdown_vllm_manager(
            self,
            empty_cache=empty_cache,
            ipc_collect=ipc_collect,
            synchronize=synchronize,
            verbose=verbose,
            terminate_children=terminate_children,
            terminate_timeout_s=terminate_timeout_s,
        )

    # def __del__(self) -> None:
    #     # Never raise from a destructor.
    #     try:
    #         self.shutdown_vllm_manager()
    #     except Exception as e:
    #         print(f"WARNING: Failed to shutdown vLLM in destructor, Error: {e}", file=sys.stderr)

    # ---- Required TemplateLM properties ----
    @property
    def eot_token_id(self) -> int:
        return self._tokenizer.eos_id

    @property
    def max_length(self) -> int:
        return self.decoder_budget
    
    @property
    def decoder_budget(self) -> int:
        budget = getattr(self, "_decoder_budget", None)
        if budget is not None:
            try:
                return int(budget)
            except Exception:
                pass
        max_len = getattr(self, "_max_seq_length", None)
        try:
            max_len_int = int(max_len) if max_len is not None else 0
        except Exception:
            max_len_int = 0
        return max_len_int if max_len_int > 0 else 2048

    @property
    def max_gen_toks(self) -> int:
        # lm-eval uses this as the default `max_gen_toks` for generation when a
        # task doesn't specify it. For native compression models, we historically
        # re-used `compress_threshold` as a reasonable default cap (many tasks
        # override this via `generation_kwargs` anyway).
        return self._compress_threshold or self._max_seq_length or self._vllm_max_model_len

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def tokenizer_name(self) -> str:
        return getattr(self._tokenizer, "name_or_path", "native_tokenizer")

    @property
    def eot_token(self) -> int:
        return self.eot_token_id

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_id

    @property
    def pad_token_id(self) -> int:
        pad = getattr(self._tokenizer, "pad_id", None)
        if pad is None:
            pad = getattr(self._tokenizer, "pad_token_id", None)
        if pad is None or (isinstance(pad, int) and pad < 0):
            return self.eos_token_id
        return pad
    
    @property
    def stop_ids(self) -> List[int]:
        # vLLM can stop on token ids. We include common "end of turn" markers
        # used by Qwen-style chat templates, plus EOS.
        tokens = ["<|im_end|>","</s>","<|eot_id|>"]
        ids = []
        for token in tokens:
            token_ids = self._tokenizer.encode(token, bos=False, eos=False)
            if len(token_ids) == 1:
                ids.append(token_ids[0])
            else:
                print(f"Warning: token {token} has {len(token_ids)} ids: {token_ids}")
        ids.append(self.eos_token_id)
        return ids

    def _append_loglikelihood_debug_rows(self, rows: List[dict]) -> None:
        if not self._save_loglikelihood_debug:
            return
        if self._distributed_args.rank != 0:
            return
        if not rows:
            return
        self._last_loglikelihood_debug.extend(rows)
        out_path = self._loglikelihood_debug_path
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if out_path is None:
            # Prefer lm-eval output directory so debug artifacts live next to evaluator outputs.
            base_dir = (
                _derive_lm_eval_output_dir(
                    output_path=os.environ.get("LM_EVAL_OUTPUT_PATH", ""),
                    checkpoint_dir=getattr(self, "_vllm_checkpoint_dir", None),
                    default_model_tag="native",
                )
                or self._vllm_output_root
                or os.getcwd()
            )
            os.makedirs(base_dir, exist_ok=True)
            out_path = os.path.join(base_dir, f"loglikelihood_debug_{datetime_str}.jsonl")
            self._loglikelihood_debug_path = out_path
        else:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        
        with open(out_path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if bool(getattr(self, "_verbose_compress", False)) or not bool(getattr(self, "_loglikelihood_debug_notified", False)):
            print(f"Saved loglikelihood debug rows to {out_path}")
            self._loglikelihood_debug_notified = True

    def _append_generate_debug_rows(self, rows: List[dict]) -> None:
        """Persist generate_until debug cases (rank0 only)."""
        if not getattr(self, "_save_generate_debug", False):
            return
        if not rows:
            return
        if self._distributed_args.rank != 0:
            return
        max_cases = int(getattr(self, "_generate_debug_max_cases", 0) or 0)
        # 0 disables, <0 means "unlimited"
        if max_cases == 0:
            return
        dumped = int(getattr(self, "_generate_debug_dumped", 0) or 0)
        if max_cases > 0 and dumped >= max_cases:
            return

        debug_dir = getattr(self, "_generate_debug_dir", "") or ""
        if not debug_dir:
            debug_dir = (
                _derive_lm_eval_output_dir(
                    output_path=os.environ.get("LM_EVAL_OUTPUT_PATH", ""),
                    checkpoint_dir=getattr(self, "_vllm_checkpoint_dir", None),
                    default_model_tag="native",
                )
                or getattr(self, "_vllm_output_root", None)
                or os.getcwd()
            )
        os.makedirs(debug_dir, exist_ok=True)
        legacy_path = os.path.join(debug_dir, "generate_debug_cases.jsonl")
        run_path = getattr(self, "_generate_debug_run_path", None)
        if not run_path:
            run_id = getattr(self, "_generate_debug_run_id", "") or datetime.now().strftime("%Y%m%d_%H%M%S")
            run_path = os.path.join(debug_dir, f"generate_debug_cases_{run_id}.jsonl")
            self._generate_debug_run_path = run_path
        to_write = rows if max_cases < 0 else rows[: max_cases - dumped]
        for out_path in [legacy_path, run_path]:
            if not out_path:
                continue
            with open(out_path, "a", encoding="utf-8") as f:
                for row in to_write:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._generate_debug_dumped = dumped + len(to_write)
        self._last_generate_debug.extend(to_write)
        if not bool(getattr(self, "_generate_debug_notified", False)):
            print(f"Saved generate debug cases to {legacy_path}")
            if run_path and run_path != legacy_path:
                print(f"Saved generate debug cases to {run_path}")
            self._generate_debug_notified = True

    def _append_niah_debug_rows(self, rows: List[dict]) -> None:
        """Persist NIAH generate_until debug cases (rank0 only)."""
        if not rows:
            return
        if self._distributed_args.rank != 0:
            return
        debug_dir = getattr(self, "_niah_debug_dir", "") or ""
        if not debug_dir:
            return
        max_cases = int(getattr(self, "_niah_debug_max_cases", 0) or 0)
        # 0 disables, <0 means "unlimited"
        if max_cases == 0:
            return
        dumped = int(getattr(self, "_niah_debug_dumped", 0) or 0)
        if max_cases > 0 and dumped >= max_cases:
            return
        os.makedirs(debug_dir, exist_ok=True)
        legacy_path = os.path.join(debug_dir, "niah_debug_cases.jsonl")
        run_path = getattr(self, "_niah_debug_run_path", None)
        if not run_path:
            run_id = getattr(self, "_niah_debug_run_id", "") or datetime.now().strftime("%Y%m%d_%H%M%S")
            run_path = os.path.join(debug_dir, f"niah_debug_cases_{run_id}.jsonl")
            self._niah_debug_run_path = run_path
        to_write = rows if max_cases < 0 else rows[: max_cases - dumped]
        for out_path in [legacy_path, run_path]:
            if not out_path:
                continue
            with open(out_path, "a", encoding="utf-8") as f:
                for row in to_write:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._niah_debug_dumped = dumped + len(to_write)
        if not bool(getattr(self, "_niah_debug_notified", False)):
            print(f"Saved NIAH debug cases to {legacy_path}")
            if run_path and run_path != legacy_path:
                print(f"Saved NIAH debug cases to {run_path}")
            self._niah_debug_notified = True

    # ---- Tokenization helpers ----
    def tok_encode(self, string: str, add_special_tokens: Optional[bool] = None,add_thinking_tokens: Optional[bool] = False, **kwargs) -> List[int]:
        # native tokenizer already includes BOS/EOS control; keep minimal
        if add_thinking_tokens:
            string = string + "<think>"
            return self._tokenizer.encode(string, bos=False, eos=False)
        else:
            return self._tokenizer.encode(string, bos=False, eos=False)

    def _split_contexts_to_spans(self, contexts: Optional[List[str]], span_len: int) -> List[List[int]]:
        from .reconstruct import _split_contexts_to_spans as _impl

        return _impl(self, contexts, span_len)

    def _format_chat(
        self,
        user_text: str,
        assistant_text: Optional[str] = None,
        add_generation_prompt: Optional[bool] = None,
        contexts: Optional[List[str]] = None,
        max_spans: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
            Optionally wrap text with chat template. If assistant_text is None, will
            produce a prompt that expects model generation; otherwise returns a full
            conversation with assistant content included.
        """
        
        if not self._chat_use_template:
            text = user_text if assistant_text is None else user_text + "\n" + assistant_text
            tokens = self._tokenizer.encode(text, bos=False, eos=False)
            return {
                "n_spans": 0,
                "total_comp_slots": 0,
                "total_encoder_tokens": 0,
                "comp_offsets": [0],
                "available": max(0, int(self.decoder_budget) - int(len(tokens))),
                "decoder_prefix": text,
                "decoder_prefix_text": text,
                "decoder_prefix_tokens": tokens,
                "comp_mask": [False] * len(tokens),
            }
            # return user_text if assistant_text is None else user_text + "\n" + assistant_text
        
        if self._chat_template_version == "v2":
            try:
                add_gen = self._chat_add_generation_prompt if add_generation_prompt is None else add_generation_prompt
                messages = [{"role": "user", "content": user_text}]
                if assistant_text is not None:
                    messages.append({"role": "assistant", "content": assistant_text})
                    add_gen = False if add_generation_prompt is None else add_generation_prompt
                text = self._tokenizer.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_gen)
                tokens = self._tokenizer.encode(text, bos=False, eos=False)
                return {
                    "n_spans": 0,
                    "total_comp_slots": 0,
                    "total_encoder_tokens": 0,
                    "comp_offsets": [0],
                    "available": max(0, int(self.decoder_budget) - int(len(tokens))),
                    "decoder_prefix": text,
                    "decoder_prefix_text": text,
                    "decoder_prefix_tokens": tokens,
                    "comp_mask": [False] * len(tokens),
                }
                # return self._tokenizer.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_gen)
            except Exception as e:
                print(f"WARNING: Failed to apply chat template, Error: {e}", file=sys.stderr)
                
                # fallback to raw text on any failure
                # return user_text if assistant_text is None else user_text + "\n" + assistant_text
            text = user_text if assistant_text is None else user_text + "\n" + assistant_text
            tokens = self._tokenizer.encode(text, bos=False, eos=False)
            return {
                "n_spans": 0,
                "total_comp_slots": 0,
                "total_encoder_tokens": 0,
                "comp_offsets": [0],
                "available": max(0, int(self.decoder_budget) - int(len(tokens))),
                "decoder_prefix": text,
                "decoder_prefix_text": text,
                "decoder_prefix_tokens": tokens,
                "comp_mask": [False] * len(tokens),
            }
        elif self._chat_template_version == "v3":
            """
            V3 chat template layout:
                <|im_start|>memory\n[BOM comp_slots EOM]*n<|im_end|>\n
                <|im_start|>user\n[query_tokens]<|im_end|>\n
                <|im_start|>assistant\n[generation...]<|im_end|>
                
            Each span takes (2 + num_compression_tokens) tokens in decoder:
                1 (BOM) + num_comp + 1 (EOM) = 2 + num_comp
                
            """
            bom_id = BEGIN_OF_MEMORY_INDEX
            eom_id = END_OF_MEMORY_INDEX
            memory_start = self._tokenizer.encode("<|im_start|>memory\n", bos=False, eos=False)
            user_start = self._tokenizer.encode("<|im_start|>user\n", bos=False, eos=False)
            assistant_start = self._tokenizer.encode("<|im_start|>assistant\n", bos=False, eos=False)
            im_end = self._tokenizer.encode("<|im_end|>\n", bos=False, eos=False)
            query_tokens = self._tokenizer.encode(user_text, bos=False, eos=False)
            assistant_tokens = self._tokenizer.encode(assistant_text, bos=False, eos=False) if assistant_text else []
            spans = self._split_contexts_to_spans(contexts, self._max_mem_span_len)
            # max_spans == 0 means "no memory spans" (still emit the chat scaffold).
            if max_spans is not None and max_spans >= 0 and len(spans) > max_spans:
                if bool(getattr(self, "_verbose_compress", False)):
                    print(f"Truncating memory spans from {len(spans)} to {max_spans}")
                    print("budget:", self.decoder_budget, "span occupancy:", (2 + self._num_compression_tokens) * max_spans)
                spans = [] if max_spans == 0 else spans[-max_spans:]
            n_spans = len(spans)
            comp_tokens: List[int] = []
            comp_mask: List[bool] = []

            comp_mask.extend([False] * len(memory_start))

            for _ in spans:
                comp_tokens.append(bom_id)
                comp_tokens.extend([0] * self._num_compression_tokens)
                comp_tokens.append(eom_id)
                comp_mask.append(False)
                comp_mask.extend([True] * self._num_compression_tokens)
                comp_mask.append(False)

            memory_tokens = memory_start + comp_tokens + im_end
            comp_mask.extend([False] * len(im_end))
            boq_tokens = [BEGIN_OF_QUERY_INDEX] if self._add_boq_index else []
            user_tokens = user_start + boq_tokens + query_tokens + im_end
            comp_mask.extend([False] * len(user_tokens))
            
            # Always open an assistant turn; if `assistant_text` is provided, prefill
            # it so generation continues from that prefix (no <|im_end|> here).
            ret_assistant_tokens = assistant_start + assistant_tokens
            comp_mask.extend([False] * len(ret_assistant_tokens))
            
            
            total_comp_slots = n_spans * self._num_compression_tokens
            decoder_prefix_tokens = memory_tokens + user_tokens + ret_assistant_tokens
            available = int(self.decoder_budget) - int(len(decoder_prefix_tokens))
            total_encoder_tokens = sum(len(sp) for sp in spans)
            comp_offsets = [i * self._num_compression_tokens for i in range(n_spans + 1)]
            
            # Ensure comp_mask matches token layout.
            if len(comp_mask) != len(decoder_prefix_tokens):
                raise RuntimeError(
                    f"Internal error: chat comp_mask length {len(comp_mask)} != prefix token length {len(decoder_prefix_tokens)}"
                )

            text = self._tokenizer.decode_w_special_tokens(decoder_prefix_tokens)

            return {
                "n_spans": n_spans,
                "total_comp_slots": total_comp_slots,
                "total_encoder_tokens": total_encoder_tokens,
                "comp_offsets": comp_offsets,
                "available": max(0, available),
                "decoder_prefix": text,
                "decoder_prefix_text": text,
                "decoder_prefix_tokens": decoder_prefix_tokens,
                "comp_mask": comp_mask,
            }

    def _maybe_tail_truncate_prompt_embeds(
        self,
        *,
        idx: int,
        embeds: torch.Tensor,
        vllm_max_len: int,
        embeds_meta: Optional[Dict[str, Any]],
        target_max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[str]]:
        from .reconstruct import _maybe_tail_truncate_prompt_embeds as _impl

        return _impl(
            self,
            idx=idx,
            embeds=embeds,
            vllm_max_len=vllm_max_len,
            embeds_meta=embeds_meta,
            target_max_len=target_max_len,
        )

    def tok_decode(self, tokens: List[int]) -> str:
        return self._tokenizer.decode(tokens)

    def tok_decode_w_special_tokens(self, tokens: List[int]) -> str:
        return self._tokenizer.decode_w_special_tokens(tokens)

    # NOTE: scoring/verifier helpers moved to `scoring_mixin.py`

    def get_full_text_apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        
        return self._tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=False)

    def tok_batch_encode(self, strings: List[str], left_truncate_len: Optional[int] = None, **kwargs):
        tokens = [self._tokenizer.encode(s, bos=False, eos=False) for s in strings]
        if left_truncate_len is not None:
            tokens = [t[-left_truncate_len:] for t in tokens]
        max_len = max(len(t) for t in tokens)
        padded = [t + [self.pad_token_id] * (max_len - len(t)) for t in tokens]
        tensor = torch.tensor(padded, device=self.device, dtype=torch.long)
        return tensor, tensor  # attention mask not used


    def _compress_multi_batches_with_progress(
        self,
        enc_tokens_mb: List[torch.Tensor],
        enc_ctx_mb: List[Dict[str, torch.Tensor]],
        *,
        desc: str,
    ) -> torch.Tensor:
        from .reconstruct import _compress_multi_batches_with_progress as _impl

        return _impl(self, enc_tokens_mb, enc_ctx_mb, desc=desc)



    def _compress_plain_sequence(self, tokens: torch.Tensor, num_comp: Optional[int] = None) -> torch.Tensor:
        from .reconstruct import _compress_plain_sequence as _impl

        return _impl(self, tokens, num_comp)



    # ---- Core model calls ----
    @torch.no_grad()
    def _model_call(self, inps: torch.Tensor, **kwargs) -> torch.Tensor:
        original_shape = inps.shape
        ctx = torch.autocast(device_type="cuda", dtype=self._dtype)
        cu_seqlens = torch.arange(0, original_shape[0] + 1, device=inps.device, dtype=torch.int32) * original_shape[1]
        positions = torch.arange(original_shape[1], device=inps.device, dtype=torch.int32)[:, None].repeat(original_shape[0], 1).flatten()
        context = {
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": cu_seqlens,
            "max_seqlen_q": original_shape[1],
            "max_seqlen_k": original_shape[1],
            "positions": positions,
        }
        with ctx:
            if hasattr(self.model, "compression_embeddings"):
                # Build dummy encoder/decoder contexts for compression model when used as plain LM
                decoder_context = dict(context)
                decoder_context["compression_token_mask"] = torch.zeros_like(positions, dtype=torch.bool)
                x_tokens = _token_embed(self.model, inps.flatten()).to(self._dtype)
                h = x_tokens
                for layer in self.model.layers:
                    h = layer(h, context=decoder_context)
                h = self.model.norm(h)
                logits = h
            else:
                # Two supported non-compression backends:
                #  1) Native decoder-only checkpoints (our `Model`): expects flattened tokens + `context=...`.
                #  2) HF checkpoints (transformers PreTrainedModel): expects `input_ids` (B, T) and does
                #     NOT accept the native `context` dict.
                is_hf_model = False
                try:
                    from transformers.modeling_utils import PreTrainedModel  # type: ignore

                    is_hf_model = isinstance(self.model, PreTrainedModel)
                except Exception:
                    is_hf_model = False

                if is_hf_model:
                    pad_id = self.pad_token_id if self.pad_token_id is not None else self.eot_token_id
                    attention_mask = (inps != pad_id).to(dtype=torch.long)
                    out = self.model(
                        input_ids=inps,
                        attention_mask=attention_mask,
                        use_cache=False,
                        **kwargs,
                    )
                    logits = out.logits
                    return logits

                logits = self.model(inps.flatten(), context=context, **kwargs)
        return logits.view(original_shape[0], original_shape[1], -1)

    @torch.no_grad()
    def _model_logits(self, logits: torch.Tensor) -> torch.Tensor:
        ctx = torch.autocast(device_type="cuda", dtype=self._dtype)
        with ctx:
            if hasattr(self.model, "compression_embeddings"):
                # Compression checkpoints: `_model_call` returns hidden states [bs, seq, d_model].
                proj = self.model.output(logits)  # [bs, seq, vocab]
            else:
                # Non-compression checkpoints: `Model.forward` already applies `self.output`,
                # so `_model_call` returns vocab logits [bs, seq, vocab] and we must NOT
                # project again.
                proj = logits
        if self._model_parallel_group is not None:
            from distributed.tensor_parallel import gather_from_model_parallel_region

            proj = gather_from_model_parallel_region(proj, self._model_parallel_group)
        proj = F.log_softmax(proj.float(), dim=-1)
        return proj

    @torch.no_grad()
    def _model_generate(self, context: torch.Tensor, max_length: int, **generation_kwargs) -> torch.Tensor:
        if hasattr(self.model, "compression_embeddings"):
            # Simple greedy generation for compression model: treat prompt as both encoder/decoder input,
            # no compression slots inserted. This is a plain AR decode without cached KV.
            bsz = context.size(0)
            pad_id = self.pad_token_id if self.pad_token_id is not None else self.eot_token_id
            tokens = context.tolist()
            tokens = [t[: t.index(pad_id)] if pad_id in t else t for t in tokens]
            generated_lists: List[List[int]] = [[] for _ in range(bsz)]
            max_new = max(0, max_length - max(len(t) for t in tokens))

            for _ in range(max_new):
                next_tokens = []
                for i, t in enumerate(tokens):
                    # If this sample already hit max_length, skip
                    if len(t) >= max_length:
                        next_tokens.append(self.eot_token_id)
                        continue
                    enc = torch.tensor(t, device=self.device, dtype=torch.long)
                    enc_ctx = {
                        "cu_seqlens_q": torch.tensor([0, enc.numel()], device=self.device, dtype=torch.int32),
                        "cu_seqlens_k": torch.tensor([0, enc.numel()], device=self.device, dtype=torch.int32),
                        "max_seqlen_q": enc.numel(),
                        "max_seqlen_k": enc.numel(),
                        "positions": torch.arange(enc.numel(), device=self.device, dtype=torch.int32),
                        "encoder_mem_mask": torch.zeros(enc.numel(), device=self.device, dtype=torch.bool),
                    }
                    dec_ctx = {
                        "cu_seqlens_q": torch.tensor([0, enc.numel()], device=self.device, dtype=torch.int32),
                        "cu_seqlens_k": torch.tensor([0, enc.numel()], device=self.device, dtype=torch.int32),
                        "max_seqlen_q": enc.numel(),
                        "max_seqlen_k": enc.numel(),
                        "positions": torch.arange(enc.numel(), device=self.device, dtype=torch.int32),
                        "compression_token_mask": torch.zeros(enc.numel(), device=self.device, dtype=torch.bool),
                    }
                    with torch.autocast(device_type="cuda", dtype=self._dtype):
                        x_tokens = _token_embed(self.model, enc).to(self._dtype)
                        h = x_tokens
                        for layer in self.model.layers:
                            h = layer(h, context=dec_ctx)
                        h = self.model.norm(h)
                        # Only project the last token to avoid allocating logits for the full
                        # prompt length (which can be extremely large for long-context tasks).
                        logits_last = self.model.output(h[-1:]).float()
                    nxt = int(torch.argmax(logits_last[0]))
                    next_tokens.append(nxt)
                # append tokens
                for i, nxt in enumerate(next_tokens):
                    if len(tokens[i]) < max_length:
                        tokens[i].append(nxt)
                        generated_lists[i].append(nxt)

            max_gen = max((len(g) for g in generated_lists), default=0)
            output = torch.full((bsz, max_gen), pad_id, device=self.device, dtype=torch.long)
            for i, g in enumerate(generated_lists):
                if not g:
                    continue
                output[i, : len(g)] = torch.tensor(g, device=self.device, dtype=torch.long)
            return output
        # HF / transformers checkpoints: use `generate()` (native checkpoints rely on
        # custom KV-cache + `self.model.args` and do not implement HF generate).
        if hasattr(self.model, "generate") and not hasattr(self.model, "args"):
            pad_id = self.pad_token_id if self.pad_token_id is not None else self.eot_token_id
            # `tok_batch_encode` right-pads with pad_id (often eos). Use attention_mask
            # so HF generate ignores padding tokens.
            attention_mask = (context != pad_id).to(dtype=torch.long)
            prompt_lens = attention_mask.sum(dim=1).to(dtype=torch.long)

            temperature = float(generation_kwargs.get("temperature", 0.0) or 0.0)
            top_p = float(generation_kwargs.get("top_p", 1.0) or 1.0)
            do_sample = bool(temperature > 0.0)

            gen_kwargs: Dict[str, Any] = {
                "max_length": int(max_length),
                "do_sample": bool(do_sample),
                "pad_token_id": int(pad_id),
                "eos_token_id": int(self.eot_token_id),
                "use_cache": True,
            }
            if do_sample:
                gen_kwargs["temperature"] = float(temperature)
                gen_kwargs["top_p"] = float(top_p)

            outputs = self.model.generate(
                input_ids=context,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
            # `generate` returns prompt+continuation; convert to per-sample continuation-only.
            gen_only: List[List[int]] = []
            for i in range(int(outputs.shape[0])):
                start = int(prompt_lens[i].item()) if i < int(prompt_lens.numel()) else 0
                gen_only.append(outputs[i, start:].tolist())
            max_new = max((len(g) for g in gen_only), default=0)
            if max_new <= 0:
                return torch.empty((int(outputs.shape[0]), 0), device=self.device, dtype=torch.long)
            out = torch.full((int(outputs.shape[0]), max_new), int(pad_id), device=self.device, dtype=torch.long)
            for i, g in enumerate(gen_only):
                if not g:
                    continue
                out[i, : len(g)] = torch.tensor(g, device=self.device, dtype=torch.long)
            return out
        bsz = context.size(0)
        tokens = context.tolist()
        tokens = [t[: t.index(self.pad_token_id)] if self.pad_token_id in t else t for t in tokens]
        seqlens = torch.tensor([len(t) for t in tokens], device=self.device, dtype=torch.int32)
        max_seqlen = seqlens.max().item()
        generation_length = max_length - max_seqlen

        eos_reached = torch.tensor([False] * bsz, device=self.device)
        PAGE_BLOCK_SIZE = 256
        MAX_NUM_TOKENS = 262144
        SWA_NUM_PAGES = self.model.args.yoco_window_size // PAGE_BLOCK_SIZE
        max_length = (max_length + PAGE_BLOCK_SIZE - 1) // PAGE_BLOCK_SIZE * PAGE_BLOCK_SIZE
        max_length_local = (SWA_NUM_PAGES + 1) * PAGE_BLOCK_SIZE
        MAX_PREFILL_BATCH = MAX_NUM_TOKENS // max_length_local
        assert MAX_PREFILL_BATCH > 0, f"MAX_PREFILL_BATCH: {MAX_PREFILL_BATCH}, max_length_local: {max_length_local}, MAX_NUM_TOKENS: {MAX_NUM_TOKENS}"
        kv_cache = create_kv_cache(self.model.args, bsz, max_length, max_length_local, self._dtype, self.device, page_block_size=PAGE_BLOCK_SIZE)
        last_page_table = torch.zeros(bsz, device=self.device, dtype=torch.int32)
        output = torch.zeros(bsz, generation_length, dtype=torch.long, device=self.device).fill_(self.pad_token_id)
        ctx = torch.autocast(device_type="cuda", dtype=self._dtype)
        for cur_pos in range(generation_length):
            if cur_pos == 0:
                last_logits = torch.zeros(bsz, self.model.args.vocab_size, device=self.device, dtype=torch.float16)
                for batch_start in range(0, bsz, MAX_PREFILL_BATCH):
                    batch_end = min(batch_start + MAX_PREFILL_BATCH, bsz)
                    prefill_tokens = torch.cat([torch.tensor(tokens[b], device=self.device) for b in range(batch_start, batch_end)], dim=0)
                    batch_seqlens = seqlens[batch_start:batch_end]
                    batch_idx = torch.cat([torch.full((seqlen,), i + batch_start, device=self.device, dtype=torch.int32) for i, seqlen in enumerate(batch_seqlens)], dim=0)
                    batch_positions = torch.cat([torch.arange(0, seqlen, device=self.device, dtype=torch.int32) for seqlen in batch_seqlens], dim=0)
                    cu_seqlens = torch.cat([torch.tensor([0], device=self.device, dtype=torch.int32), batch_seqlens.cumsum(dim=0).to(torch.int32)], dim=0)
                    slot_mapping = batch_idx * max_length + batch_positions
                    block_tables = torch.arange(batch_start, batch_end, device=self.device, dtype=torch.int32)[:, None] * max_length // PAGE_BLOCK_SIZE + torch.arange(max_length // PAGE_BLOCK_SIZE, device=self.device, dtype=torch.int32)
                    skip_length_local = torch.maximum(torch.zeros_like(batch_seqlens), batch_seqlens // PAGE_BLOCK_SIZE - SWA_NUM_PAGES) * PAGE_BLOCK_SIZE
                    slot_mapping_local = batch_positions - skip_length_local[batch_idx]
                    slot_mapping_local = batch_idx * max_length_local + torch.where(slot_mapping_local < 0, -1, slot_mapping_local)
                    slot_mapping_prev = torch.full(((batch_end - batch_start) * max_length_local,), -1, device=self.device, dtype=torch.int32)
                    slot_mapping_curr = torch.arange(0, cu_seqlens[-1], device=self.device, dtype=torch.int32)
                    block_tables_local = torch.arange(batch_start, batch_end, device=self.device, dtype=torch.int32)[:, None] * (SWA_NUM_PAGES + 1) + torch.arange(SWA_NUM_PAGES + 1, device=self.device, dtype=torch.int32)
                    last_page_table[batch_start:batch_end] = (batch_seqlens - skip_length_local + 1) // PAGE_BLOCK_SIZE
                    attention_context = {
                        "kv_cache": kv_cache,
                        "cu_seqlens_q": cu_seqlens,
                        "cu_seqlens_k": cu_seqlens,
                        "max_seqlen_q": batch_seqlens.max().item(),
                        "max_seqlen_k": batch_seqlens.max().item(),
                        "positions": batch_positions,
                        "slot_mapping": slot_mapping,
                        "block_tables": block_tables,
                        "slot_mapping_local": slot_mapping_local,
                        "slot_mapping_prev": slot_mapping_prev,
                        "slot_mapping_curr": slot_mapping_curr,
                        "block_tables_local": block_tables_local,
                    }
                    with ctx:
                        logits = self.model(prefill_tokens, context=attention_context, last_hidden_only=True)
                        last_logits[batch_start:batch_end] = self.model.output(logits[batch_seqlens.cumsum(dim=0) - 1])
            else:
                next_input = torch.where(eos_reached, 0, output[:, cur_pos - 1])
                positions = seqlens + cur_pos - 1
                slot_mapping = torch.arange(bsz, device=self.device) * max_length + positions
                block_tables = torch.arange(bsz * max_length // PAGE_BLOCK_SIZE, device=self.device, dtype=torch.int32).view(bsz, max_length // PAGE_BLOCK_SIZE)
                first_block_table_local = torch.where(positions >= max_length_local, (last_page_table + 1) % (SWA_NUM_PAGES + 1), 0)
                block_tables_local = (torch.arange(SWA_NUM_PAGES + 1, device=self.device, dtype=torch.int32) + first_block_table_local[:, None]) % (SWA_NUM_PAGES + 1)
                block_tables_local = block_tables_local + torch.arange(bsz, device=self.device, dtype=torch.int32)[:, None] * (SWA_NUM_PAGES + 1)
                cache_seqlens_local = torch.where(positions >= max_length_local, self.model.args.yoco_window_size + (positions + 1) % PAGE_BLOCK_SIZE, positions + 1)
                slot_mapping_local = (torch.arange(bsz, device=self.device, dtype=torch.int32) * (SWA_NUM_PAGES + 1) + last_page_table) * PAGE_BLOCK_SIZE + (positions % PAGE_BLOCK_SIZE)
                last_page_table = (last_page_table + ((positions + 1) % PAGE_BLOCK_SIZE == 0).int()) % (SWA_NUM_PAGES + 1)
                attention_context = {
                    "kv_cache": kv_cache,
                    "positions": positions,
                    "cache_seqlens": positions + 1,
                    "slot_mapping": slot_mapping,
                    "block_tables": block_tables,
                    "cache_seqlens_local": cache_seqlens_local,
                    "slot_mapping_local": slot_mapping_local,
                    "block_tables_local": block_tables_local,
                }
                with ctx:
                    last_logits = self.model(next_input, context=attention_context)
            if "temperature" in generation_kwargs and generation_kwargs["temperature"] > 0:
                probs = torch.softmax(last_logits / generation_kwargs.get("temperature", 1.0), dim=-1)
                top_p = generation_kwargs.get("top_p", 1.0)
                probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
                probs_sum = torch.cumsum(probs_sort, dim=-1)
                probs_sort[probs_sum - probs_sort > top_p] = 0.0
                probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
                next_tokens = torch.multinomial(probs_sort, num_samples=1)
                next_tokens = torch.gather(probs_idx, -1, next_tokens).reshape(-1)
            else:
                next_tokens = torch.argmax(last_logits, dim=-1).reshape(-1)
            output[:, cur_pos] = torch.where(eos_reached, output[:, cur_pos], next_tokens)
            eos_reached |= next_tokens == self.eos_token_id
            if eos_reached.all():
                break
        # Return **only newly generated tokens** (pad-filled after EOS), not prompt+continuation.
        return output

    # ---- Evaluation API implementations ----

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: Optional[int] = None,
    ) -> List[Tuple[float, bool]]:
        from .likelihood import _loglikelihood_tokens as _impl

        return _impl(self, requests, disable_tqdm=disable_tqdm, override_bs=override_bs)

    def loglikelihood(self, requests, disable_tqdm: bool = False):  # type: ignore[override]
        from .likelihood import loglikelihood as _impl

        return _impl(self, requests, disable_tqdm=disable_tqdm)

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
        from .likelihood import loglikelihood_rolling as _impl

        return _impl(self, requests, disable_tqdm=disable_tqdm)

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        from .generate import generate_until as _impl

        return _impl(self, requests, disable_tqdm=disable_tqdm)

    def _generate_compress_answer(
        self,
        prompt: str,
        max_gen_len: int,
        temperature: float,
        top_p: float,
        until: Optional[List[str]],
        include_bor: bool = False,
    ) -> str:
        from .generate import _generate_compress_answer as _impl

        return _impl(self, prompt, max_gen_len, temperature, top_p, until, include_bor)

    @staticmethod
    def _truncate_until(text: str, until: Optional[List[str]]) -> str:
        if not until:
            return text
        stops = until if isinstance(until, list) else [until]
        cutoff = len(text)
        for s in stops:
            idx = text.find(s)
            if idx != -1:
                cutoff = min(cutoff, idx)
        return text[:cutoff]

    def _generate_vllm_with_compress(
        self,
        prompt: str,
        max_gen_len: int,
        temperature: Optional[float],
        top_p: Optional[float],
        until: Optional[List[str]],
    ) -> str:
        from .generate import _generate_vllm_with_compress as _impl

        return _impl(self, prompt, max_gen_len, temperature, top_p, until)

    def _generate_with_vllm_decoder(
        self,
        prompt: str,
        max_gen_len: int,
        temperature: float,
        top_p: float,
        until: Optional[List[str]],
    ) -> str:
        from .generate import _generate_with_vllm_decoder as _impl

        return _impl(self, prompt, max_gen_len, temperature, top_p, until)

    def _generate_compress_answer_vllm(
        self,
        prompt: str,
        max_gen_len: int,
        temperature: float,
        top_p: float,
        until: Optional[List[str]],
        include_bor: bool = False,
    ) -> str:
        from .generate import _generate_compress_answer_vllm as _impl

        return _impl(self, prompt, max_gen_len, temperature, top_p, until, include_bor)

    def _build_compress_prompt_embeds_batch(
        self,
        prompts: List[str],
        gen_lens: List[int],
        include_bor: bool,
        *,
        decoder_include_prompt_tokens: bool = False,
        decoder_memory_layout: str = "per_span", #"single",
        return_meta: bool = False,
        prompt_tokens_override: Optional[List[List[int]]] = None,
        not_add_boq_index: bool = False,
        query_list: Optional[List[str]] = None,
        assistant_prefix_list: Optional[List[str]] = None,
        context_list: Optional[List[str]] = None,
        
    ) -> Tuple[List[Optional[torch.Tensor]], Optional[Dict[str, List[Any]]]]:
        from .reconstruct import _build_compress_prompt_embeds_batch as _impl

        return _impl(
            self,
            prompts,
            gen_lens,
            include_bor,
            decoder_include_prompt_tokens=decoder_include_prompt_tokens,
            decoder_memory_layout=decoder_memory_layout,
            return_meta=return_meta,
            prompt_tokens_override=prompt_tokens_override,
            not_add_boq_index=not_add_boq_index,
            query_list=query_list,
            assistant_prefix_list=assistant_prefix_list,
            context_list=context_list,
        )



        # ---- compression-aware scoring ----
    

    @torch.no_grad()
    def _loglikelihood_tokens_compress_answer(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: Optional[int] = None,
    ) -> List[Tuple[float, bool]]:
        from .likelihood import _loglikelihood_tokens_compress_answer as _impl

        return _impl(self, requests, disable_tqdm=disable_tqdm, override_bs=override_bs)

    def _get_doc_and_context(self, ctx_tokens_list: List[List[int]], *, batch_start: int = 0) -> Dict[str, List[str]]:
        """Split the current batch into structured (context, question/query) fields.

        For long-context suites and compression-aware modes we prefer to operate
        on structured doc fields rather than regex-splitting the raw prompt.

        This helper reads `Instance.doc` + `Instance.task_name` values that were
        captured by `NativeCausalLM.loglikelihood()` into `_active_loglikelihood_*`
        fields, then calls `split_doc_and_query(...)` to produce:
        - context_list: the "document" portion that should be compressed
        - query_list: the question/instruction that should be kept near the answer
        - assistant_prefix_list: any assistant prefill used by chat templates

        `batch_start` is used because lm-eval calls loglikelihood in batches and
        we need to slice the saved doc list consistently.
        """
        if self._active_loglikelihood_docs is None or self._active_loglikelihood_task_names is None:
            raise RuntimeError("_active_loglikelihood_docs/task_names not set; call via NativeCausalLM.loglikelihood().")

        start = max(0, int(batch_start))
        end = start + len(ctx_tokens_list)
        docs_slice = list(self._active_loglikelihood_docs[start:end])
        tasks_slice = list(self._active_loglikelihood_task_names[start:end])
        if len(docs_slice) != len(ctx_tokens_list) or len(tasks_slice) != len(ctx_tokens_list):
            raise RuntimeError(
                "Internal error: active doc/task slices do not match batch size: "
                f"docs={len(docs_slice)}, tasks={len(tasks_slice)}, batch={len(ctx_tokens_list)} "
                f"(batch_start={start})."
            )
        if any(d is None for d in docs_slice) or any(t is None for t in tasks_slice):
            raise RuntimeError(f"Missing Instance.doc/task_name in loglikelihood batch at start index {start}.")

        task0 = str(tasks_slice[0])
        keys = get_doc_query_keys_by_task_name(task0)
        doc_key, question_key = keys["doc_key"], keys["question_key"]

        # Use decode-with-special-tokens so any BOM/EOM/BOQ markers are preserved in debug.
        prompt_list = [self._tokenizer.decode_w_special_tokens(ctx) for ctx in ctx_tokens_list]
        split_doc_and_query_results = _split_doc_and_query(
            active_lg_docs=docs_slice,
            active_tasks_names=tasks_slice,
            prompt_list=prompt_list,
            doc_key=doc_key,
            question_key=question_key,
            niah_use_bor=False,
        )
        context_list = split_doc_and_query_results["context_list"]
        question_list = split_doc_and_query_results["question_list"]
        query_list = split_doc_and_query_results["query_list"]
        if len(context_list) != len(ctx_tokens_list) or len(query_list) != len(ctx_tokens_list):
            raise RuntimeError(
                "Internal error: split_doc_and_query returned mismatched lengths: "
                f"context={len(context_list)}, query={len(query_list)}, batch={len(ctx_tokens_list)}."
            )
        return {
            "context_list": context_list,
            "question_list": question_list,
            "query_list": query_list,
            "prompt_list": prompt_list,
            "doc_key": doc_key,
            "question_key": question_key,
        }



    @torch.no_grad()
    def _loglikelihood_tokens_reconstruct_first(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        from .likelihood import _loglikelihood_tokens_reconstruct_first as _impl

        return _impl(self, requests, disable_tqdm=disable_tqdm)
