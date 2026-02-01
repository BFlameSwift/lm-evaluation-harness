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

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "arch").is_dir() and (parent / "eval_func").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return


_maybe_add_native_rag_root_to_syspath()

import json
import re
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model

from arch.model import ModelArgs, create_kv_cache
# from arch.comp_mem import CompressedMemoryModel as Model
from arch.comp_mem import MassiveCompressedMemoryModel as Model
from config import DistributedArgs
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
from eval_func.model2safetensors import convert_checkpoint
from eval_func.vllm_runner import (
    VLLMEngineWrapper,
    VLLMEngineConfig,
    VLLMDecoderManager,
    VLLMRemoteEngineWrapper,
)
from eval_func.utils import load_checkpoint_harness, _build_device_mesh
from lm_eval.models.native_doc_utils import get_doc_query_keys_by_task_name, split_doc_and_query
import math
from typing import Type, TypeVar, Mapping, Any, Dict
import inspect
import gc
import atexit
import weakref

T = TypeVar("T")

_split_doc_and_query = split_doc_and_query



def filter_kwargs_for(
    cls: Type[T],
    raw_kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Filter kwargs for a class __init__ method.
    """
    sig = inspect.signature(cls)

    # Only keep arguments that can be passed as keyword arguments
    valid_names = {
        name
        for name, param in sig.parameters.items()
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        # Usually we don't want to include self
        and name != "self"
    }

    return {k: v for k, v in raw_kwargs.items() if k in valid_names}


def filter_kwargs_for_callable(
    fn: Any,
    raw_kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Filter kwargs for an arbitrary callable (bound method/function).

    If the callable accepts **kwargs, return raw_kwargs unchanged.
    """
    try:
        sig = inspect.signature(fn)
    except Exception:
        return {}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(raw_kwargs)
    valid_names = {
        name
        for name, param in sig.parameters.items()
        if param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {k: v for k, v in raw_kwargs.items() if k in valid_names}


def _str_to_dtype(name: Optional[str]) -> torch.dtype:
    if name is None or name == "auto":
        return torch.bfloat16
    if isinstance(name, torch.dtype):
        return name
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _parse_mode(name: Optional[str]) -> str:
    if name is None:
        return "decoder"
    name = name.lower()
    if name not in {"decoder", "compress_answer", "reconstruct_first", "vllm_decoding_with_compress", "niah_generate"}:
        raise ValueError(f"Unsupported native model mode: {name}")
    return name


def _coerce_int(value: Optional[Any], default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw or raw.lower() == "none":
            return default
        try:
            return int(raw)
        except Exception:
            return default
    try:
        return int(value)
    except Exception:
        return default


def _normalize_optional_text(value: Optional[Any]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        raw = value.strip()
        if not raw or raw.lower() == "none":
            return ""
        return value
    return str(value)


def _derive_lm_eval_output_dir(
    *,
    output_path: Optional[str],
    checkpoint_dir: Optional[str],
    default_model_tag: str = "native",
) -> Optional[str]:
    """
    Mirror lm-eval's output directory layout.

    `--output_path` can be either a file (json/jsonl) or a directory. When it's a
    directory, lm-eval writes into a model-specific subfolder. We derive that same
    path so native debug artifacts are colocated with evaluator outputs.
    """
    out_path = _normalize_optional_text(output_path)
    if not out_path:
        return None
    ext = os.path.splitext(out_path)[1].lower()
    if ext in (".json", ".jsonl"):
        return os.path.dirname(out_path) or "."
    model_tag = ""
    if checkpoint_dir:
        norm = str(checkpoint_dir).rstrip("/\\")
        base = os.path.basename(norm)
        parent = os.path.basename(os.path.dirname(norm))
        model_tag = f"{parent}/{base}" if parent and parent != norm else base
    if not model_tag:
        model_tag = default_model_tag
    model_dir = re.sub(r"[\"<>:/\|\\?\*\[\]]+", "__", model_tag)
    return os.path.join(out_path, model_dir)


def _default_distributed_args() -> DistributedArgs:
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return DistributedArgs(rank=rank, local_rank=local_rank, world_size=world_size)


_NIAH_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}\b"
)
_NIAH_MAGIC_NUMBER_RE = re.compile(r"\b\d{7}\b")


def _infer_niah_needle_type(outputs: Any) -> str:
    """Infer needle type from references (numbers vs uuids)."""
    if not outputs:
        return ""
    if isinstance(outputs, (list, tuple)):
        sample = str(outputs[0]) if outputs else ""
    else:
        sample = str(outputs)
    if _NIAH_UUID_RE.search(sample):
        return "uuids"
    if _NIAH_MAGIC_NUMBER_RE.search(sample):
        return "numbers"
    return ""


def _extract_niah_needles(text: str, needle_type: str) -> List[str]:
    if not text:
        return []
    if needle_type == "uuids":
        return _NIAH_UUID_RE.findall(text)
    if needle_type == "numbers":
        return _NIAH_MAGIC_NUMBER_RE.findall(text)
    return []


def resolve_generation_kwargs(
    gen_kwargs: Mapping[str, Any],
    *,
    default_temperature: float,
    default_top_p: float = 1.0,
    override_do_sample: Optional[bool] = None,
    override_temperature: Optional[float] = None,
    override_top_p: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Resolve per-request generation kwargs into a normalized set for vLLM/torch decoding.

    Precedence (highest -> lowest):
      1) explicit overrides passed via `--model_args gen_*`
      2) task-provided `generation_kwargs` on the request
      3) model defaults (`default_temperature`, `default_top_p`)

    Note:
      - `do_sample=False` forces greedy decoding (`temperature=0.0`).
      - We keep `top_p` at 1.0 for greedy mode (top_p is irrelevant when temperature=0).
    """
    # Task defaults
    try:
        temperature = float(gen_kwargs.get("temperature", default_temperature))
    except Exception:
        temperature = float(default_temperature)
    try:
        top_p = float(gen_kwargs.get("top_p", default_top_p))
    except Exception:
        top_p = float(default_top_p)

    # Infer do_sample if not explicitly provided.
    do_sample_val = gen_kwargs.get("do_sample", None)
    do_sample = bool(do_sample_val) if do_sample_val is not None else (temperature > 0.0)

    # Apply overrides.
    if override_do_sample is not None:
        do_sample = bool(override_do_sample)
    if override_temperature is not None:
        try:
            temperature = float(override_temperature)
        except Exception:
            pass
    if override_top_p is not None:
        try:
            top_p = float(override_top_p)
        except Exception:
            pass

    # Enforce greedy behavior when not sampling.
    if not do_sample:
        temperature = 0.0
        top_p = 1.0

    return {
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
    }


def resolve_max_gen_toks(
    gen_kwargs: Mapping[str, Any],
    *,
    default_max_gen_toks: int,
    override_max_gen_toks: Optional[int] = None,
) -> int:
    """Resolve `max_gen_toks` with an optional override."""
    if override_max_gen_toks is not None:
        try:
            return int(override_max_gen_toks)
        except Exception:
            pass
    for key in ("max_gen_toks", "max_generation_length"):
        if key in gen_kwargs and gen_kwargs.get(key) is not None:
            try:
                return int(gen_kwargs[key])
            except Exception:
                continue
    return int(default_max_gen_toks)


@register_model("native")
class NativeCausalLM(TemplateLM):
    """
    Minimal lm-evaluation-harness adapter for the native arch.Model checkpoints.

    Usage example:
    --model native --model_args checkpoint_dir=/path/to/ckpt,batch_size=4,max_seq_length=8192,mode=decoder

    mode:
      - decoder (default): vanilla causal scoring on decoder tokens only.
      - compress_answer: compress the context via encoder, then score the answer conditioned on memory.
      - reconstruct_first: reconstruct the context (optionally with vLLM prompt_embeds), then score continuation PPL.
      - vllm_decoding_with_compress: use vLLM to decode the context, then compress the context conditioned on memory.
      - niah_generate: NIAH-focused generate-only path (no likelihood); uses compressed-memory decoding and optional BOR.
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
        compress_start_tokens: Optional[str] = "<think>",
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
        
        self._active_context_key = "context"
        self._active_question_key = "question"
        
        # self._vllm_reconstruct_batch_size = max(1, int(vllm_reconstruct_batch_size))
        self._vllm_reconstruct_batch_size = self._batch_size
        self._ppl_batch_size = max(1, int(ppl_batch_size)) if ppl_batch_size is not None else self._batch_size
        self._compress_threshold = max(1, int(compress_threshold))
        self._compress_chunk = max(1, int(compress_chunk))
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
            model, tokenizer, _, device_mesh = load_checkpoint_harness(checkpoint_dir, distributed_args, tokenizer_path)

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

        self.model = model.to(dtype=self._dtype, device=self._device)
        self.model.eval()
        self._tokenizer = tokenizer
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

        # Optional vLLM init for reconstruction speedup or decoder fast path (decoder-only, prompt_embeds)
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
        self._use_remote_vllm = bool(self._vllm_server_host and self._vllm_server_port)
        
        # NOTE: vLLM initialization is expensive and not needed for pure torch loglikelihood
        # runs (e.g., multiple-choice scoring). Defer vLLM init until first use.
        
    def _init_vllm_param(self):
        if getattr(self, "_use_remote_vllm", False):
            # A remote/persistent vLLM server owns model weights/config; skip local
            # safetensors conversion and config patching.
            return
        model_path = self._vllm_model_path
        
        # breakpoint()
        if model_path is None:
            # HF checkpoints: vLLM can load directly from checkpoint_dir (already a transformers directory).
            is_native_ckpt = self._vllm_checkpoint_dir is not None and os.path.exists(os.path.join(self._vllm_checkpoint_dir, "metadata.json"))
            if not is_native_ckpt:
                model_path = self._vllm_checkpoint_dir
            else:
                base_dir = self._vllm_output_root or self._vllm_checkpoint_dir
                if base_dir is None:
                    raise ValueError("vLLM reconstruction requires vllm_model_path or checkpoint_dir (or vllm_output_root).")
                safedir = os.path.join(base_dir, "safemodel")
                model_path = safedir
                self._vllm_model_dir  = model_path
                self._vllm_output_root = os.path.join(base_dir, "vllm_output")
                need_convert = not os.path.exists(os.path.join(safedir, "model.safetensors")) or not os.path.exists(
                    os.path.join(safedir, "config.json")
                )
                if need_convert:
                    os.makedirs(safedir, exist_ok=True)
                    convert_checkpoint(
                        checkpoint_dir=self._vllm_checkpoint_dir,
                        output_dir=safedir,
                        tokenizer_path=self._vllm_tokenizer_path,
                        dtype=self._dtype,
                        additional_kwargs={
                            "max_position_embeddings": self._vllm_max_model_len,
                            "eos_token_id": self.eos_token_id,
                            "pad_token_id": self.pad_token_id,
                            "bos_token_id": getattr(self._tokenizer, "bos_id", None),
                            "temperature": self._temperature,
                            "max_seq_len": self._max_seq_length,
                        },
                    )
                self._ensure_vllm_config(safedir)
        
    def _init_vllm(self) -> None:
        # Remote vLLM server path (no local engine init).
        if getattr(self, "_use_remote_vllm", False):
            try:
                engine = VLLMRemoteEngineWrapper(
                    host=self._vllm_server_host,
                    port=self._vllm_server_port,
                    authkey=self._vllm_server_authkey,
                    timeout=self._vllm_server_timeout,
                )
                self._vllm_manager = VLLMDecoderManager(engine_wrapper=engine)
            except Exception as e:
                print(
                    f"WARNING: Failed to init remote vLLM client, falling back to torch backend. Error: {e}",
                    file=sys.stderr,
                )
                self._vllm_manager = None
            return

        # Prepare decoder-only safetensors if path not provided
        model_path = self._vllm_model_path or getattr(self, "_vllm_model_dir", None)
        if model_path is None:
            base_dir = self._vllm_output_root or self._vllm_checkpoint_dir
            if base_dir is None:
                raise ValueError("vLLM reconstruction requires vllm_model_path or checkpoint_dir (or vllm_output_root).")
            model_path = os.path.join(base_dir, "safemodel")
        # breakpoint()
        try:
            # breakpoint()
            cfg = VLLMEngineConfig(
                model_path=model_path,
                tensor_parallel_size=self._vllm_tensor_parallel,
                dtype=self._dtype,
                max_model_len= self._vllm_max_model_len or self._max_seq_length,
                enforce_eager=bool(getattr(self, "_vllm_enforce_eager", False)),
                enable_prompt_embeds=True,
                tokenizer=self._vllm_tokenizer_path or self._vllm_checkpoint_dir,
                additional_kwargs={"gpu_memory_utilization": self._vllm_gpu_memory_utilization},
            )
            # breakpoint()
            engine = VLLMEngineWrapper(cfg)
            # breakpoint()
            self._vllm_manager = VLLMDecoderManager(
                engine_wrapper=engine,
            )
            # breakpoint()
        except Exception as e:
            print(f"WARNING: Failed to init vLLM, falling back to torch backend. Error: {e}", file=sys.stderr)
            self._vllm_manager = None
            
    def _ensure_vllm_manager(self, *, caller: str) -> None:
        """
        Best-effort lazy vLLM initialization.

        Some eval modes only need vLLM for generation/reconstruction. Initializing vLLM eagerly
        can both slow down scoring-only tasks and introduce avoidable failures.
        """
        if getattr(self, "_vllm_manager", None) is not None:
            return
        if bool(getattr(self, "_vllm_init_attempted", False)):
            return
        setattr(self, "_vllm_init_attempted", True)
        try:
            self._init_vllm_param()
            self._init_vllm()
        except Exception as e:
            self._vllm_manager = None
            print(f"WARNING: Failed to init vLLM ({caller}). Error: {e}", file=sys.stderr)
    
                
    def _ensure_vllm_config(self, safedir: str) -> None:

        cfg_path = os.path.join(safedir, "config.json")
        if not os.path.exists(cfg_path):
            return
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            return

        updated = False
        target_len = int(self._vllm_max_model_len or self._max_seq_length or 2048)
        for key in ("max_position_embeddings", "max_seq_len", "model_max_length"):
            val = cfg.get(key)
            if not isinstance(val, int) or val <= 0:
                cfg[key] = target_len
                updated = True

        if updated:
            try:
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)
            except Exception:
                pass

        gen_path = os.path.join(safedir, "generation_config.json")
        if os.path.exists(gen_path):
            try:
                with open(gen_path, "r", encoding="utf-8") as f:
                    gen_cfg = json.load(f)
            except Exception:
                gen_cfg = {}
            gen_updated = False
            max_len_val = gen_cfg.get("max_length")
            if not isinstance(max_len_val, int) or max_len_val <= 0:
                gen_cfg["max_length"] = target_len
                gen_updated = True
            if gen_updated:
                try:
                    with open(gen_path, "w", encoding="utf-8") as f:
                        json.dump(gen_cfg, f, indent=2)
                except Exception:
                    pass


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
        """
        Best-effort release of GPU memory held by `self._vllm_manager`.

        Notes:
        - vLLM does not guarantee a perfect "shutdown" API across versions.
        - This helper drops references, triggers GC, and clears CUDA caching allocator.
        """
        def _call(obj: Any, name: str) -> None:
            fn = getattr(obj, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception as e:
                    print(f"WARNING: vLLM cleanup failed calling {type(obj).__name__}.{name}(): {e}", file=sys.stderr)

        def _get(obj: Any, path: List[str]) -> Any:
            cur = obj
            for p in path:
                cur = getattr(cur, p, None)
                if cur is None:
                    return None
            return cur

        before_alloc = before_reserved = None
        before_free = before_total = None
        if verbose and torch.cuda.is_available():
            try:
                before_alloc = int(torch.cuda.memory_allocated())
                before_reserved = int(torch.cuda.memory_reserved())
                before_free, before_total = torch.cuda.mem_get_info()
            except Exception:
                pass

        mgr = getattr(self, "_vllm_manager", None)
        if mgr is None:
            return

        try:
            engine_wrapper = getattr(mgr, "engine_wrapper", None)
            if engine_wrapper is not None:
                engine = getattr(engine_wrapper, "engine", None)
                if engine is not None:
                    # Try common top-level hooks.
                    _call(engine, "shutdown")
                    _call(engine, "close")
                    _call(engine, "__del__")

                    # Try common nested engines/executors across vLLM versions.
                    for sub_path in (
                        ["llm_engine"],
                        ["llm_engine", "engine_core"],
                        ["llm_engine", "executor"],
                        ["llm_engine", "model_executor"],
                        ["engine_core"],
                        ["executor"],
                        ["model_executor"],
                    ):
                        sub = _get(engine, list(sub_path))
                        if sub is not None:
                            _call(sub, "shutdown")
                            _call(sub, "close")
                            _call(sub, "__del__")

                try:
                    # Our wrapper supports a soft shutdown (drops Python refs).
                    _call(engine_wrapper, "shutdown")
                except Exception as e:
                    print(f"WARNING: Failed to shutdown vLLM, Error: {e}", file=sys.stderr)
                try:
                    # Ensure attribute is gone to break reference cycles.
                    if hasattr(engine_wrapper, "engine"):
                        
                        engine_wrapper.engine = None
                except Exception as e:
                    print(f"WARNING: Failed to set engine_wrapper.engine to None, Error: {e}", file=sys.stderr)
                try:
                    mgr.engine_wrapper = None
                except Exception:
                    pass

            # Optional vLLM-side cleanup when available.
            try:
                from vllm.distributed import destroy_model_parallel  # type: ignore

                destroy_model_parallel()
            except Exception as e:
                print(f"WARNING: Failed to destroy model parallel, Error: {e}", file=sys.stderr)
                pass
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel as destroy_mp2  # type: ignore

                destroy_mp2()
            except Exception:
                pass
            # vLLM (and some torch backends) may initialize torch.distributed even in
            # single-GPU runs. Avoid leaving a dangling process group.
            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:
                pass
        finally:
            if terminate_children:
                try:
                    import multiprocessing as mp

                    children = mp.active_children()
                    if verbose and children:
                        print(f"[native] vLLM shutdown: terminating {len(children)} child processes", file=sys.stderr)
                    for p in children:
                        try:
                            p.terminate()
                        except Exception:
                            pass
                    for p in children:
                        try:
                            p.join(timeout=float(terminate_timeout_s))
                        except Exception:
                            pass
                    for p in children:
                        try:
                            if p.is_alive():
                                p.kill()
                        except Exception:
                            pass
                except Exception as e:
                    print(f"WARNING: Failed to terminate child processes, Error: {e}", file=sys.stderr)

            self._vllm_manager = None
            try:
                gc.collect()
            except Exception as e:
                print(f"WARNING: Failed to collect garbage, Error: {e}", file=sys.stderr)
                pass
            if torch.cuda.is_available():
                try:
                    if synchronize:
                        torch.cuda.synchronize()
                except Exception as e:
                    print(f"WARNING: Failed to synchronize, Error: {e}", file=sys.stderr)
                try:
                    if ipc_collect:
                        torch.cuda.ipc_collect()
                except Exception as e:
                    print(f"WARNING: Failed to ipc collect, Error: {e}", file=sys.stderr)
                try:
                    if empty_cache:
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"WARNING: Failed to empty cache, Error: {e}", file=sys.stderr)

            if verbose and torch.cuda.is_available():
                try:
                    after_alloc = int(torch.cuda.memory_allocated())
                    after_reserved = int(torch.cuda.memory_reserved())
                    after_free, after_total = torch.cuda.mem_get_info()
                    print(
                        f"[native] vLLM shutdown mem: allocated {before_alloc}->{after_alloc}, "
                        f"reserved {before_reserved}->{after_reserved}, free {before_free}->{after_free} / total {after_total}",
                        file=sys.stderr,
                    )
                except Exception:
                    pass

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
        if contexts is None:
            return []
        if isinstance(contexts, str):
            contexts = [contexts]
        spans: List[List[int]] = []
        for ctx in contexts:
            if not ctx:
                continue
            toks_full = self._tokenizer.encode(ctx, bos=False, eos=False)
            if not toks_full:
                continue
            spans.extend([toks_full[i : i + span_len] for i in range(0, len(toks_full), span_len)])
        return spans

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
        """
        vLLM rejects any request whose prompt length exceeds `max_model_len`. For compression-heavy
        modes, the prompt is mostly made of repeated memory-span blocks; if we overshoot, we can
        usually recover by dropping *earlier* spans (tail-span behavior) while keeping the query.

        This is a best-effort guard rail that avoids hard crashes and is especially important for
        NIAH/RULER where needles may appear late.
        """
        try:
            prompt_len = int(getattr(embeds, "shape", [0])[0])
        except Exception:
            prompt_len = 0
        if vllm_max_len <= 0 or prompt_len <= 0:
            return embeds, None

        max_len = int(target_max_len) if target_max_len is not None else int(vllm_max_len)
        if max_len <= 0:
            max_len = int(vllm_max_len)
        if prompt_len <= max_len:
            return embeds, None

        if not embeds_meta:
            return embeds, "tail_truncation_unavailable:no_meta"

        def _get_list_value(name: str) -> Optional[Any]:
            val = embeds_meta.get(name)
            if isinstance(val, list) and 0 <= idx < len(val):
                return val[idx]
            return None

        n_spans = _coerce_int(_get_list_value("n_spans"), None)
        if n_spans is None:
            return embeds, "tail_truncation_unavailable:no_n_spans"
        n_spans = max(0, int(n_spans))

        decoder_memory_layout = str(embeds_meta.get("decoder_memory_layout") or "per_span")
        num_comp = _coerce_int(embeds_meta.get("num_comp"), None)
        if num_comp is None:
            try:
                num_comp = int(getattr(self.model.args, "num_compression_tokens", 0))
            except Exception:
                num_comp = 0
        num_comp = max(0, int(num_comp))
        if num_comp <= 0 or n_spans <= 0:
            return embeds, "tail_truncation_unavailable:no_memory_spans"

        # Determine how many spans to drop.
        overflow = int(prompt_len) - int(max_len)
        if overflow <= 0:
            return embeds, None

        if decoder_memory_layout == "single":
            span_cost = max(1, int(num_comp))
            drop_spans = int(math.ceil(float(overflow) / float(span_cost)))
        else:
            # per-span: BOM + slots + EOM
            span_cost = max(1, int(num_comp) + 2)
            drop_spans = int(math.ceil(float(overflow) / float(span_cost)))
        drop_spans = max(0, min(drop_spans, n_spans))
        if drop_spans <= 0:
            return embeds, None

        chat_v3 = bool(getattr(self, "_chat_use_template", False)) and str(
            getattr(self, "_chat_template_version", "")
        ).lower() == "v3"

        try:
            if decoder_memory_layout == "single":
                # Memory layout: [BOM] + slots + [EOM] (plus optional chat scaffold around it).
                mem_prefix = 0
                if chat_v3:
                    mem_prefix = len(self._tokenizer.encode("<|im_start|>memory\n", bos=False, eos=False))
                bom_pos = int(mem_prefix)
                slot_start = bom_pos + 1
                slot_total = int(num_comp) * int(n_spans)
                slot_end = slot_start + slot_total
                # Keep tail slots (later spans).
                drop_slots = min(slot_total, int(drop_spans) * int(num_comp))
                kept_embeds = torch.cat(
                    [
                        embeds[:slot_start],
                        embeds[slot_start + drop_slots : slot_end],
                        embeds[slot_end:],
                    ],
                    dim=0,
                )
            else:
                # Memory layout: ([BOM] + slots + [EOM]) * n_spans.
                if chat_v3:
                    mem_start_len = len(self._tokenizer.encode("<|im_start|>memory\n", bos=False, eos=False))
                    start = int(mem_start_len) + int(drop_spans) * int(span_cost)
                    kept_embeds = torch.cat([embeds[:mem_start_len], embeds[start:]], dim=0)
                else:
                    start = int(drop_spans) * int(span_cost)
                    kept_embeds = embeds[start:]
        except Exception as e:
            return embeds, f"tail_truncation_failed:{type(e).__name__}:{e}"

        new_len = int(getattr(kept_embeds, "shape", [0])[0])
        if new_len > int(vllm_max_len):
            # Still too long; caller will decide whether to skip.
            note = f"tail_truncated_but_still_too_long:{prompt_len}->{new_len}>{vllm_max_len}"
        else:
            note = f"tail_truncated:{prompt_len}->{new_len} drop_spans={drop_spans}/{n_spans}"

        # Best-effort update meta so debug rows match the actual prompt_embeds.
        try:
            new_n_spans = max(0, int(n_spans) - int(drop_spans))
            for name, value in (
                ("n_spans", new_n_spans),
                ("slots", int(new_n_spans) * int(num_comp)),
                ("prefix_lens", int(new_len)),
            ):
                lst = embeds_meta.get(name)
                if isinstance(lst, list) and 0 <= idx < len(lst):
                    lst[idx] = value
            # `flat_ctx_len` is a proxy for raw context coverage; approximate removal by span_len.
            span_len = _coerce_int(embeds_meta.get("span_len"), None)
            if span_len is not None:
                flat_lst = embeds_meta.get("flat_ctx_len")
                if isinstance(flat_lst, list) and 0 <= idx < len(flat_lst):
                    flat_lst[idx] = max(0, int(flat_lst[idx]) - int(drop_spans) * int(span_len))
        except Exception:
            pass

        return kept_embeds, note

    def tok_decode(self, tokens: List[int]) -> str:
        return self._tokenizer.decode(tokens)
    def tok_decode_w_special_tokens(self, tokens: List[int]) -> str:
        return self._tokenizer.decode_w_special_tokens(tokens)

    def _get_likelihood_prefix_tokens(self, mode: str) -> List[int]:
        if mode == "reconstruct_first":
            cached = getattr(self, "_likelihood_prefix_tokens_reconstruct", None)
            if cached is None:
                text = getattr(self, "_likelihood_prefix_reconstruct", "")
                cached = self._tokenizer.encode(text, bos=False, eos=False) if text else []
                setattr(self, "_likelihood_prefix_tokens_reconstruct", cached)
            return cached
        if mode == "compress_answer":
            cached = getattr(self, "_likelihood_prefix_tokens_compress_answer", None)
            if cached is None:
                text = getattr(self, "_likelihood_prefix_compress_answer", "")
                cached = self._tokenizer.encode(text, bos=False, eos=False) if text else []
                setattr(self, "_likelihood_prefix_tokens_compress_answer", cached)
            return cached
        return []

    @torch.no_grad()
    def _chunked_logprob_and_greedy(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        *,
        chunk_size: int = 32,
    ) -> Tuple[float, bool]:
        """
        Compute sum log-prob and greedy-match for token targets, using chunked full-vocab
        projection to avoid OOM.

        hidden: [N, d_model] (already normalized)
        targets: [N] (token ids)
        """
        n = int(targets.numel())
        if n == 0:
            return 0.0, True
        chunk_size = max(1, int(chunk_size))

        # When the full [N, vocab] projection is small, doing it in one shot
        # avoids tiny numeric drift from per-chunk reductions (and is faster).
        try:
            vocab = int(self.model.output.weight.shape[0])  # type: ignore[attr-defined]
        except Exception:
            vocab = 0
        if self._model_parallel_group is None and vocab > 0:
            # float32 logits would be ~4 bytes * N * vocab
            max_full_elems = 2_000_000
            if n * vocab <= max_full_elems:
                logits = self.model.output(hidden).float()
                logprobs = F.log_softmax(logits, dim=-1)
                invalid = (targets < 0) | (targets >= int(logprobs.shape[-1]))
                if bool(invalid.any().item()):
                    safe_tgt = targets.clone()
                    safe_tgt[invalid] = 0
                    gathered = logprobs.gather(-1, safe_tgt.unsqueeze(-1)).squeeze(-1)
                    gathered[invalid] = float("-inf")
                    total_lp = float(gathered.sum().item())
                    greedy_ok = bool((logprobs.argmax(dim=-1) == safe_tgt).all().item()) and not bool(invalid.any().item())
                else:
                    total_lp = float(logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum().item())
                    greedy_ok = bool((logprobs.argmax(dim=-1) == targets).all().item())
                return total_lp, greedy_ok

        greedy_ok = True
        total_lp = 0.0
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            h = hidden[s:e]
            t = targets[s:e]
            logits = self.model.output(h).float()
            if self._model_parallel_group is not None:
                from distributed.tensor_parallel import gather_from_model_parallel_region

                logits = gather_from_model_parallel_region(logits, self._model_parallel_group)
            logprobs = F.log_softmax(logits, dim=-1)
            invalid = (t < 0) | (t >= int(logprobs.shape[-1]))
            if bool(invalid.any().item()):
                safe_t = t.clone()
                safe_t[invalid] = 0
                gathered = logprobs.gather(-1, safe_t.unsqueeze(-1)).squeeze(-1)
                gathered[invalid] = float("-inf")
                total_lp += float(gathered.sum().item())
                greedy_ok = False
            else:
                total_lp += float(logprobs.gather(-1, t.unsqueeze(-1)).squeeze(-1).sum().item())
            if greedy_ok:
                greedy_ok = bool((logprobs.argmax(dim=-1) == t).all().item())
        return total_lp, greedy_ok

    @torch.no_grad()
    def _forward_score_token_ranges(
        self,
        *,
        seq_embeds: List[torch.Tensor],
        seq_token_ids: List[torch.Tensor],
        score_token_ranges: List[Tuple[int, int]],
        normalize_lengths: Optional[List[float]] = None,
        comp_mask_list: Optional[List[torch.Tensor]] = None,
        rows_per_chunk: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Forward a ragged batch (provided as per-sample decoder embeddings) and compute
        log-likelihood over *specified target token ranges*.

        This is intended as a reusable building block for:
        - multi-choice scoring where you reuse a shared prefix,
        - scoring only parts of a long continuation,
        - cases where you want to inject extra query text before likelihood.

        Alignment rule:
          Given `token_ids` aligned with `embeds`, scoring target tokens
          `token_ids[t0:t1]` uses logits at positions `[t0-1 : t1-1]`
          (standard next-token prediction).

        Args:
          seq_embeds: list of [L_i, d_model] tensors on CUDA (or will be moved)
          seq_token_ids: list of [L_i] token-id tensors aligned with seq_embeds
          score_token_ranges: list of (t0, t1) ranges in token indices, per sample
            - scores tokens token_ids[t0:t1]
            - requires t0 >= 1
          normalize_lengths: optional per-sample denominators for length-normalized ll
            (if None, uses #scored tokens)
          comp_mask_list: optional per-sample boolean masks for placeholder positions
            (not used by attention layers, kept for compatibility/debug)
          rows_per_chunk: chunk size for projecting hidden -> vocab (controls peak memory)

        Returns:
          {
            "per_sample": [
              {"ll": float, "ll_norm": float, "loss": float, "ppl": float|None,
               "tokens": int, "greedy": bool},
              ...
            ],
            "total": {"ll": float, "tokens": int, "loss": float|None, "ppl": float|None}
          }
        """
        n = len(seq_embeds)
        if len(seq_token_ids) != n or len(score_token_ranges) != n:
            raise ValueError(
                f"seq_embeds/seq_token_ids/score_token_ranges must have same length; "
                f"got {n}/{len(seq_token_ids)}/{len(score_token_ranges)}"
            )
        if normalize_lengths is not None and len(normalize_lengths) != n:
            raise ValueError(f"normalize_lengths must have length {n}, got {len(normalize_lengths)}")
        if comp_mask_list is not None and len(comp_mask_list) != n:
            raise ValueError(f"comp_mask_list must have length {n}, got {len(comp_mask_list)}")

        dec_lens: List[int] = []
        for i in range(n):
            e = seq_embeds[i]
            t = seq_token_ids[i]
            if e.ndim != 2:
                raise ValueError(f"seq_embeds[{i}] must be 2D [L,d]; got shape {tuple(e.shape)}")
            if t.ndim != 1:
                raise ValueError(f"seq_token_ids[{i}] must be 1D [L]; got shape {tuple(t.shape)}")
            if int(e.shape[0]) != int(t.numel()):
                raise ValueError(
                    f"seq_embeds[{i}] length {int(e.shape[0])} != seq_token_ids[{i}] length {int(t.numel())}"
                )
            dec_lens.append(int(e.shape[0]))

        if not dec_lens or sum(dec_lens) == 0:
            return {
                "per_sample": [
                    {"ll": 0.0, "ll_norm": 0.0, "loss": 0.0, "ppl": None, "tokens": 0, "greedy": True}
                    for _ in range(n)
                ],
                "total": {"ll": 0.0, "tokens": 0, "loss": None, "ppl": None},
            }

        dec_cu = torch.tensor([0] + list(torch.tensor(dec_lens).cumsum(0).tolist()), device=self.device, dtype=torch.int32)
        max_dec = max(dec_lens)
        dec_positions = torch.cat([torch.arange(l, device=self.device, dtype=torch.int32) for l in dec_lens], dim=0)

        if comp_mask_list is not None:
            comp_mask_flat = torch.cat(comp_mask_list, dim=0)
        else:
            comp_mask_flat = torch.zeros(sum(dec_lens), device=self.device, dtype=torch.bool)

        dec_ctx = {
            "cu_seqlens_q": dec_cu,
            "cu_seqlens_k": dec_cu,
            "max_seqlen_q": max_dec,
            "max_seqlen_k": max_dec,
            "positions": dec_positions,
            "compression_token_mask": comp_mask_flat,
        }

        embeds_flat = torch.cat(seq_embeds, dim=0)
        with torch.autocast(device_type="cuda", dtype=self._dtype):
            h = embeds_flat
            for layer in self.model.layers:
                h = layer(h, context=dec_ctx)
            h = self.model.norm(h)

        # Build flattened score positions and targets.
        score_pos_chunks: List[torch.Tensor] = []
        score_tgt_chunks: List[torch.Tensor] = []
        score_ranges_flat: List[Tuple[int, int]] = []
        running = 0
        for i, (t0, t1) in enumerate(score_token_ranges):
            t0 = int(t0)
            t1 = int(t1)
            if t1 <= t0:
                score_ranges_flat.append((running, running))
                continue
            if t0 < 1:
                raise ValueError(
                    f"score_token_ranges[{i}] starts at {t0}, but must be >= 1 because logits at position t0-1 "
                    "are needed to score token_ids[t0]."
                )
            if t1 > dec_lens[i]:
                raise ValueError(f"score_token_ranges[{i}] ends at {t1}, exceeds sequence length {dec_lens[i]}")

            base = int(dec_cu[i].item())
            pos0 = base + (t0 - 1)
            length = t1 - t0
            pos1 = pos0 + length

            score_pos_chunks.append(torch.arange(pos0, pos1, device=self.device, dtype=torch.long))
            score_tgt_chunks.append(seq_token_ids[i][t0:t1].to(device=self.device, dtype=torch.long))
            score_ranges_flat.append((running, running + length))
            running += length

        token_logprob = torch.empty(running, device=self.device, dtype=torch.float32)
        token_greedy_ok = torch.empty(running, device=self.device, dtype=torch.bool)

        if running > 0:
            score_pos = torch.cat(score_pos_chunks, dim=0)
            score_targets = torch.cat(score_tgt_chunks, dim=0)
            h_score = h.index_select(0, score_pos)

            if rows_per_chunk is None:
                rows_per_chunk = int(getattr(self.model.args, "cross_entropy_chunk", 8)) * 16
                rows_per_chunk = max(16, min(rows_per_chunk, 512))
            rows_per_chunk = max(8, int(rows_per_chunk))

            for off in range(0, running, rows_per_chunk):
                off2 = min(off + rows_per_chunk, running)
                h_chunk = h_score[off:off2]
                tgt_chunk = score_targets[off:off2]

                with torch.autocast(device_type="cuda", dtype=self._dtype):
                    logits_chunk = self.model.output(h_chunk)

                if self._model_parallel_group is not None:
                    from distributed.tensor_parallel import gather_from_model_parallel_region

                    logits_chunk = gather_from_model_parallel_region(logits_chunk, self._model_parallel_group)

                token_greedy_ok[off:off2] = logits_chunk.argmax(dim=-1).to(torch.long).eq(tgt_chunk)

                logits_f = logits_chunk.float()
                logprobs = F.log_softmax(logits_f, dim=-1)
                vocab = int(logprobs.shape[-1])
                invalid = (tgt_chunk < 0) | (tgt_chunk >= vocab)
                if bool(invalid.any().item()):
                    safe_tgt = tgt_chunk.clone()
                    safe_tgt[invalid] = 0
                    gathered = logprobs.gather(-1, safe_tgt.unsqueeze(-1)).squeeze(-1)
                    gathered[invalid] = float("-inf")
                    token_logprob[off:off2] = gathered
                else:
                    token_logprob[off:off2] = logprobs.gather(-1, tgt_chunk.unsqueeze(-1)).squeeze(-1)
                del logits_f, logprobs, invalid

        # Reduce to per-sample stats.
        per_sample: List[Dict[str, Any]] = []
        total_ll = 0.0
        total_toks = 0
        for i, (s, e) in enumerate(score_ranges_flat):
            s = int(s)
            e = int(e)
            nt = e - s
            if nt <= 0:
                ll = 0.0
                greedy = True
                ll_norm = 0.0
                loss = 0.0
                ppl = None
            else:
                ll = float(token_logprob[s:e].sum().item())
                greedy = bool(token_greedy_ok[s:e].all().item())
                denom = float(normalize_lengths[i]) if normalize_lengths is not None else float(nt)
                ll_norm = ll / denom if denom > 0 else ll
                loss = -ll / float(nt)
                try:
                    ppl = float(math.exp(loss))
                except OverflowError:
                    ppl = float("inf")
                total_ll += ll
                total_toks += nt
            per_sample.append({"ll": ll, "ll_norm": ll_norm, "loss": loss, "ppl": ppl, "tokens": nt, "greedy": greedy})

        if total_toks > 0:
            total_loss = -total_ll / float(total_toks)
            try:
                total_ppl = float(math.exp(total_loss))
            except OverflowError:
                total_ppl = float("inf")
        else:
            total_loss = None
            total_ppl = None

        return {"per_sample": per_sample, "total": {"ll": total_ll, "tokens": total_toks, "loss": total_loss, "ppl": total_ppl}}

    @torch.no_grad()
    def _forward_score_continuations(
        self,
        *,
        seq_embeds: List[torch.Tensor],
        cont_targets: List[torch.Tensor],
        prefix_lens: List[int],
        comp_mask_list: Optional[List[torch.Tensor]] = None,
        rows_per_chunk: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Forward a ragged batch and score the continuation tokens that start at
        `prefix_lens[i]` for each sample.
        """
        n = len(seq_embeds)
        if len(cont_targets) != n or len(prefix_lens) != n:
            raise ValueError(
                "seq_embeds/cont_targets/prefix_lens must have same length; "
                f"got {n}/{len(cont_targets)}/{len(prefix_lens)}"
            )
        if comp_mask_list is not None and len(comp_mask_list) != n:
            raise ValueError(f"comp_mask_list must have length {n}, got {len(comp_mask_list)}")

        dec_lens: List[int] = []
        for i in range(n):
            e = seq_embeds[i]
            t = cont_targets[i]
            if e.ndim != 2:
                raise ValueError(f"seq_embeds[{i}] must be 2D [L,d]; got shape {tuple(e.shape)}")
            if t.ndim != 1:
                raise ValueError(f"cont_targets[{i}] must be 1D [T]; got shape {tuple(t.shape)}")
            dec_lens.append(int(e.shape[0]))

        if not dec_lens or sum(dec_lens) == 0:
            return {
                "per_sample": [
                    {"ll": 0.0, "ll_norm": 0.0, "loss": 0.0, "ppl": None, "tokens": 0, "greedy": True}
                    for _ in range(n)
                ],
                "total": {"ll": 0.0, "tokens": 0, "loss": None, "ppl": None},
            }

        dec_cu = torch.tensor([0] + list(torch.tensor(dec_lens).cumsum(0).tolist()), device=self.device, dtype=torch.int32)
        max_dec = max(dec_lens)
        dec_positions = torch.cat([torch.arange(l, device=self.device, dtype=torch.int32) for l in dec_lens], dim=0)

        if comp_mask_list is not None:
            comp_mask_flat = torch.cat(comp_mask_list, dim=0)
        else:
            comp_mask_flat = torch.zeros(sum(dec_lens), device=self.device, dtype=torch.bool)

        dec_ctx = {
            "cu_seqlens_q": dec_cu,
            "cu_seqlens_k": dec_cu,
            "max_seqlen_q": max_dec,
            "max_seqlen_k": max_dec,
            "positions": dec_positions,
            "compression_token_mask": comp_mask_flat,
        }

        embeds_flat = torch.cat(seq_embeds, dim=0)
        with torch.autocast(device_type="cuda", dtype=self._dtype):
            h = embeds_flat
            for layer in self.model.layers:
                h = layer(h, context=dec_ctx)
            h = self.model.norm(h)

        score_pos_chunks: List[torch.Tensor] = []
        score_tgt_chunks: List[torch.Tensor] = []
        score_ranges_flat: List[Tuple[int, int]] = []
        running = 0
        for i in range(n):
            cont_len = int(cont_targets[i].numel())
            pref_len = int(prefix_lens[i])
            rel_start = pref_len - 1
            rel_end = rel_start + cont_len
            if rel_start < 0 or rel_end > dec_lens[i] - 1 or cont_len <= 0:
                score_ranges_flat.append((running, running))
                continue

            base = int(dec_cu[i].item())
            pos0 = base + rel_start
            pos1 = pos0 + cont_len
            score_pos_chunks.append(torch.arange(pos0, pos1, device=self.device, dtype=torch.long))
            score_tgt_chunks.append(cont_targets[i])
            score_ranges_flat.append((running, running + cont_len))
            running += cont_len

        token_logprob = torch.empty(running, device=self.device, dtype=torch.float32)
        token_greedy_ok = torch.empty(running, device=self.device, dtype=torch.bool)

        if running > 0:
            score_pos = torch.cat(score_pos_chunks, dim=0)
            score_targets = torch.cat(score_tgt_chunks, dim=0)
            h_score = h.index_select(0, score_pos)

            if rows_per_chunk is None:
                rows_per_chunk = int(getattr(self.model.args, "cross_entropy_chunk", 8)) * 16
                rows_per_chunk = max(8, min(rows_per_chunk, 512))
            rows_per_chunk = max(8, int(rows_per_chunk))

            for off in range(0, running, rows_per_chunk):
                off2 = min(off + rows_per_chunk, running)
                h_chunk = h_score[off:off2]
                tgt_chunk = score_targets[off:off2]

                with torch.autocast(device_type="cuda", dtype=self._dtype):
                    logits_chunk = self.model.output(h_chunk)

                if self._model_parallel_group is not None:
                    from distributed.tensor_parallel import gather_from_model_parallel_region

                    logits_chunk = gather_from_model_parallel_region(logits_chunk, self._model_parallel_group)

                token_greedy_ok[off:off2] = logits_chunk.argmax(dim=-1).to(torch.long).eq(tgt_chunk)

                logits_f = logits_chunk.float()
                lse = torch.logsumexp(logits_f, dim=-1)
                vocab = int(logits_f.shape[-1])
                invalid = (tgt_chunk < 0) | (tgt_chunk >= vocab)
                if bool(invalid.any().item()):
                    safe_tgt = tgt_chunk.clone()
                    safe_tgt[invalid] = 0
                    tgt_logits = logits_f.gather(-1, safe_tgt.unsqueeze(-1)).squeeze(-1)
                    tgt_logits[invalid] = float("-inf")
                else:
                    tgt_logits = logits_f.gather(-1, tgt_chunk.unsqueeze(-1)).squeeze(-1)
                token_logprob[off:off2] = (tgt_logits - lse)
                del logits_f, lse, tgt_logits, invalid

        per_sample: List[Dict[str, Any]] = []
        total_ll = 0.0
        total_toks = 0
        for i, (s, e) in enumerate(score_ranges_flat):
            s = int(s)
            e = int(e)
            nt = e - s
            if nt <= 0:
                ll = 0.0
                greedy = True
                ll_norm = 0.0
                loss = 0.0
                ppl = None
            else:
                ll = float(token_logprob[s:e].sum().item())
                greedy = bool(token_greedy_ok[s:e].all().item())
                ll_norm = ll / float(nt)
                loss = -ll / float(nt)
                try:
                    ppl = float(math.exp(loss))
                except OverflowError:
                    ppl = float("inf")
                total_ll += ll
                total_toks += nt
            per_sample.append({"ll": ll, "ll_norm": ll_norm, "loss": loss, "ppl": ppl, "tokens": nt, "greedy": greedy})

        if total_toks > 0:
            total_loss = -total_ll / float(total_toks)
            try:
                total_ppl = float(math.exp(total_loss))
            except OverflowError:
                total_ppl = float("inf")
        else:
            total_loss = None
            total_ppl = None

        return {"per_sample": per_sample, "total": {"ll": total_ll, "tokens": total_toks, "loss": total_loss, "ppl": total_ppl}}

    @torch.no_grad()
    def _score_continuation_fixed_base(
        self,
        *,
        base_embeds: torch.Tensor,
        cont_tokens: List[int],
        base_comp_mask: Optional[torch.Tensor] = None,
        decoder_budget: Optional[int] = None,
        rows_per_chunk: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Score `cont_tokens` conditioned on a fixed `base_embeds` prefix, optionally sliding
        over long continuations when `len(base)+len(cont) > decoder_budget`.

        Returns a dict compatible with debug rows:
          {"ll": float, "greedy": bool, "tokens": int, "loss": float|None, "ppl": float|None,
           "windows": int, "rolled": bool}
        """
        budget = int(self.decoder_budget if decoder_budget is None else decoder_budget)
        base_len = int(base_embeds.shape[0])
        cont_len = int(len(cont_tokens))
        if cont_len <= 0:
            return {"ll": 0.0, "greedy": True, "tokens": 0, "loss": 0.0, "ppl": None, "windows": 0, "rolled": False}
        if base_len <= 0:
            return {"ll": float("-inf"), "greedy": False, "tokens": 0, "loss": float("inf"), "ppl": float("inf"), "windows": 0, "rolled": True}

        if base_comp_mask is None:
            base_comp_mask = torch.zeros(base_len, device=self.device, dtype=torch.bool)
        else:
            base_comp_mask = base_comp_mask.to(device=self.device, dtype=torch.bool)
            if int(base_comp_mask.numel()) != base_len:
                base_comp_mask = torch.zeros(base_len, device=self.device, dtype=torch.bool)

        avail = budget - base_len
        if avail <= 0:
            return {"ll": float("-inf"), "greedy": False, "tokens": 0, "loss": float("inf"), "ppl": float("inf"), "windows": 0, "rolled": True}

        if rows_per_chunk is None:
            rows_per_chunk = int(getattr(self.model.args, "cross_entropy_chunk", 8)) * 16
            rows_per_chunk = max(16, min(int(rows_per_chunk), 512))

        def _score_window(prefix: torch.Tensor, prefix_mask: torch.Tensor, targets: List[int], prefix_len: int) -> Tuple[float, bool]:
            if not targets:
                return 0.0, True
            t = torch.tensor(targets, device=self.device, dtype=torch.long)
            with torch.autocast(device_type="cuda", dtype=self._dtype):
                e = self.model.tok_embeddings(t).to(dtype=self._dtype)
            seq = torch.cat([prefix, e], dim=0)
            mask = torch.cat([prefix_mask, torch.zeros(int(t.numel()), device=self.device, dtype=torch.bool)], dim=0)
            out = self._forward_score_continuations(
                seq_embeds=[seq],
                cont_targets=[t],
                prefix_lens=[int(prefix_len)],
                comp_mask_list=[mask],
                rows_per_chunk=rows_per_chunk,
            )
            ps = (out.get("per_sample") or [{}])[0]
            return float(ps.get("ll", 0.0)), bool(ps.get("greedy", True))

        # Fits in one window.
        if cont_len <= avail:
            ll, greedy = _score_window(base_embeds, base_comp_mask, cont_tokens, base_len)
            tokens = cont_len
            if tokens > 0 and math.isfinite(ll):
                loss = -ll / float(tokens)
                try:
                    ppl = float(math.exp(loss))
                except OverflowError:
                    ppl = float("inf")
            else:
                loss = float("inf")
                ppl = float("inf")
            return {"ll": ll, "greedy": greedy, "tokens": tokens, "loss": loss, "ppl": ppl, "windows": 1, "rolled": False}

        # Need rolling. Require >=2 available continuation positions so we can overlap by 1 token.
        if avail < 2:
            return {"ll": float("-inf"), "greedy": False, "tokens": 0, "loss": float("inf"), "ppl": float("inf"), "windows": 0, "rolled": True}

        total_ll = 0.0
        greedy_all = True
        windows = 0

        # Window 0: score the first `avail` tokens.
        ll0, g0 = _score_window(base_embeds, base_comp_mask, cont_tokens[:avail], base_len)
        total_ll += ll0
        greedy_all = greedy_all and g0
        windows += 1

        # Subsequent windows: overlap by 1 token.
        step = avail - 1
        start = avail - 1
        while start < cont_len - 1:
            end = min(cont_len, start + avail)
            overlap_id = int(cont_tokens[start])
            overlap_t = torch.tensor([overlap_id], device=self.device, dtype=torch.long)
            with torch.autocast(device_type="cuda", dtype=self._dtype):
                overlap_e = self.model.tok_embeddings(overlap_t).to(dtype=self._dtype)
            prefix = torch.cat([base_embeds, overlap_e], dim=0)
            prefix_mask = torch.cat([base_comp_mask, torch.zeros(1, device=self.device, dtype=torch.bool)], dim=0)
            llw, gw = _score_window(prefix, prefix_mask, cont_tokens[start + 1 : end], base_len + 1)
            total_ll += llw
            greedy_all = greedy_all and gw
            windows += 1
            if end >= cont_len:
                break
            start += step

        tokens = cont_len
        if tokens > 0 and math.isfinite(total_ll):
            loss = -total_ll / float(tokens)
            try:
                ppl = float(math.exp(loss))
            except OverflowError:
                ppl = float("inf")
        else:
            loss = float("inf")
            ppl = float("inf")
        return {
            "ll": total_ll,
            "greedy": greedy_all,
            "tokens": tokens,
            "loss": loss,
            "ppl": ppl,
            "windows": windows,
            "rolled": True,
        }
    
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
        """Call `compress_multi_batches` with best-effort progress reporting.

        Some checkpoints' `compress_multi_batches` do not expose any progress hooks.
        For long contexts (thousands of spans), it can look like the run is hanging.
        When `--model_args show_compress_progress=true`, we fall back to chunking the
        micro-batches and showing a tqdm progress bar on rank0.
        """
        if not hasattr(self.model, "compress_multi_batches"):
            raise RuntimeError("Compression model missing compress_multi_batches; expected MassiveCompressedMemoryModel.")

        fn = getattr(self.model, "compress_multi_batches")
        show = bool(getattr(self, "_show_compress_progress", False)) and int(
            getattr(getattr(self, "_distributed_args", None), "rank", 0)
        ) == 0
        if not show:
            return fn(enc_tokens_mb, enc_ctx_mb)

        # Prefer native progress hooks when available.
        hook_kwargs = filter_kwargs_for_callable(fn, {"show_progress": True, "progress_desc": str(desc)})
        if hook_kwargs:
            return fn(enc_tokens_mb, enc_ctx_mb, **hook_kwargs)

        # Fallback: chunked compression with tqdm updates.
        total = int(len(enc_tokens_mb))
        if total <= 0:
            d_model = int(getattr(getattr(self.model, "args", None), "d_model", 0) or 0)
            return torch.empty((0, d_model), device=self.device, dtype=self._dtype)

        chunk_size = int(getattr(self, "_compress_progress_chunk_size", 0) or 0)
        if chunk_size <= 0:
            chunk_size = 128

        bar = tqdm(total=total, desc=str(desc), disable=False)
        outs: List[torch.Tensor] = []
        for start in range(0, total, chunk_size):
            end = min(total, start + chunk_size)
            outs.append(fn(enc_tokens_mb[start:end], enc_ctx_mb[start:end]))
            bar.update(end - start)
        bar.close()
        if not outs:
            d_model = int(getattr(getattr(self.model, "args", None), "d_model", 0) or 0)
            return torch.empty((0, d_model), device=self.device, dtype=self._dtype)
        return torch.cat(outs, dim=0)



    def _compress_plain_sequence(self, tokens: torch.Tensor, num_comp: Optional[int] = None) -> torch.Tensor:
        """
        Compress a single token sequence without inserting slots/BOM/EOM.
        Useful for streaming/iterative compression where spans are managed externally.
        """
        if num_comp is None:
            num_comp = getattr(self.model.args, "num_compression_tokens", 0)
        d_model = int(getattr(getattr(self, "model", None), "args", None).d_model) if hasattr(getattr(self, "model", None), "args") else 0
        span_limit = getattr(self.model.args, "max_mem_span_len", None)
        if span_limit is None or span_limit <= 0:
            max_len = self._max_seq_length
            span_limit = max_len - num_comp if max_len is not None else tokens.numel()
        span_limit = max(1, span_limit)
        has_multi = hasattr(self.model, "compress_multi_batches")

        if tokens.numel() == 0:
            return (
                torch.empty((0, d_model), device=self.device, dtype=self._dtype)
                if d_model > 0
                else torch.empty(0, device=self.device, dtype=self._dtype)
            )

        # we always append placeholders for compression tokens
        if num_comp <= 0:
            return (
                torch.empty((0, d_model), device=self.device, dtype=self._dtype)
                if d_model > 0
                else torch.empty(0, device=self.device, dtype=self._dtype)
            )

        enc_tokens_mb: List[torch.Tensor] = []
        enc_ctx_mb: List[Dict[str, torch.Tensor]] = []

        for i in range(0, tokens.numel(), span_limit):
            chunk = tokens[i : i + span_limit]
            # tokens + placeholders
            enc_tokens_seq = torch.cat(
                [chunk, torch.full((num_comp,), 0, device=self.device, dtype=torch.long)],
                dim=0,
            )
            enc_mask_seq = torch.cat(
                [torch.zeros_like(chunk, dtype=torch.bool), torch.ones(num_comp, device=self.device, dtype=torch.bool)],
                dim=0,
            )
            clen = enc_tokens_seq.numel()
            cu = torch.tensor([0, clen], device=self.device, dtype=torch.int32)
            enc_tokens_mb.append(enc_tokens_seq)
            enc_ctx_mb.append({
                "cu_seqlens_q": cu,
                "cu_seqlens_k": cu,
                "max_seqlen_q": clen,
                "max_seqlen_k": clen,
                "positions": torch.arange(clen, device=self.device, dtype=torch.int32),
                "encoder_mem_mask": enc_mask_seq,
            })

        with torch.autocast(device_type="cuda", dtype=self._dtype):
            if has_multi:
                return self._compress_multi_batches_with_progress(enc_tokens_mb, enc_ctx_mb, desc="compress_plain")

            # Fallback: run per-span compression sequentially to keep positions valid.
            outs: List[torch.Tensor] = []
            for t, ctx in zip(enc_tokens_mb, enc_ctx_mb):
                outs.append(self.model.compress(encoder_tokens=t, encoder_context=ctx))
            if outs:
                return torch.cat(outs, dim=0)
            return (
                torch.empty((0, d_model), device=self.device, dtype=self._dtype)
                if d_model > 0
                else torch.empty(0, device=self.device, dtype=self._dtype)
            )



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
                x_tokens = self.model.tok_embeddings(inps.flatten()).to(self._dtype)
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
                        x_tokens = self.model.tok_embeddings(enc).to(self._dtype)
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
        # Optional mode: compress the context into memory first, then score the answer.
        if self._mode == "compress_answer" and hasattr(self.model, "compression_embeddings"):
            out = self._loglikelihood_tokens_compress_answer(requests, disable_tqdm, override_bs)

            return out
        if self._mode == "reconstruct_first" and hasattr(self.model, "compression_embeddings"):
            out = self._loglikelihood_tokens_reconstruct_first(requests, disable_tqdm)

            return out

        res: List[Tuple[float, bool]] = []
        bs = override_bs or self.batch_size
        try:
            bs = int(bs)
        except Exception:
            bs = 1
        iterator = range(0, len(requests), bs)
        pbar = tqdm(iterator, disable=disable_tqdm or self.rank != 0, desc="native loglikelihood")
        for batch_start in pbar:
            chunk = requests[batch_start : batch_start + bs]
            seq_lens = [min(self.max_length, len(c) + len(cont)) for _, c, cont in chunk]
            max_seq = max(seq_lens)
            inp_batch = torch.full((len(chunk), max_seq - 1), self.pad_token_id, device=self.device, dtype=torch.long)
            tgt_batch: List[List[int]] = []
            cont_lens: List[int] = []
            tgt_full_lens: List[int] = []
            for i, (_, context_enc, continuation_enc) in enumerate(chunk):
                tokens = (context_enc + continuation_enc)[-(max_seq):]
                if len(tokens) <= 1:
                    tokens = [self.eot_token_id, self.eot_token_id]
                inp = tokens[:-1]
                tgt = tokens[1:]
                inp_batch[i, -len(inp) :] = torch.tensor(inp, device=self.device)
                tgt_batch.append(tgt)
                cont_lens.append(len(continuation_enc))
                tgt_full_lens.append(len(tgt))

            logits = self._model_call(inp_batch)
            logprobs = self._model_logits(logits)

            debug_rows: List[dict] = []
            for i, tgt in enumerate(tgt_batch):
                tgt_tensor = torch.tensor(tgt, device=self.device, dtype=torch.long)
                seq_logprobs = logprobs[i, -tgt_full_lens[i] :, :]
                cont_len = cont_lens[i]
                tail_logprobs = seq_logprobs[-cont_len:, :]
                token_logprobs = tail_logprobs.gather(-1, tgt_tensor[-cont_len:].unsqueeze(-1)).squeeze(-1)
                logprob = float(token_logprobs.sum().item())
                greedy = bool((tail_logprobs.argmax(dim=-1) == tgt_tensor[-cont_len:]).all().item())
                res.append((logprob, greedy))

                if self._save_loglikelihood_debug and self._distributed_args.rank == 0:
                    loss = None
                    ppl = None
                    if cont_len > 0 and math.isfinite(logprob):
                        loss = -float(logprob) / float(cont_len)
                        try:
                            ppl = math.exp(loss)
                        except OverflowError:
                            ppl = float("inf")

                    row = {
                        "request_index": batch_start + i,
                        "mode": "decoder",
                        "cont_len": int(cont_len),
                        "logprob": logprob,
                        "greedy": greedy,
                        "ppl": ppl,
                    }
                    if loss is not None:
                        row["loss"] = loss

                    # Best-effort: include identifiers for easier debugging.
                    try:
                        if self._active_loglikelihood_task_names:
                            task_name = self._active_loglikelihood_task_names[batch_start + i]
                            if task_name:
                                row["task_name"] = task_name
                        if self._active_loglikelihood_doc_ids:
                            doc_id = self._active_loglikelihood_doc_ids[batch_start + i]
                            if doc_id is not None:
                                row["doc_id"] = int(doc_id)
                    except Exception:
                        pass

                    try:
                        raw_cont = chunk[i][0][1] if chunk[i] and chunk[i][0] and len(chunk[i][0]) > 1 else ""
                    except Exception:
                        raw_cont = ""
                    if raw_cont:
                        row["cont_str_len"] = int(len(raw_cont))
                        row["cont_str_preview"] = raw_cont[:200]

                    try:
                        raw_ctx = chunk[i][0][0] if chunk[i] and chunk[i][0] else ""
                    except Exception:
                        raw_ctx = ""
                    if raw_ctx:
                        row["ctx_str_len"] = int(len(raw_ctx))
                        row["ctx_str_preview"] = raw_ctx[:200]

                    if cont_len > 0:
                        row["cont_tokens_len"] = int(cont_len)
                        row["cont_tokens_preview"] = list(
                            tgt_tensor[-cont_len:].detach().to("cpu").tolist()[:20]
                        )
                    debug_rows.append(row)

            self._append_loglikelihood_debug_rows(debug_rows)
        return res

    # ---------------------------------------------------------------------
    # Harness integration: keep access to structured `Instance.doc` fields.
    # ---------------------------------------------------------------------
    def loglikelihood(self, requests, disable_tqdm: bool = False):  # type: ignore[override]
        """
        Override TemplateLM.loglikelihood so we can access `Instance.doc` for
        structured prompt parts (e.g., `doc["context"]`, `doc["question"]`,
        `doc["choices"]`) inside our custom loglikelihood implementations.

        Upstream TemplateLM.loglikelihood discards Instance.doc and only forwards
        (context_str, continuation_str) + tokenized ids into `_loglikelihood_tokens`.
        """
        self._active_loglikelihood_docs = [getattr(r, "doc", None) for r in requests]
        self._active_loglikelihood_task_names = [getattr(r, "task_name", None) for r in requests]
        self._active_loglikelihood_doc_ids = [getattr(r, "doc_id", None) for r in requests]
        try:
            new_reqs: List[Tuple[Tuple[str, str], List[int], List[int]]] = []
            for req in requests:
                args = req.args
                if not isinstance(args, tuple) or len(args) < 2:
                    raise ValueError(f"Unexpected loglikelihood Instance.args: {args!r}")
                context = args[0]
                continuation = args[1]

                if context == "":
                    continuation_enc = self.tok_encode(continuation, add_special_tokens=False)
                    if not continuation_enc:
                        context_enc, continuation_enc = ([self.prefix_token_id], [])
                    else:
                        context_enc, continuation_enc = (
                            ([self.prefix_token_id], continuation_enc)
                            if self.prefix_token_id != continuation_enc[0]
                            else (continuation_enc[:1], continuation_enc[1:])
                        )
                else:
                    context_enc, continuation_enc = self._encode_pair(context, continuation)

                new_reqs.append(((context, continuation), context_enc, continuation_enc))

            return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)
        finally:
            # Clear to avoid accidental reuse across evaluations.
            self._active_loglikelihood_docs = None
            self._active_loglikelihood_task_names = None
            self._active_loglikelihood_doc_ids = None

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
        results: List[float] = []
        for (text,) in [req.args for req in requests]:
            tokens = [self.eot_token_id] + self.tok_encode(text)
            total = 0.0
            for start in range(0, len(tokens) - 1, self.max_length):
                window = tokens[start : start + self.max_length + 1]
                if len(window) <= 1:
                    continue
                inp = torch.tensor(window[:-1], device=self.device, dtype=torch.long).unsqueeze(0)
                tgt = torch.tensor(window[1:], device=self.device, dtype=torch.long)
                logits = self._model_call(inp)
                logprobs = self._model_logits(logits)[0, -len(tgt) :, :]
                token_lp = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
                total += token_lp
            results.append(total)
        return results

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        results: List[str] = []

        def _get_max_gen_tokens(gen_kwargs: dict) -> int:
            try:
                default = int(self.max_gen_toks)
            except Exception:
                default = 0
            return resolve_max_gen_toks(
                gen_kwargs,
                default_max_gen_toks=default,
                override_max_gen_toks=getattr(self, "_gen_max_gen_toks_override", None),
            )

        packed: List[Tuple[Tuple[str, dict], Optional[dict], Optional[str], Any]] = [
            (req.args, getattr(req, "doc", None), getattr(req, "task_name", None), getattr(req, "doc_id", None))
            for req in requests
        ]

        # `compress_answer` / `niah_generate` generation is implemented via vLLM prompt_embeds.
        # Falling back to torch generation can OOM because it materializes full logits for the
        # entire (potentially very long) prompt. Lazily init vLLM here so callers do not need
        # to remember to set extra flags for these modes.
        if self._mode in {"compress_answer", "reconstruct_first", "vllm_decoding_with_compress", "niah_generate"} and hasattr(
            self.model, "compression_embeddings"
        ):
            self._ensure_vllm_manager(caller=f"generate_until(mode={self._mode})")
            if self._vllm_manager is None:
                raise RuntimeError(
                    f"generate_until(mode={self._mode}) requires vLLM prompt_embeds backend for compression models. "
                    "Set vllm_max_model_len (and optionally vllm_gpu_memory_utilization) or provide a working "
                    "vllm_server_host/vllm_server_port."
                )

        def _infer_doc_id(doc: Optional[dict], doc_id: Any) -> Any:
            if doc_id is not None:
                return doc_id
            if not isinstance(doc, dict):
                return None
            for key in ("id", "_id", "doc_id", "query_id", "index"):
                if key in doc:
                    return doc.get(key)
            return None

        iterator = tqdm(
            range(0, len(packed), self.batch_size),
            disable=disable_tqdm,
            desc=f"native generate ({self._mode})",
        )
        for start in iterator:
            packed_chunk = packed[start : start + self.batch_size]
            chunk = [args for args, _, _, _ in packed_chunk]
            chunk_docs = [doc for _, doc, _, _ in packed_chunk]
            chunk_tasks = [task for _, _, task, _ in packed_chunk]
            chunk_doc_ids = [doc_id for _, _, _, doc_id in packed_chunk]
            debug_rows: List[dict] = []

            gkwargs = [g for _, g in chunk]
            if gkwargs[0].get("add_thinking_tokens", False):
                self._add_thinking_tokens = True
            if gkwargs[0].get("use_chat_template", False):
                self._chat_use_template = True
            # breakpoint()
            # Try batched vLLM paths first
            if (
                self._mode == "decoder"
                and self._vllm_manager is not None
                and not hasattr(self.model, "compression_embeddings")
            ):

                if self._chat_use_template:
                    prompts = [self._format_chat(c, add_generation_prompt=True)["decoder_prefix"] for c, _ in chunk]
                else:
                    prompts = [c for c, _ in chunk]

                resolved = resolve_generation_kwargs(
                    gkwargs[0],
                    default_temperature=self._temperature,
                    default_top_p=1.0,
                    override_do_sample=getattr(self, "_gen_do_sample_override", None),
                    override_temperature=getattr(self, "_gen_temperature_override", None),
                    override_top_p=getattr(self, "_gen_top_p_override", None),
                )
                sampling_params = {
                    "max_tokens": max(_get_max_gen_tokens(g) for g in gkwargs),
                    "temperature": resolved["temperature"],
                    "top_p": resolved["top_p"],
                }
                outputs = self._vllm_manager.engine_wrapper.generate(prompts, sampling_params)
                for i, (out, (_, gk)) in enumerate(zip(outputs, chunk)):
                    text = out.outputs[0].text if out.outputs else ""
                    text = self._truncate_until(text, gk.get("until"))
                    results.append(text)
                    prompt = prompts[i] if i < len(prompts) else ""
                    try:
                        prompt_len_tokens = len(self.tok_encode(prompt))
                    except Exception:
                        prompt_len_tokens = 0
                    debug_rows.append(
                        {
                            "task": chunk_tasks[i] if i < len(chunk_tasks) else None,
                            "doc_id": _infer_doc_id(chunk_docs[i] if i < len(chunk_docs) else None, chunk_doc_ids[i]),
                            "request_index": int(start + i),
                            "mode": str(self._mode),
                            "backend": "vllm_text",
                            "prompt": prompt,
                            "prompt_len_tokens": prompt_len_tokens,
                            "prompt_len_chars": len(prompt),
                            "generation_kwargs": gk,
                            "sampling_params": sampling_params,
                            "skip_reason": None,
                            "response": text,
                        }
                    )
                self._append_generate_debug_rows(debug_rows)
                continue

            if (
                self._mode == "decoder"
                and self._vllm_manager is not None
                and hasattr(self.model, "compression_embeddings")
            ):
                embeds: List[Optional[torch.Tensor]] = []
                prompts: List[str] = []
                gkwargs = []
                for c, gk in chunk:
                    if self._chat_use_template:
                        prompt_c = self._format_chat(c, add_generation_prompt=True)["decoder_prefix"]
                    else:
                        prompt_c = c
                    prompts.append(prompt_c)
                    tokens = self.tok_encode(prompt_c)
                    if len(tokens) == 0:
                        embeds.append(None)
                        gkwargs.append(gk)
                        continue
                    tok_tensor = torch.tensor(tokens, device=self.device, dtype=torch.long)
                    embeds.append(self.model.tok_embeddings(tok_tensor).to(dtype=self._dtype))
                    gkwargs.append(gk)
                    
                valid_indices = [i for i, e in enumerate(embeds) if e is not None]
                outs_text = [""] * len(chunk)
                if valid_indices:
                    resolved = resolve_generation_kwargs(
                        gkwargs[valid_indices[0]],
                        default_temperature=self._temperature,
                        default_top_p=1.0,
                        override_do_sample=getattr(self, "_gen_do_sample_override", None),
                        override_temperature=getattr(self, "_gen_temperature_override", None),
                        override_top_p=getattr(self, "_gen_top_p_override", None),
                    )
                    sampling_params = {
                        "max_tokens": max(_get_max_gen_tokens(gkwargs[i]) for i in valid_indices),
                        "temperature": resolved["temperature"],
                        "top_p": resolved["top_p"],
                    }
                    batch_embeds = [embeds[i] for i in valid_indices]
                    outs = self._vllm_manager.generate_from_embeddings(batch_embeds, sampling_params=sampling_params)
                    for idx, out in zip(valid_indices, outs):
                        text = out.outputs[0].text if out.outputs else ""
                        outs_text[idx] = self._truncate_until(text, gkwargs[idx].get("until"))
                results.extend(outs_text)
                for i, (_, gk) in enumerate(chunk):
                    e = embeds[i] if i < len(embeds) else None
                    prompt = prompts[i] if i < len(prompts) else ""
                    debug_rows.append(
                        {
                            "task": chunk_tasks[i] if i < len(chunk_tasks) else None,
                            "doc_id": _infer_doc_id(chunk_docs[i] if i < len(chunk_docs) else None, chunk_doc_ids[i]),
                            "request_index": int(start + i),
                            "mode": str(self._mode),
                            "backend": "vllm_embeds",
                            "prompt": prompt,
                            "prompt_len_tokens": int(e.shape[0]) if e is not None else 0,
                            "prompt_len_chars": len(prompt),
                            "generation_kwargs": gk,
                            "sampling_params": sampling_params if valid_indices else None,
                            "skip_reason": None if e is not None else "no_prompt_embeds",
                            "response": outs_text[i] if i < len(outs_text) else "",
                        }
                    )
                self._append_generate_debug_rows(debug_rows)
                continue

            if (
                self._mode in {"compress_answer", "reconstruct_first", "vllm_decoding_with_compress", "niah_generate"}
                and self._vllm_manager is not None
                and hasattr(self.model, "compression_embeddings")
            ):
                include_bor = self._mode == "reconstruct_first"
                if self._mode == "niah_generate":
                    include_bor = bool(getattr(self, "_niah_use_bor", False))
                if self._mode == "vllm_decoding_with_compress":
                    # new iterative vLLM decode with compression
                    for i, (context_str, gk) in enumerate(chunk):
                        prompt = self._format_chat(context_str, add_generation_prompt=True)["decoder_prefix"]
                        max_gen_len = _get_max_gen_tokens(gk)
                        resolved = resolve_generation_kwargs(
                            gk,
                            default_temperature=self._temperature,
                            default_top_p=1.0,
                            override_do_sample=getattr(self, "_gen_do_sample_override", None),
                            override_temperature=getattr(self, "_gen_temperature_override", None),
                            override_top_p=getattr(self, "_gen_top_p_override", None),
                        )
                        temperature = resolved["temperature"]
                        top_p = resolved["top_p"]
                        text = self._generate_vllm_with_compress(
                            prompt=prompt,
                            max_gen_len=max_gen_len,
                            temperature=temperature,
                            top_p=top_p,
                            until=gk.get("until"),
                        )

                        results.append(text)

                        prompt_dbg = prompt
                        max_chars = int(getattr(self, "_generate_debug_max_prompt_chars", 0) or 0)
                        if max_chars > 0 and len(prompt_dbg) > max_chars:
                            prompt_dbg = prompt_dbg[:max_chars] + "\n...[truncated]..."
                        try:
                            prompt_len_tokens = len(self.tok_encode(prompt))
                        except Exception:
                            prompt_len_tokens = 0
                        debug_rows.append(
                            {
                                "task": chunk_tasks[i] if i < len(chunk_tasks) else None,
                                "doc_id": _infer_doc_id(chunk_docs[i] if i < len(chunk_docs) else None, chunk_doc_ids[i]),
                                "request_index": int(start + i),
                                "mode": str(self._mode),
                                "backend": "vllm_iterative_compress",
                                "prompt": prompt_dbg,
                                "prompt_len_tokens": prompt_len_tokens,
                                "prompt_len_chars": len(prompt_dbg),
                                "generation_kwargs": gk,
                                "sampling_params": {
                                    "max_tokens": int(max_gen_len),
                                    "temperature": float(temperature),
                                    "top_p": float(top_p),
                                },
                                "skip_reason": None,
                                "response": text,
                            }
                        )
                    self._append_generate_debug_rows(debug_rows)
                    continue

                # If we have structured docs and a supported task, split (context, query) so the
                # query is *not* compressed away.
                #
                # - LongBench2: requires native_rag chat template (memory/user/assistant scaffold).
                # - RULER NIAH: allow splitting even when `use_chat_template=false`, because the
                #   raw haystack can exceed `vllm_max_model_len` but is expected to fit via spans.
                #
                # Important: in harness, padding requests can have `doc=None`. Do not disable
                # splitting for the *entire* batch if a single request is missing docs. Instead,
                # split only the valid requests and skip the padded ones.
                task0 = ""
                for t in chunk_tasks:
                    if t is not None:
                        task0 = str(t)
                        break
                task0_lower = task0.lower()
                is_niah_task = (self._mode == "niah_generate") and ("niah" in task0_lower)
                is_longbench2_task = "longbench2" in task0_lower
                is_infinitebench_task = "infinitebench" in task0_lower
                is_longbench1_task = task0_lower.startswith("longbench_") and ("longbench2" not in task0_lower)
                is_babilong_task = task0_lower.startswith("babilong")
                is_ruler_custom_task = task0_lower.startswith("ruler_custom")
                is_ruler_qa_task = task0_lower.startswith("ruler_qa_")
                split_supported = bool(
                    is_niah_task
                    or is_infinitebench_task
                    or is_longbench1_task
                    or is_longbench2_task
                    or is_babilong_task
                    or is_ruler_custom_task
                    or is_ruler_qa_task
                )

                gen_lens = [_get_max_gen_tokens(g) for g in gkwargs]
                split_data: Optional[Dict[str, List[str]]] = None
                embeds_meta: Optional[Dict[str, List[Any]]] = None
                prompts: List[str] = []
                embeds: List[Optional[torch.Tensor]] = []
                force_skip_split: List[bool] = [False] * len(chunk)
                force_skip_reasons: List[Optional[str]] = [None] * len(chunk)
                if split_supported and task0:
                    try:
                        keys = get_doc_query_keys_by_task_name(task0)
                    except Exception:
                        keys = None
                    if keys is not None:
                        subset_docs: List[dict] = []
                        subset_tasks: List[str] = []
                        subset_prompts: List[str] = []
                        subset_indices: List[int] = []
                        for i in range(len(chunk)):
                            doc = chunk_docs[i] if i < len(chunk_docs) else None
                            tname = chunk_tasks[i] if i < len(chunk_tasks) else None
                            if doc is None or tname is None or not isinstance(doc, dict):
                                force_skip_split[i] = True
                                force_skip_reasons[i] = "missing_doc_or_task"
                                continue
                            # Best-effort: skip samples that don't match the batch task.
                            if task0 and str(tname) != task0:
                                force_skip_split[i] = True
                                force_skip_reasons[i] = "mixed_task_in_batch"
                                continue
                            if keys["doc_key"] not in doc or keys["question_key"] not in doc:
                                force_skip_split[i] = True
                                force_skip_reasons[i] = "doc_missing_required_fields"
                                continue
                            subset_indices.append(i)
                            subset_docs.append(doc)
                            subset_tasks.append(str(tname))
                            subset_prompts.append(chunk[i][0])

                        if subset_indices:
                            subset_split = _split_doc_and_query(
                                active_lg_docs=subset_docs,
                                active_tasks_names=subset_tasks,
                                prompt_list=subset_prompts,
                                doc_key=keys["doc_key"],
                                question_key=keys["question_key"],
                                niah_use_bor=bool(getattr(self, "_niah_use_bor", False)),
                            )
                            # Expand back to full batch-aligned lists.
                            full_context_list = [""] * len(chunk)
                            full_query_list = [""] * len(chunk)
                            full_question_list = [""] * len(chunk)
                            full_assistant_prefix_list: Optional[List[str]] = None
                            if "assistant_prefix_list" in subset_split:
                                full_assistant_prefix_list = [""] * len(chunk)
                            for j, idx in enumerate(subset_indices):
                                if j < len(subset_split.get("context_list", [])):
                                    full_context_list[idx] = subset_split["context_list"][j]
                                if j < len(subset_split.get("query_list", [])):
                                    full_query_list[idx] = subset_split["query_list"][j]
                                if j < len(subset_split.get("question_list", [])):
                                    full_question_list[idx] = subset_split["question_list"][j]
                                if full_assistant_prefix_list is not None and j < len(
                                    subset_split.get("assistant_prefix_list", [])
                                ):
                                    full_assistant_prefix_list[idx] = subset_split["assistant_prefix_list"][j] or ""

                            split_data = {
                                "context_list": full_context_list,
                                "query_list": full_query_list,
                                "question_list": full_question_list,
                            }
                            if full_assistant_prefix_list is not None:
                                split_data["assistant_prefix_list"] = full_assistant_prefix_list

                            prompts = [""] * len(chunk)
                            build_ret = self._build_compress_prompt_embeds_batch(
                                prompts,
                                gen_lens,
                                include_bor,
                                decoder_include_prompt_tokens=False,
                                context_list=split_data["context_list"],
                                query_list=split_data["query_list"],
                                assistant_prefix_list=split_data.get("assistant_prefix_list"),
                                # Always request meta: needed for best-effort tail-span truncation
                                # when vLLM `max_model_len` is smaller than the compressed prompt.
                                return_meta=True,
                            )
                            if isinstance(build_ret, tuple):
                                embeds, embeds_meta = build_ret
                            else:
                                embeds = build_ret
                        else:
                            # Nothing to split in this batch (e.g., all padded / invalid docs).
                            # Do NOT fall back to prompt-based compression for supported tasks,
                            # because the raw prompt may exceed `vllm_max_model_len`.
                            prompts = [c for c, _ in chunk]
                            embeds = [None] * len(chunk)
                    else:
                        # Cannot determine structured doc/query keys; skip rather than risk
                        # constructing an over-long raw prompt for vLLM.
                        for i in range(len(chunk)):
                            if not force_skip_split[i]:
                                force_skip_split[i] = True
                                force_skip_reasons[i] = "split_keys_unavailable"
                        prompts = [c for c, _ in chunk]
                        embeds = [None] * len(chunk)
                else:
                    prompts = [self._format_chat(c, add_generation_prompt=True)["decoder_prefix"] for c, _ in chunk]
                    embeds = self._build_compress_prompt_embeds_batch(
                        prompts,
                        gen_lens,
                        include_bor,
                        # Best-effort: include prompt tokens so the decoder sees the question.
                        decoder_include_prompt_tokens=True,
                    )
                # vLLM enforces a hard prompt length limit (`max_model_len`). Filter/clip
                # requests so a single over-long prompt does not crash the whole run.
                vllm_max_len = int(getattr(self, "_vllm_max_model_len", 0) or 0)
                valid_indices: List[int] = []
                max_tokens_caps: Dict[int, int] = {}
                skip_reason: List[Optional[str]] = [None] * len(chunk)
                clip_note: List[Optional[str]] = [None] * len(chunk)
                for i, e in enumerate(embeds):
                    if i < len(force_skip_split) and force_skip_split[i]:
                        skip_reason[i] = force_skip_reasons[i] or "missing_doc_or_task"
                        embeds[i] = None
                        continue
                    if e is None:
                        skip_reason[i] = "no_prompt_embeds"
                        continue
                    try:
                        e_len = int(e.shape[0])
                    except Exception:
                        e_len = 0
                    if vllm_max_len > 0:
                        # vLLM validates prompt length strictly. If we exceed the budget, try
                        # a best-effort tail-span truncation (drop earlier spans, keep query).
                        reserve = 0
                        try:
                            reserve = min(int(gen_lens[i]), 256)
                        except Exception:
                            reserve = 0
                        target = int(vllm_max_len) - int(reserve)
                        if target <= 0:
                            target = int(vllm_max_len)

                        if e_len > vllm_max_len or (reserve > 0 and e_len > target):
                            new_e, note = self._maybe_tail_truncate_prompt_embeds(
                                idx=i,
                                embeds=e,
                                vllm_max_len=vllm_max_len,
                                embeds_meta=embeds_meta,
                                target_max_len=target,
                            )
                            embeds[i] = new_e
                            clip_note[i] = note
                            try:
                                e_len = int(new_e.shape[0])
                            except Exception:
                                e_len = 0

                        if e_len > vllm_max_len:
                            # Still too long after tail-truncation; skip this sample.
                            skip_reason[i] = f"prompt_too_long:{e_len}>{vllm_max_len}"
                            embeds[i] = None
                            continue
                    if vllm_max_len > 0:
                        max_tokens_caps[i] = max(0, vllm_max_len - e_len)
                    valid_indices.append(i)
                outs_text = [""] * len(chunk)
                outs_token_ids: List[Optional[List[int]]] = [None] * len(chunk)
                if valid_indices:
                    resolved = resolve_generation_kwargs(
                        gkwargs[valid_indices[0]],
                        default_temperature=self._temperature,
                        default_top_p=1.0,
                        override_do_sample=getattr(self, "_gen_do_sample_override", None),
                        override_temperature=getattr(self, "_gen_temperature_override", None),
                        override_top_p=getattr(self, "_gen_top_p_override", None),
                    )
                    batch_embeds: List[torch.Tensor] = []
                    sampling_params_list: List[Dict[str, Any]] = []
                    filtered_indices: List[int] = []
                    for idx in valid_indices:
                        e = embeds[idx]
                        if e is None:
                            continue
                        max_toks = int(gen_lens[idx])
                        if vllm_max_len > 0:
                            max_toks = min(max_toks, int(max_tokens_caps.get(idx, max_toks)))
                        if max_toks <= 0:
                            # No room to generate any tokens; skip.
                            skip_reason[idx] = (
                                f"no_generation_room:prompt_len={vllm_max_len - int(max_tokens_caps.get(idx, 0) or 0)}"
                                f" max_model_len={vllm_max_len}"
                                if vllm_max_len > 0
                                else "no_generation_room"
                            )
                            embeds[idx] = None
                            continue
                        params = {
                            "max_tokens": max_toks,
                            "temperature": resolved["temperature"],
                            "top_p": resolved["top_p"],
                        }
                        # NIAH (BOR-enabled) can emit <EOR> then immediately EOS. By default vLLM
                        # stops on EOS, which makes it *look* like generation stops at EOR.
                        # `ignore_eos=True` ensures we honor the requested token budget.
                        if self._mode == "niah_generate" and include_bor:
                            params.setdefault("ignore_eos", True)
                        filtered_indices.append(idx)
                        batch_embeds.append(e)
                        sampling_params_list.append(params)
                    valid_indices = filtered_indices
                    if valid_indices:
                        try:
                            outs = self._vllm_manager.generate_from_embeddings(
                                batch_embeds,
                                sampling_params=sampling_params_list,
                            )
                        except Exception as e:
                            # Do not crash the entire evaluation on a single overlong/invalid request.
                            err = f"vllm_generate_failed:{type(e).__name__}:{e}"
                            for idx in valid_indices:
                                if skip_reason[idx] is None:
                                    skip_reason[idx] = err
                            outs = []
                        for idx, out in zip(valid_indices, outs):
                            choice0 = out.outputs[0] if getattr(out, "outputs", None) else None
                            text = choice0.text if choice0 is not None else ""
                            outs_text[idx] = self._truncate_until(text, gkwargs[idx].get("until"))
                            try:
                                token_ids = getattr(choice0, "token_ids", None) if choice0 is not None else None
                                outs_token_ids[idx] = list(token_ids) if token_ids is not None else None
                            except Exception:
                                outs_token_ids[idx] = None
                results.extend(outs_text)

                # Write per-request generate debug rows for the compress-vLLM path.
                try:
                    sampling_params_by_idx: Dict[int, Dict[str, Any]] = {}
                    for j, idx in enumerate(valid_indices):
                        if 0 <= j < len(sampling_params_list):
                            sampling_params_by_idx[int(idx)] = dict(sampling_params_list[j])
                except Exception:
                    sampling_params_by_idx = {}

                gen_debug_rows: List[dict] = []
                for i in range(len(chunk)):
                    prompt_text = ""
                    ctx_preview: Optional[str] = None
                    if split_data is not None:
                        try:
                            prompt_text = split_data["query_list"][i] if i < len(split_data["query_list"]) else ""
                        except Exception:
                            prompt_text = ""
                        try:
                            ctx_text = split_data["context_list"][i] if i < len(split_data["context_list"]) else ""
                            ctx_preview = ctx_text[:5000] + ("\n...[truncated]..." if len(ctx_text) > 5000 else "")
                        except Exception:
                            ctx_preview = None
                    else:
                        prompt_text = prompts[i] if i < len(prompts) else ""

                    max_chars = int(getattr(self, "_generate_debug_max_prompt_chars", 0) or 0)
                    prompt_dbg = prompt_text
                    if max_chars > 0 and len(prompt_dbg) > max_chars:
                        prompt_dbg = prompt_dbg[:max_chars] + "\n...[truncated]..."

                    e = embeds[i] if i < len(embeds) else None
                    prompt_len_tokens = int(e.shape[0]) if e is not None else 0

                    compress_meta: Dict[str, Any] = {}
                    if embeds_meta:
                        for key in (
                            "n_spans",
                            "orig_n_spans",
                            "slots",
                            "flat_ctx_len",
                            "orig_flat_ctx_len",
                            "fixed_len",
                            "avail_for_memory",
                            "max_spans",
                            "span_len",
                            "decoder_budget",
                            "vllm_max_model_len",
                        ):
                            try:
                                val = embeds_meta.get(key)
                                if isinstance(val, list):
                                    if i < len(val):
                                        compress_meta[key] = val[i]
                                elif val is not None:
                                    compress_meta[key] = val
                            except Exception:
                                continue

                    gen_debug_rows.append(
                        {
                            "task": chunk_tasks[i] if i < len(chunk_tasks) else None,
                            "doc_id": _infer_doc_id(
                                chunk_docs[i] if i < len(chunk_docs) else None,
                                chunk_doc_ids[i] if i < len(chunk_doc_ids) else None,
                            ),
                            "request_index": int(start + i),
                            "mode": str(self._mode),
                            "backend": "vllm_compress_embeds",
                            "prompt": prompt_dbg,
                            "context_preview": ctx_preview,
                            "prompt_len_tokens": prompt_len_tokens,
                            "prompt_len_chars": len(prompt_dbg),
                            "generation_kwargs": gkwargs[i] if i < len(gkwargs) else None,
                            "sampling_params": sampling_params_by_idx.get(i),
                            "skip_reason": skip_reason[i] if i < len(skip_reason) else None,
                            "clip_note": clip_note[i] if i < len(clip_note) else None,
                            "compress_meta": compress_meta or None,
                            "response": outs_text[i] if i < len(outs_text) else "",
                        }
                    )
                self._append_generate_debug_rows(gen_debug_rows)

                # NIAH: dump debug cases (rank0), including special-token prompt/response.
                if (
                    split_data is not None
                    and "niah" in task0.lower()
                    and self._mode == "niah_generate"
                    and getattr(self, "_niah_debug_dir", "")
                    and int(getattr(self, "_niah_debug_max_cases", 0) or 0) != 0
                ):
                    debug_rows: List[dict] = []
                    for dbg_idx in range(len(chunk_docs)):
                        if not (0 <= dbg_idx < len(chunk_docs)):
                            continue
                        if chunk_docs[dbg_idx] is None:
                            continue
                        doc = chunk_docs[dbg_idx] or {}
                        ctx_text = split_data["context_list"][dbg_idx] if dbg_idx < len(split_data["context_list"]) else ""
                        query_text = split_data["query_list"][dbg_idx] if dbg_idx < len(split_data["query_list"]) else ""
                        question_text = (
                            split_data["question_list"][dbg_idx] if dbg_idx < len(split_data.get("question_list", [])) else ""
                        )
                        assistant_prefix = ""
                        if "assistant_prefix_list" in split_data and dbg_idx < len(split_data["assistant_prefix_list"]):
                            assistant_prefix = split_data["assistant_prefix_list"][dbg_idx] or ""

                        # Build an accurate special-token prefix snapshot matching the
                        # actual prompt_embeds layout (avoid re-splitting long contexts).
                        n_spans_dbg: Optional[int] = None
                        try:
                            if embeds_meta and "n_spans" in embeds_meta and dbg_idx < len(embeds_meta["n_spans"]):
                                n_spans_dbg = int(embeds_meta["n_spans"][dbg_idx])
                        except Exception:
                            n_spans_dbg = None
                        # Extra compression / budgeting diagnostics (helps verify span truncation).
                        meta_prompt_len: Optional[int] = None
                        meta_slots: Optional[int] = None
                        meta_flat_ctx_len: Optional[int] = None
                        meta_orig_n_spans: Optional[int] = None
                        meta_orig_flat_ctx_len: Optional[int] = None
                        meta_fixed_len: Optional[int] = None
                        meta_avail_for_memory: Optional[int] = None
                        meta_max_spans: Optional[int] = None
                        meta_gen_len: Optional[int] = None
                        meta_span_len: Optional[int] = None
                        meta_decoder_budget: Optional[int] = None
                        meta_vllm_max_model_len: Optional[int] = None
                        meta_decoder_memory_layout: Optional[str] = None
                        meta_num_comp: Optional[int] = None
                        try:
                            if embeds_meta:
                                def _get_list_value(name: str) -> Optional[Any]:
                                    val = embeds_meta.get(name)
                                    if isinstance(val, list) and dbg_idx < len(val):
                                        return val[dbg_idx]
                                    return None

                                meta_prompt_len = _coerce_int(_get_list_value("prefix_lens"), None)
                                meta_slots = _coerce_int(_get_list_value("slots"), None)
                                meta_flat_ctx_len = _coerce_int(_get_list_value("flat_ctx_len"), None)
                                meta_orig_n_spans = _coerce_int(_get_list_value("orig_n_spans"), None)
                                meta_orig_flat_ctx_len = _coerce_int(_get_list_value("orig_flat_ctx_len"), None)
                                meta_fixed_len = _coerce_int(_get_list_value("fixed_len"), None)
                                meta_avail_for_memory = _coerce_int(_get_list_value("avail_for_memory"), None)
                                meta_max_spans = _coerce_int(_get_list_value("max_spans"), None)
                                meta_gen_len = _coerce_int(_get_list_value("gen_lens"), None)

                                meta_span_len = _coerce_int(embeds_meta.get("span_len"), None)
                                meta_decoder_budget = _coerce_int(embeds_meta.get("decoder_budget"), None)
                                meta_vllm_max_model_len = _coerce_int(embeds_meta.get("vllm_max_model_len"), None)
                                dml = embeds_meta.get("decoder_memory_layout")
                                meta_decoder_memory_layout = str(dml) if dml is not None else None
                                meta_num_comp = _coerce_int(embeds_meta.get("num_comp"), None)
                        except Exception:
                            pass

                        try:
                            num_comp = int(getattr(self.model.args, "num_compression_tokens", 0))
                            placeholder_id = 0
                            prefix_tokens: List[int] = []

                            if bool(getattr(self, "_chat_use_template", False)) and str(
                                getattr(self, "_chat_template_version", "")
                            ).lower() == "v3":
                                memory_start = self._tokenizer.encode("<|im_start|>memory\n", bos=False, eos=False)
                                user_start = self._tokenizer.encode("<|im_start|>user\n", bos=False, eos=False)
                                assistant_start = self._tokenizer.encode(
                                    "<|im_start|>assistant\n", bos=False, eos=False
                                )
                                im_end = self._tokenizer.encode("<|im_end|>\n", bos=False, eos=False)

                                memory_body: List[int] = []
                                for _ in range(max(0, int(n_spans_dbg or 0))):
                                    memory_body.extend(
                                        [BEGIN_OF_MEMORY_INDEX]
                                        + ([placeholder_id] * num_comp)
                                        + [END_OF_MEMORY_INDEX]
                                    )
                                memory_tokens = memory_start + memory_body + im_end

                                user_tokens: List[int] = list(user_start)
                                if bool(getattr(self, "_add_boq_index", False)):
                                    user_tokens.append(BEGIN_OF_QUERY_INDEX)
                                user_tokens.extend(self._tokenizer.encode(query_text, bos=False, eos=False))
                                user_tokens.extend(im_end)

                                assistant_tokens: List[int] = list(assistant_start)
                                if assistant_prefix:
                                    assistant_tokens.extend(
                                        self._tokenizer.encode(str(assistant_prefix), bos=False, eos=False)
                                    )

                                prefix_tokens = memory_tokens + user_tokens + assistant_tokens
                            else:
                                # Non-chat prefix layout: [memory blocks]* + [BOQ] + query + (space+gen_prefix)
                                for _ in range(max(0, int(n_spans_dbg or 0))):
                                    prefix_tokens.extend(
                                        [BEGIN_OF_MEMORY_INDEX]
                                        + ([placeholder_id] * num_comp)
                                        + [END_OF_MEMORY_INDEX]
                                    )
                                if bool(getattr(self, "_add_boq_index", False)):
                                    prefix_tokens.append(BEGIN_OF_QUERY_INDEX)
                                prefix_tokens.extend(self._tokenizer.encode(query_text, bos=False, eos=False))
                                if assistant_prefix:
                                    # Match lm-eval's `target_delimiter: " "` between question and gen_prefix.
                                    ap = str(assistant_prefix)
                                    if ap and not ap[:1].isspace():
                                        ap = " " + ap
                                    prefix_tokens.extend(self._tokenizer.encode(ap, bos=False, eos=False))
                        except Exception:
                            prefix_tokens = []

                        if include_bor:
                            prefix_tokens.append(BEGIN_OF_RECONSTRUCTION_INDEX)
                        prompt_special = self.tok_decode_w_special_tokens(prefix_tokens)
                        max_chars = int(getattr(self, "_niah_debug_max_prompt_chars", 0) or 0)
                        if max_chars > 0 and len(prompt_special) > max_chars:
                            prompt_special = prompt_special[:max_chars] + "\n...[truncated]..."

                        raw_out = outs_text[dbg_idx] if dbg_idx < len(outs_text) else ""
                        resp_token_ids = outs_token_ids[dbg_idx] if dbg_idx < len(outs_token_ids) else None
                        resp_special = (
                            self.tok_decode_w_special_tokens(resp_token_ids) if resp_token_ids else raw_out
                        )

                        needle_type = _infer_niah_needle_type(doc.get("outputs", []))
                        extracted = _extract_niah_needles(raw_out, needle_type) if needle_type else []
                        # Record effective generation settings (after model_args overrides).
                        try:
                            effective_gen = resolve_generation_kwargs(
                                gkwargs[dbg_idx],
                                default_temperature=self._temperature,
                                default_top_p=1.0,
                                override_do_sample=getattr(self, "_gen_do_sample_override", None),
                                override_temperature=getattr(self, "_gen_temperature_override", None),
                                override_top_p=getattr(self, "_gen_top_p_override", None),
                            )
                        except Exception:
                            effective_gen = {}
                        debug_rows.append(
                            {
                                "task": task0,
                                "idx_in_batch": dbg_idx,
                                "doc_index": doc.get("index"),
                                "max_length": doc.get("max_length"),
                                "include_bor": bool(include_bor),
                                "niah_use_bor": bool(getattr(self, "_niah_use_bor", False)),
                                "max_bor": int(getattr(self, "_reconstruct_max_bor", 0) or 0),
                                # Decoder/memory budgeting diagnostics
                                "max_mem_span_len": int(getattr(self, "_max_mem_span_len", 0) or 0),
                                "max_mem_span_len_override": getattr(self, "_max_mem_span_len_override", None),
                                "span_len_used": meta_span_len,
                                "decoder_budget_used": meta_decoder_budget,
                                "vllm_max_model_len": meta_vllm_max_model_len,
                                "decoder_memory_layout": meta_decoder_memory_layout,
                                "num_compression_tokens": meta_num_comp,
                                "prompt_len": meta_prompt_len,
                                "n_spans": n_spans_dbg,
                                "orig_n_spans": meta_orig_n_spans,
                                "slots": meta_slots,
                                "flat_ctx_len": meta_flat_ctx_len,
                                "orig_flat_ctx_len": meta_orig_flat_ctx_len,
                                "fixed_len": meta_fixed_len,
                                "avail_for_memory": meta_avail_for_memory,
                                "max_spans": meta_max_spans,
                                "gen_len": meta_gen_len,
                                "skip_reason": skip_reason[dbg_idx] if dbg_idx < len(skip_reason) else None,
                                "clip_note": clip_note[dbg_idx] if dbg_idx < len(clip_note) else None,
                                "needle_type": needle_type,
                                "gold_outputs": doc.get("outputs"),
                                "question": question_text,
                                "query": query_text,
                                "gen_prefix": doc.get("gen_prefix"),
                                "task_generation_kwargs": gkwargs[dbg_idx],
                                "effective_generation_kwargs": {
                                    **(effective_gen or {}),
                                    "max_gen_toks": int(gen_lens[dbg_idx]),
                                }
                                if isinstance(effective_gen, dict) and dbg_idx < len(gen_lens)
                                else None,
                                "prompt_with_special_tokens": prompt_special,
                                "response": raw_out,
                                "response_with_special_tokens": resp_special,
                                "extracted": extracted,
                            }
                        )
                    self._append_niah_debug_rows(debug_rows)
                continue

            # Fallback: torch paths, process one by one within chunk
            # generate for greedy decoding
            for i, ((context_str, gen_kwargs), task_name) in enumerate(zip(chunk, chunk_tasks)):
                until = gen_kwargs.get("until", None)
                max_gen_len = _get_max_gen_tokens(gen_kwargs)
                resolved = resolve_generation_kwargs(
                    gen_kwargs,
                    default_temperature=self._temperature,
                    default_top_p=1.0,
                    override_do_sample=getattr(self, "_gen_do_sample_override", None),
                    override_temperature=getattr(self, "_gen_temperature_override", None),
                    override_top_p=getattr(self, "_gen_top_p_override", None),
                )
                temperature = resolved["temperature"]
                top_p = resolved["top_p"]

                if (
                    self._mode in {"compress_answer", "reconstruct_first", "niah_generate"}
                    and hasattr(self.model, "compression_embeddings")
                ):
                    include_bor = self._mode == "reconstruct_first"
                    if self._mode == "niah_generate":
                        include_bor = bool(getattr(self, "_niah_use_bor", False))
                    text = self._generate_compress_answer(
                        prompt=context_str,
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                        until=until,
                        include_bor=include_bor,
                    )
                    results.append(text)
                    prompt_dbg_src = context_str
                    backend = "torch_compress"
                else:
                    prompt_dbg_src = self._format_chat(context_str, add_generation_prompt=True)["decoder_prefix"]
                    backend = "torch_generate"
                    ctx_tokens, _ = self.tok_batch_encode([prompt_dbg_src])
                # breakpoint()
                    max_len = min(self.max_length, ctx_tokens.size(1) + max_gen_len)
                    # NOTE: `_model_generate` returns **only the newly generated tokens**
                    # (not the prompt+continuation concatenation). Do NOT slice by the
                    # prompt length here, or you will drop the whole output for long prompts.
                    gen_only_tokens = self._model_generate(
                        ctx_tokens.to(self.device),
                        max_len,
                        temperature=temperature,
                        top_p=top_p,
                    )[0].tolist()
                    # Strip trailing padding after EOS (the generator fills pad ids).
                    pad_id = self.pad_token_id if self.pad_token_id is not None else self.eot_token_id
                    if pad_id in gen_only_tokens:
                        gen_only_tokens = gen_only_tokens[: gen_only_tokens.index(pad_id)]
                    text = self.tok_decode(gen_only_tokens)

                    if until:
                        stops = until if isinstance(until, list) else [until]
                        cutoff = len(text)
                        for s in stops:
                            idx = text.find(s)
                            if idx != -1:
                                cutoff = min(cutoff, idx)
                        text = text[:cutoff]
                    results.append(text)

                max_chars = int(getattr(self, "_generate_debug_max_prompt_chars", 0) or 0)
                prompt_dbg = prompt_dbg_src
                if max_chars > 0 and len(prompt_dbg) > max_chars:
                    prompt_dbg = prompt_dbg[:max_chars] + "\n...[truncated]..."
                try:
                    prompt_len_tokens = len(self.tok_encode(prompt_dbg_src))
                except Exception:
                    prompt_len_tokens = 0
                debug_rows.append(
                    {
                        "task": task_name,
                        "doc_id": _infer_doc_id(chunk_docs[i] if i < len(chunk_docs) else None, chunk_doc_ids[i]),
                        "request_index": int(start + i),
                        "mode": str(self._mode),
                        "backend": backend,
                        "prompt": prompt_dbg,
                        "prompt_len_tokens": prompt_len_tokens,
                        "prompt_len_chars": len(prompt_dbg),
                        "generation_kwargs": gen_kwargs,
                        "sampling_params": {
                            "max_tokens": int(max_gen_len),
                            "temperature": float(temperature),
                            "top_p": float(top_p),
                        },
                        "skip_reason": None,
                        "response": text,
                    }
                )
            self._append_generate_debug_rows(debug_rows)
        return results

    def _generate_compress_answer(
        self,
        prompt: str,
        max_gen_len: int,
        temperature: float,
        top_p: float,
        until: Optional[List[str]],
        include_bor: bool = False,
    ) -> str:
        """
        Generation path that first compresses the prompt into memory slots, then
        decodes the answer conditioned on those slots. This mirrors the logic
        used in _loglikelihood_tokens_compress_answer but produces text instead
        of logprobs.
        """
        num_comp = getattr(self.model.args, "num_compression_tokens", 0)
        if num_comp <= 0:
            # fallback to vanilla decoding
            ctx_tokens, _ = self.tok_batch_encode([prompt])
            max_len = min(self.max_length, ctx_tokens.size(1) + max_gen_len)
            gen_only = self._model_generate(
                ctx_tokens.to(self.device),
                max_len,
                temperature=temperature,
                top_p=top_p,
            )[0].tolist()
            pad_id = self.pad_token_id if self.pad_token_id is not None else self.eot_token_id
            if pad_id in gen_only:
                gen_only = gen_only[: gen_only.index(pad_id)]
            text = self.tok_decode(gen_only)
            return self._truncate_until(text, until)

        max_mem_span_len = getattr(self.model.args, "max_mem_span_len", self.max_length)
        placeholder_id = 0

        prompt_tokens = self.tok_encode(prompt)
        static_count = 2 if self._add_boq_index and include_bor else 1 if include_bor or self._add_boq_index else 0
        static_count += 1 # for the eos
        

        
        # Split prompt into spans for encoder compression
        ctx_spans = [prompt_tokens[i : i + max_mem_span_len] for i in range(0, len(prompt_tokens), max_mem_span_len)]
        if not ctx_spans:
            ctx_spans = [[]]

        # Budget: (BOM + span + EOM ) * slots + prompt (+ BOR) + answer
        # Ensure we keep room for generation tokens
        bor_extra = 1 if include_bor else 0
        max_comp_tokens = max(0, self.max_length - (len(prompt_tokens) + 4 + bor_extra + max_gen_len))
        max_chunks = max_comp_tokens // (num_comp + 2) if num_comp > 0 else 0
        if max_chunks <= 0:
            max_chunks = 1
        ctx_spans = ctx_spans[-max_chunks:]
        
        # breakpoint()

        total_comp_slots = (num_comp + 2) * len(ctx_spans)
        # Build encoder packed tensors
        enc_tokens: List[int] = []
        enc_mem_mask: List[bool] = []
        for sp in ctx_spans:
            enc_tokens.extend(sp)
            enc_mem_mask.extend([False] * len(sp))
            enc_tokens.extend([placeholder_id] * num_comp)
            enc_mem_mask.extend([True] * num_comp)

        # Decoder prefix: BOM + slots + EOM + prompt (question); optionally BOR (for reconstruct_first)
        # dec_prefix = [BEGIN_OF_MEMORY_INDEX] + ([placeholder_id] * total_comp_slots) + [END_OF_MEMORY_INDEX] + prompt_tokens
        # comp_mask = [False] + ([True] * total_comp_slots) + [False] + ([False] * len(prompt_tokens))
        
        dec_prefix = []
        comp_mask = []
        for _ in range(len(ctx_spans)):
            dec_prefix.extend([BEGIN_OF_MEMORY_INDEX] + [placeholder_id] * num_comp + [END_OF_MEMORY_INDEX])
            comp_mask.extend([False] + [True] * num_comp + [False])
            
        if self._add_boq_index:
            dec_prefix.append(BEGIN_OF_QUERY_INDEX)
            comp_mask.append(False)
        
        dec_prefix.extend(prompt_tokens)
        comp_mask.extend([False] * len(prompt_tokens))
  
        # if self._fill_decoder_prefix_embeds:
        #     dec_prefix.extend(suffix_tokens)
        #     comp_mask.extend([False] * len(suffix_tokens))
        
        if include_bor:
            dec_prefix.append(BEGIN_OF_RECONSTRUCTION_INDEX)
            comp_mask.append(False)
            
        # breakpoint()

        # Cap generation so total length <= max_length
        max_new = max(0, min(max_gen_len, self.max_length - len(dec_prefix)))

        if len(enc_tokens) == 0:
            enc_tokens = [placeholder_id]
            enc_mem_mask = [False]

        enc_tokens_t = torch.tensor(enc_tokens, device=self.device, dtype=torch.long)
        enc_mem_mask_t = torch.tensor(enc_mem_mask, device=self.device, dtype=torch.bool)
        enc_cu = torch.tensor([0, len(enc_tokens)], device=self.device, dtype=torch.int32)
        enc_ctx = {
            "cu_seqlens_q": enc_cu,
            "cu_seqlens_k": enc_cu,
            "max_seqlen_q": len(enc_tokens),
            "max_seqlen_k": len(enc_tokens),
            "positions": torch.arange(len(enc_tokens), device=self.device, dtype=torch.int32),
            "encoder_mem_mask": enc_mem_mask_t,
        }

        dec_tokens = torch.tensor(dec_prefix, device=self.device, dtype=torch.long)
        comp_mask_t = torch.tensor(comp_mask, device=self.device, dtype=torch.bool)
        generated: List[int] = []

        for _ in range(max_new):
            dec_cu = torch.tensor([0, dec_tokens.numel()], device=self.device, dtype=torch.int32)
            dec_ctx = {
                "cu_seqlens_q": dec_cu,
                "cu_seqlens_k": dec_cu,
                "max_seqlen_q": dec_tokens.numel(),
                "max_seqlen_k": dec_tokens.numel(),
                "positions": torch.arange(dec_tokens.numel(), device=self.device, dtype=torch.int32),
                "compression_token_mask": comp_mask_t,
            }
            with torch.autocast(device_type="cuda", dtype=self._dtype):
                logits = self.model(
                    encoder_tokens=enc_tokens_t,
                    encoder_context=enc_ctx,
                    decoder_tokens=dec_tokens,
                    decoder_context=dec_ctx,
                    last_hidden_only=False,
                )
            last_logits = logits[-1]
            if temperature and temperature > 0:
                probs = torch.softmax(last_logits / temperature, dim=-1)
                probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
                probs_sum = torch.cumsum(probs_sort, dim=-1)
                probs_sort[probs_sum - probs_sort > top_p] = 0.0
                probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
                next_token = torch.multinomial(probs_sort, num_samples=1)
                next_token = torch.gather(probs_idx, -1, next_token).reshape(-1)[0]
            else:
                next_token = torch.argmax(last_logits)

            next_id = int(next_token.item())
            if next_id in {self.eos_token_id, END_OF_RECONSTRUCTION_INDEX}:
                break
            generated.append(next_id)
            dec_tokens = torch.cat([dec_tokens, torch.tensor([next_id], device=self.device, dtype=torch.long)], dim=0)
            comp_mask_t = torch.cat([comp_mask_t, torch.tensor([False], device=self.device, dtype=torch.bool)], dim=0)

        text = self.tok_decode(generated)
        return self._truncate_until(text, until)

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
        """
        mode not = "compress_answer" or "reconstruct_first", but 
        Iterative vLLM decoding with on-the-fly compression of generated tokens.
        Compress only generated tokens; prompt tokens stay uncompressed.
        """
        if self._vllm_manager is None:
            raise ValueError("vLLM manager not initialized")
        num_comp = getattr(self.model.args, "num_compression_tokens", 0)
        if num_comp <= 0:
            return self._generate_with_vllm_decoder(prompt, max_gen_len, temperature, top_p, until)

        placeholder_id = 0
        compress_threshold = self._compress_threshold
        max_span_len = getattr(self.model.args, "max_mem_span_len", None)
        compress_chunk = max(1, self._compress_chunk)
        max_cycles = self._max_cycles
        marker_id_seqs: List[List[int]] = []
        for marker in self._compress_start_tokens:
            ids = self.tok_encode(marker)
            if ids:
                marker_id_seqs.append(ids)
        # breakpoint()
        prompt_tokens = self.tok_encode(prompt, add_thinking_tokens=self._add_thinking_tokens)
        gen_tokens: List[int] = []
        comp_blocks: List[torch.Tensor] = []
        cycles = 0
        done = False
        out_text_parts: List[str] = []
        total_raw_compressed = 0

        stop_ids = self.stop_ids

        while not done and cycles < max_cycles:
            tokens: List[int] = []
            comp_mask: List[bool] = []
            tokens.extend(prompt_tokens)
            comp_mask.extend([False] * len(prompt_tokens))
            for comp in comp_blocks:
                tokens.extend([BEGIN_OF_MEMORY_INDEX] + [placeholder_id] * num_comp + [END_OF_MEMORY_INDEX])
                comp_mask.extend([False] + [True] * num_comp + [False])
            tokens.extend(gen_tokens)
            comp_mask.extend([False] * len(gen_tokens))
            
            # tokens_decoded = self.tok_decode(tokens)
            tokens_decoded_prompt = self.tok_decode_w_special_tokens(prompt_tokens)
            tokens_decoded_gen = self.tok_decode_w_special_tokens(gen_tokens)

            # if len(tokens) == 0:
                # tokens = [self.pad_token_id]
                # comp_mask = [False]
            

            tok_tensor = torch.tensor(tokens, device=self.device, dtype=torch.long)
            embeds = self.model.tok_embeddings(tok_tensor).to(self._dtype)
            if comp_blocks:
                comp_concat = torch.cat(comp_blocks, dim=0)
                comp_mask_t = torch.tensor(comp_mask, device=self.device, dtype=torch.bool)
                need = comp_mask_t.sum().item()
                if comp_concat.shape[0] != need:
                    # shape mismatch guard: trim or pad with zeros
                    if comp_concat.shape[0] > need:
                        comp_concat = comp_concat[:need]
                    else:
                        pad = torch.zeros(need - comp_concat.shape[0], comp_concat.shape[1], device=comp_concat.device, dtype=comp_concat.dtype)
                        comp_concat = torch.cat([comp_concat, pad], dim=0)
                embeds[comp_mask_t] = comp_concat.to(embeds.dtype)
                
            print("prompt_tokens: ", len(prompt_tokens))
            print("embeds: ", embeds.shape)
            outs = self._vllm_manager.generate_from_embeddings(
                [embeds],
                sampling_params={
                    # for max_tokens, we need to subtract the prompt tokens and the compressed tokens
                    "max_tokens": min(self._max_seq_length, max_gen_len) - (len(prompt_tokens) + len(comp_blocks) * (num_comp + 2)) ,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stop_token_ids": stop_ids,
                },
            )
            out = outs[0] if outs else None
            print("generated tokens: ", len(out.outputs[0].token_ids))
            if out and out.outputs and out.outputs[0].token_ids is not None:
                new_tokens = out.outputs[0].token_ids
                gen_tokens.extend(new_tokens)
                if out.outputs[0].finish_reason == "stop" or any(t in stop_ids for t in new_tokens):
                    done = True
                # detect marker to allow early compression
                marker_pos = None
                marker_len = 0
                if marker_id_seqs:
                    for seq in marker_id_seqs:
                        if len(seq) == 0 or len(seq) > len(gen_tokens):
                            continue
                        # simple scan for first occurrence
                        for i in range(0, len(gen_tokens) - len(seq) + 1):
                            if gen_tokens[i : i + len(seq)] == seq:
                                marker_pos = i
                                marker_len = len(seq)
                                break
                        if marker_pos is not None:
                            break
                # if markers provided: only compress suffix AFTER the marker
                # breakpoint()
                prefix_len = len(prompt_tokens) + len(comp_blocks) * (num_comp + 2)
                if marker_pos is not None:
                    tail_start = marker_pos + marker_len
                    tail_len = len(gen_tokens) - tail_start
                    allow_compress = tail_len >= compress_threshold or len(gen_tokens) + prefix_len >= self._max_seq_length
                    start_idx = tail_start
                else:
                    allow_compress = len(gen_tokens) + prefix_len >= compress_threshold or len(gen_tokens) + prefix_len >= self._max_seq_length
                    start_idx = 0
                if not done and allow_compress:
                    chunk = gen_tokens[start_idx : start_idx + compress_chunk]
                    print("compress chunk: ", len(chunk))
                    # breakpoint()
                    gen_tokens = gen_tokens[:start_idx] + gen_tokens[start_idx + len(chunk) :]
                    print("after compress gen_tokens: ", len(gen_tokens))
                    # breakpoint()
                    # out_text_parts.append(self.tok_decode(chunk))
                    out_text_parts.append(self.tok_decode_w_special_tokens(chunk))
                    total_raw_compressed += len(chunk)
                    # split chunk into spans of max_span_len, compress each
                    for j in range(0, len(chunk), max_span_len):
                        sub = chunk[j : j + max_span_len]
                        if not sub:
                            continue
                        sub_t = torch.tensor(sub, device=self.device, dtype=torch.long)
                        comp_vec = self._compress_plain_sequence(sub_t)
                        comp_blocks.append(comp_vec)
                    cycles += 1
                # breakpoint()
            else:
                done = True
        # final_text = "<ours_concat_text>".join(out_text_parts) + self.tok_decode(gen_tokens)
        final_text = "<ours_concat_text>".join(out_text_parts) + self.tok_decode_w_special_tokens(gen_tokens)
        final_text = self._truncate_until(final_text, until)
        dbg = {
            "prompt_len": len(prompt_tokens),
            "final_tokens_len": len(prompt_tokens) + sum(b.shape[0] for b in comp_blocks) + len(gen_tokens),
            "total_raw_compressed": total_raw_compressed,
            "compress_steps": cycles,
            "max_gen_len": max_gen_len,
            "generated_text": final_text,
        }
        self._last_generate_debug.append(dbg)
        # Keep the debug log bounded to avoid unbounded growth.
        if len(self._last_generate_debug) > 200:
            self._last_generate_debug = self._last_generate_debug[-200:]
        print(f"[vllm_compress_debug] {dbg}")
        return final_text

    def _generate_with_vllm_decoder(
        self,
        prompt: str,
        max_gen_len: int,
        temperature: float,
        top_p: float,
        until: Optional[List[str]],
    ) -> str:
        """Use vLLM for plain decoder generation (non-compression models only)."""
        if self._vllm_manager is None:
            raise RuntimeError("vLLM manager is not initialized for decoder mode.")
        sampling_params = {
            "max_tokens": max_gen_len,
            "temperature": temperature,
            "top_p": top_p,
        }
        outputs = self._vllm_manager.engine_wrapper.generate([prompt], sampling_params)
        if not outputs or not outputs[0].outputs:
            return ""
        text = outputs[0].outputs[0].text
        return self._truncate_until(text, until)

    def _generate_compress_answer_vllm(
        self,
        prompt: str,
        max_gen_len: int,
        temperature: float,
        top_p: float,
        until: Optional[List[str]],
        include_bor: bool = False,
    ) -> str:
        """
        Compress prompt into memory slots, fill decoder prompt_embeds, and use vLLM to decode answer.
        """
        if self._vllm_manager is None:
            raise RuntimeError("vLLM manager is not initialized for compress_answer mode.")

        prompt_embeds = self._build_compress_prompt_embeds(prompt, include_bor=include_bor)
        if prompt_embeds is None:
            return ""

        sampling_params = {
            "max_tokens": max_gen_len,
            "temperature": temperature,
            "top_p": top_p,
        }
        outputs = self._vllm_manager.generate_from_embeddings([prompt_embeds], sampling_params=sampling_params)
        if not outputs or not outputs[0].outputs:
            return ""
        text = outputs[0].outputs[0].text
        return self._truncate_until(text, until)

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
        """
        Batch build prompt embeddings for compress_answer/reconstruct_first with optional BOR.
        """
        if len(prompts) != len(gen_lens):
            raise ValueError(f"prompts and gen_lens must have same length; got {len(prompts)} vs {len(gen_lens)}")

        num_comp = int(getattr(self.model.args, "num_compression_tokens", 0))
        embeds: List[Optional[torch.Tensor]] = [None] * len(prompts)
        if decoder_memory_layout not in {"single", "per_span"}:
            raise ValueError(f"Unsupported decoder_memory_layout: {decoder_memory_layout}")
        chat_enabled = bool(getattr(self, "_chat_use_template", False))
        use_chat = bool(chat_enabled and context_list is not None and query_list is not None)
        use_split_nonchat = bool((not chat_enabled) and context_list is not None and query_list is not None)
        if (context_list is None) != (query_list is None) and (context_list is not None or query_list is not None):
            raise ValueError("context_list/query_list must both be set when using split contexts")
        if use_chat or use_split_nonchat:
            if context_list is None or query_list is None:
                raise ValueError("split mode requires both context_list and query_list")
            if len(context_list) != len(prompts) or len(query_list) != len(prompts):
                raise ValueError(
                    f"context_list/query_list must match prompts length; got {len(context_list)}/{len(query_list)} vs {len(prompts)}"
                )
            if assistant_prefix_list is not None and len(assistant_prefix_list) != len(prompts):
                raise ValueError(
                    f"assistant_prefix_list must match prompts length; got {len(assistant_prefix_list)} vs {len(prompts)}"
                )
        decoder_budget = int(self.decoder_budget)
        # vLLM enforces a hard max prompt length (`max_model_len`). When we build
        # prompt_embeds for compression-heavy modes (generate_until uses vLLM
        # prompt_embeds; reconstruct_first scoring also requires vLLM), we must
        # clamp the effective decoder budget to `vllm_max_model_len` even if the
        # vLLM engine hasn't been initialized yet.
        vllm_max_int = _coerce_int(getattr(self, "_vllm_max_model_len", None), None) or 0
        if vllm_max_int > 0:
            if self._mode in {"compress_answer", "reconstruct_first", "vllm_decoding_with_compress", "niah_generate"}:
                decoder_budget = min(decoder_budget, vllm_max_int)
            elif getattr(self, "_vllm_manager", None) is not None:
                decoder_budget = min(decoder_budget, vllm_max_int)

        # --------------------------
        # Fast path: no compression
        # --------------------------
        if num_comp <= 0:
            if use_chat:
                meta_n_spans: List[int] = []
                meta_flat_lens: List[int] = []
                meta_slots: List[int] = []
                meta_comp_masks: List[List[bool]] = []
                meta_prefix_lens: List[int] = []
                for i in range(len(prompts)):
                    ret = self._format_chat(user_text=query_list[i], contexts=context_list[i])
                    prefix_tokens = ret.get("decoder_prefix_tokens") or []
                    if not prefix_tokens:
                        embeds[i] = None
                        meta_n_spans.append(0)
                        meta_flat_lens.append(0)
                        meta_slots.append(0)
                        meta_comp_masks.append([])
                        meta_prefix_lens.append(0)
                        continue
                    tok_tensor = torch.tensor(prefix_tokens, device=self.device, dtype=torch.long)
                    embeds[i] = self.model.tok_embeddings(tok_tensor).to(dtype=self._dtype)
                    meta_n_spans.append(int(ret.get("n_spans", 0)))
                    meta_flat_lens.append(int(ret.get("total_encoder_tokens", 0)))
                    meta_slots.append(int(ret.get("total_comp_slots", 0)))
                    meta_comp_masks.append(list(ret.get("comp_mask", [])))
                    meta_prefix_lens.append(len(prefix_tokens))
                meta = {
                    "n_spans": meta_n_spans,
                    "flat_ctx_len": meta_flat_lens,
                    "slots": meta_slots,
                    "comp_mask_list": meta_comp_masks,
                    "prefix_lens": meta_prefix_lens,
                }
                return (embeds, meta) if return_meta else embeds

            meta_n_spans = [0] * len(prompts)
            meta_flat_lens = [0] * len(prompts)
            meta_slots = [0] * len(prompts)
            meta_comp_masks: List[List[bool]] = [[] for _ in prompts]
            meta_prefix_lens: List[int] = [0] * len(prompts)
            for i, p in enumerate(prompts):
                p_tokens = prompt_tokens_override[i] if prompt_tokens_override is not None else self.tok_encode(p)
                dec_tokens: List[int] = []
                if decoder_include_prompt_tokens:
                    dec_tokens.extend(p_tokens)
                if self._add_boq_index and not not_add_boq_index:
                    dec_tokens.append(BEGIN_OF_QUERY_INDEX)
                if include_bor:
                    dec_tokens.append(BEGIN_OF_RECONSTRUCTION_INDEX)
                if not dec_tokens:
                    continue
                tok_tensor = torch.tensor(dec_tokens, device=self.device, dtype=torch.long)
                embeds[i] = self.model.tok_embeddings(tok_tensor).to(dtype=self._dtype)
                meta_comp_masks[i] = [False] * len(dec_tokens)
                meta_prefix_lens[i] = len(dec_tokens)
            meta = {
                "n_spans": meta_n_spans,
                "flat_ctx_len": meta_flat_lens,
                "slots": meta_slots,
                "comp_mask_list": meta_comp_masks,
                "prefix_lens": meta_prefix_lens,
            }
            return (embeds, meta) if return_meta else embeds

        # --------------------------
        # Compression-aware path
        # --------------------------
        # Prefer the eval-time span length knob we computed in __init__ (which
        # respects `--model_args max_mem_span_len=...`) over any checkpoint-default
        # value. This is important for NIAH/RULER: if `max_mem_span_len` is smaller
        # than intended, the number of memory spans (and thus decoder prompt length)
        # can explode and exceed `vllm_max_model_len`.
        max_mem_span_len = int(getattr(self, "_max_mem_span_len", 0) or 0)
        if max_mem_span_len <= 0:
            max_mem_span_len = int(getattr(self.model.args, "max_mem_span_len", decoder_budget))
        model_max_len = int(getattr(self.model.args, "max_seq_len", self._max_seq_length))
        if model_max_len <= num_comp:
            raise ValueError(
                f"Invalid config: model_max_len={model_max_len} <= num_compression_tokens={num_comp}. "
                "Encoder span cannot fit placeholders."
            )

        # Each encoder micro-batch is: span_tokens + num_comp placeholders.
        # Ensure positions never exceed model max len.
        # Clamp span_len so `span_len + num_comp` never exceeds encoder max len.
        span_len = min(int(max_mem_span_len), int(model_max_len) - int(num_comp))
        if span_len <= 0:
            span_len = 1
        placeholder_id = 0

        add_boq = bool(self._add_boq_index and not not_add_boq_index)
        boq_extra = 1 if add_boq else 0
        bor_extra = 1 if include_bor else 0

        if use_chat:
            selected_spans_list: List[int] = []
            selected_flat_lens: List[int] = []
            total_comp_slots_list: List[int] = []
            comp_offsets: List[int] = [0]
            max_spans_list: List[int] = []
            comp_mask_list: List[List[bool]] = []
            prefix_lens: List[int] = []
            orig_spans_list: List[int] = []
            orig_flat_lens: List[int] = []
            fixed_lens_list: List[int] = []
            avail_for_memory_list: List[int] = []
            force_skip: List[bool] = [False] * len(prompts)

            enc_tokens_mb: List[torch.Tensor] = []
            enc_ctx_mb: List[Dict[str, torch.Tensor]] = []

            memory_start = self._tokenizer.encode("<|im_start|>memory\n", bos=False, eos=False)
            user_start = self._tokenizer.encode("<|im_start|>user\n", bos=False, eos=False)
            assistant_start = self._tokenizer.encode("<|im_start|>assistant\n", bos=False, eos=False)
            im_end = self._tokenizer.encode("<|im_end|>\n", bos=False, eos=False)
            span_cost = num_comp + 2
            # Reserve some room for generation so we don't end up with a prompt that
            # leaves 0 tokens for the model to output (vLLM would then skip).
            reserve_cap = 256 if self._mode == "niah_generate" else 1024

            for i, contexts in enumerate(context_list):
                spans = self._split_contexts_to_spans(contexts, span_len)
                orig_spans_list.append(len(spans))
                orig_flat_lens.append(sum(len(sp) for sp in spans))
                query_tokens = self._tokenizer.encode(query_list[i], bos=False, eos=False)
                assistant_prefix = ""
                if assistant_prefix_list is not None:
                    raw_prefix = assistant_prefix_list[i]
                    if raw_prefix is not None:
                        assistant_prefix = str(raw_prefix)
                assistant_prefix_tokens = (
                    self._tokenizer.encode(assistant_prefix, bos=False, eos=False) if assistant_prefix else []
                )
                fixed_len = (
                    len(memory_start)
                    + len(im_end)
                    + len(user_start)
                    + boq_extra
                    + len(query_tokens)
                    + len(im_end)
                    + len(assistant_start)
                    + len(assistant_prefix_tokens)
                    + bor_extra
                )
                fixed_lens_list.append(int(fixed_len))
                reserve_gen = min(int(gen_lens[i]), reserve_cap) if int(gen_lens[i]) > 0 else 0
                avail_for_memory_list.append(int(decoder_budget) - int(fixed_len) - int(reserve_gen))
                max_spans = (decoder_budget - fixed_len - int(reserve_gen)) // max(1, span_cost)
                # If the decoder budget cannot fit any memory spans, prefer dropping
                # spans rather than forcing 1 span and crashing vLLM with an overlong prompt.
                if max_spans < 0:
                    max_spans = 0
                if max_spans == 0:
                    spans = []
                elif max_spans < len(spans):
                    spans = spans[-max_spans:]
                max_spans_list.append(max_spans)
                n_spans = len(spans)
                selected_spans_list.append(n_spans)
                selected_flat_lens.append(sum(len(sp) for sp in spans))
                slots = num_comp * n_spans
                total_comp_slots_list.append(slots)
                comp_offsets.append(comp_offsets[-1] + slots)

                for sp in spans:
                    enc_seq = torch.tensor(list(sp) + ([placeholder_id] * num_comp), device=self.device, dtype=torch.long)
                    clen = int(enc_seq.numel())
                    mem_mask = torch.zeros(clen, device=self.device, dtype=torch.bool)
                    mem_mask[-num_comp:] = True
                    cu = torch.tensor([0, clen], device=self.device, dtype=torch.int32)
                    enc_tokens_mb.append(enc_seq)
                    enc_ctx_mb.append(
                        {
                            "cu_seqlens_q": cu,
                            "cu_seqlens_k": cu,
                            "max_seqlen_q": clen,
                            "max_seqlen_k": clen,
                            "positions": torch.arange(clen, device=self.device, dtype=torch.int32),
                            "encoder_mem_mask": mem_mask,
                        }
                    )

            expected_total_slots = int(comp_offsets[-1])
            d_model = int(getattr(self.model.args, "d_model", 0))
            if expected_total_slots <= 0:
                compression_vectors = torch.empty((0, d_model), device=self.device, dtype=self._dtype)
            else:
                with torch.autocast(device_type="cuda", dtype=self._dtype):
                    compression_vectors = self._compress_multi_batches_with_progress(
                        enc_tokens_mb, enc_ctx_mb, desc=f"compress[{self._mode}]"
                    )
                if int(compression_vectors.shape[0]) != expected_total_slots:
                    raise RuntimeError(
                        f"compress_multi_batches returned {int(compression_vectors.shape[0])} vectors, "
                        f"expected {expected_total_slots} (= num_comp * total_spans)."
                    )
                if compression_vectors.dtype != self._dtype:
                    compression_vectors = compression_vectors.to(dtype=self._dtype)

            for i in range(len(prompts)):
                if force_skip[i]:
                    embeds[i] = None
                    comp_mask_list.append([])
                    prefix_lens.append(0)
                    continue
                assistant_prefix = None
                if assistant_prefix_list is not None:
                    assistant_prefix = assistant_prefix_list[i]
                ret = self._format_chat(
                    user_text=query_list[i],
                    assistant_text=assistant_prefix,
                    contexts=context_list[i],
                    max_spans=max_spans_list[i],
                )
                prefix_tokens = ret.get("decoder_prefix_tokens") or []
                comp_mask = list(ret["comp_mask"])
                if include_bor:
                    prefix_tokens = list(prefix_tokens) + [BEGIN_OF_RECONSTRUCTION_INDEX]
                    comp_mask.append(False)
                total_comp_slots = int(ret.get("total_comp_slots", 0))
                if total_comp_slots != total_comp_slots_list[i]:
                    raise RuntimeError(
                        f"Internal error: chat template slots ({total_comp_slots}) != encoder slots "
                        f"({total_comp_slots_list[i]}) for sample {i}."
                    )

                prefix_t = torch.tensor(prefix_tokens, device=self.device, dtype=torch.long)
                comp_mask_t = torch.tensor(comp_mask, device=self.device, dtype=torch.bool)
                if int(comp_mask_t.numel()) != int(prefix_t.numel()):
                    raise RuntimeError(
                        f"Internal error: comp_mask length ({int(comp_mask_t.numel())}) != prefix length "
                        f"({int(prefix_t.numel())}) for sample {i}."
                    )

                pe = self.model.tok_embeddings(prefix_t).to(dtype=self._dtype)
                if total_comp_slots > 0:
                    mask_slots = int(comp_mask_t.sum().item())
                    if mask_slots != total_comp_slots:
                        raise RuntimeError(
                            f"Internal error: comp_mask slots ({mask_slots}) != expected slots ({total_comp_slots}) "
                            f"for sample {i}."
                        )
                    v0, v1 = int(comp_offsets[i]), int(comp_offsets[i + 1])
                    vec = compression_vectors[v0:v1]
                    if int(vec.shape[0]) != total_comp_slots:
                        raise RuntimeError(
                            f"Internal error: compression slice rows ({int(vec.shape[0])}) != slots ({total_comp_slots}) "
                            f"for sample {i}."
                        )
                    pe[comp_mask_t] = vec.to(dtype=self._dtype)
                embeds[i] = pe
                comp_mask_list.append(list(comp_mask))
                prefix_lens.append(len(prefix_tokens))

            meta = {
                "n_spans": selected_spans_list,
                "flat_ctx_len": selected_flat_lens,
                "orig_n_spans": orig_spans_list,
                "orig_flat_ctx_len": orig_flat_lens,
                "slots": total_comp_slots_list,
                "comp_mask_list": comp_mask_list,
                "prefix_lens": prefix_lens,
                "fixed_len": fixed_lens_list,
                "avail_for_memory": avail_for_memory_list,
                "max_spans": max_spans_list,
                "force_skip": force_skip,
                # Global config snapshot (useful for debugging budget/truncation decisions).
                "span_len": int(span_len),
                "decoder_budget": int(decoder_budget),
                "vllm_max_model_len": int(getattr(self, "_vllm_max_model_len", 0) or 0),
                "decoder_memory_layout": str(decoder_memory_layout),
                "num_comp": int(num_comp),
                "gen_lens": [int(x) for x in gen_lens],
            }
            return (embeds, meta) if return_meta else embeds

        if use_split_nonchat:
            selected_spans_list: List[int] = []
            selected_flat_lens: List[int] = []
            orig_spans_list: List[int] = []
            orig_flat_lens: List[int] = []
            total_comp_slots_list: List[int] = []
            comp_offsets: List[int] = [0]
            comp_mask_list: List[List[bool]] = []
            prefix_lens: List[int] = []
            force_skip: List[bool] = [False] * len(prompts)
            fixed_lens_list: List[int] = []
            avail_for_memory_list: List[int] = []
            max_spans_list: List[int] = []

            enc_tokens_mb: List[torch.Tensor] = []
            enc_ctx_mb: List[Dict[str, torch.Tensor]] = []

            query_tokens_list: List[List[int]] = []
            assistant_prefix_tokens_list: List[List[int]] = []

            for i, contexts in enumerate(context_list):
                spans = self._split_contexts_to_spans(contexts, span_len)
                # Record pre-truncation stats (useful for debugging overly aggressive truncation).
                orig_n_spans = len(spans)
                orig_flat_len = sum(len(sp) for sp in spans)
                query_tokens = self._tokenizer.encode(query_list[i], bos=False, eos=False)
                query_tokens_list.append(query_tokens)

                assistant_prefix = ""
                if assistant_prefix_list is not None:
                    raw_prefix = assistant_prefix_list[i]
                    if raw_prefix is not None:
                        assistant_prefix = str(raw_prefix)

                assistant_prefix_tokens: List[int] = []
                if assistant_prefix:
                    # Match lm-eval `target_delimiter: " "` between question and gen_prefix.
                    ap_text = assistant_prefix
                    if ap_text and not ap_text[:1].isspace():
                        ap_text = " " + ap_text
                    assistant_prefix_tokens = self._tokenizer.encode(ap_text, bos=False, eos=False)
                assistant_prefix_tokens_list.append(assistant_prefix_tokens)

                fixed_len = boq_extra + len(query_tokens) + len(assistant_prefix_tokens) + bor_extra
                fixed_lens_list.append(int(fixed_len))
                if fixed_len > decoder_budget:
                    # Cannot fit even without memory blocks; skip.
                    force_skip[i] = True
                    spans = []
                    orig_n_spans = 0
                    orig_flat_len = 0

                # Determine how many memory spans can fit in the decoder prompt budget.
                reserve_cap = 256 if self._mode == "niah_generate" else 1024
                reserve_gen = min(int(gen_lens[i]), reserve_cap) if int(gen_lens[i]) > 0 else 0
                avail_for_memory = int(decoder_budget) - int(fixed_len) - int(reserve_gen)
                avail_for_memory_list.append(int(avail_for_memory))
                if decoder_memory_layout == "single":
                    # single layout: [BOM] + slots + [EOM] costs 2 + (num_comp * n_spans)
                    span_cost = num_comp
                    max_spans = (avail_for_memory - 2) // max(1, span_cost)
                else:
                    # per-span layout: each span costs (BOM + slots + EOM) = num_comp + 2
                    span_cost = num_comp + 2
                    max_spans = avail_for_memory // max(1, span_cost)

                if max_spans < 0:
                    max_spans = 0
                if max_spans == 0:
                    spans = []
                elif max_spans < len(spans):
                    # Prefer tail spans (RULER/NIAH needles may appear late).
                    spans = spans[-max_spans:]
                max_spans_list.append(int(max_spans))
                orig_spans_list.append(int(orig_n_spans))
                orig_flat_lens.append(int(orig_flat_len))

                n_spans = len(spans)
                selected_spans_list.append(n_spans)
                selected_flat_lens.append(sum(len(sp) for sp in spans))
                slots = num_comp * n_spans
                total_comp_slots_list.append(slots)
                comp_offsets.append(comp_offsets[-1] + slots)

                for sp in spans:
                    enc_seq = torch.tensor(
                        list(sp) + ([placeholder_id] * num_comp), device=self.device, dtype=torch.long
                    )
                    clen = int(enc_seq.numel())
                    mem_mask = torch.zeros(clen, device=self.device, dtype=torch.bool)
                    mem_mask[-num_comp:] = True
                    cu = torch.tensor([0, clen], device=self.device, dtype=torch.int32)
                    enc_tokens_mb.append(enc_seq)
                    enc_ctx_mb.append(
                        {
                            "cu_seqlens_q": cu,
                            "cu_seqlens_k": cu,
                            "max_seqlen_q": clen,
                            "max_seqlen_k": clen,
                            "positions": torch.arange(clen, device=self.device, dtype=torch.int32),
                            "encoder_mem_mask": mem_mask,
                        }
                    )

            expected_total_slots = int(comp_offsets[-1])
            d_model = int(getattr(self.model.args, "d_model", 0))
            if expected_total_slots <= 0:
                compression_vectors = torch.empty((0, d_model), device=self.device, dtype=self._dtype)
            else:
                with torch.autocast(device_type="cuda", dtype=self._dtype):
                    compression_vectors = self._compress_multi_batches_with_progress(
                        enc_tokens_mb, enc_ctx_mb, desc=f"compress[{self._mode}]"
                    )
                if int(compression_vectors.shape[0]) != expected_total_slots:
                    raise RuntimeError(
                        f"compress_multi_batches returned {int(compression_vectors.shape[0])} vectors, "
                        f"expected {expected_total_slots} (= num_comp * total_spans)."
                    )
                if compression_vectors.dtype != self._dtype:
                    compression_vectors = compression_vectors.to(dtype=self._dtype)

            for i in range(len(prompts)):
                if force_skip[i]:
                    embeds[i] = None
                    comp_mask_list.append([])
                    prefix_lens.append(0)
                    continue

                slots = int(total_comp_slots_list[i])
                n_spans = int(selected_spans_list[i])

                prefix: List[int] = []
                comp_mask: List[bool] = []
                if decoder_memory_layout == "single":
                    if slots > 0:
                        prefix = [BEGIN_OF_MEMORY_INDEX] + ([placeholder_id] * slots) + [END_OF_MEMORY_INDEX]
                        comp_mask = [False] + ([True] * slots) + [False]
                else:
                    for _ in range(n_spans):
                        prefix.extend([BEGIN_OF_MEMORY_INDEX] + ([placeholder_id] * num_comp) + [END_OF_MEMORY_INDEX])
                        comp_mask.extend([False] + ([True] * num_comp) + [False])

                if add_boq:
                    prefix.append(BEGIN_OF_QUERY_INDEX)
                    comp_mask.append(False)

                qt = query_tokens_list[i] if i < len(query_tokens_list) else []
                prefix.extend(qt)
                comp_mask.extend([False] * len(qt))

                apt = assistant_prefix_tokens_list[i] if i < len(assistant_prefix_tokens_list) else []
                prefix.extend(apt)
                comp_mask.extend([False] * len(apt))

                if include_bor:
                    prefix.append(BEGIN_OF_RECONSTRUCTION_INDEX)
                    comp_mask.append(False)

                if not prefix:
                    embeds[i] = None
                    comp_mask_list.append([])
                    prefix_lens.append(0)
                    continue

                prefix_t = torch.tensor(prefix, device=self.device, dtype=torch.long)
                comp_mask_t = torch.tensor(comp_mask, device=self.device, dtype=torch.bool)
                pe = self.model.tok_embeddings(prefix_t).to(dtype=self._dtype)

                if slots > 0:
                    mask_slots = int(comp_mask_t.sum().item())
                    if mask_slots != slots:
                        raise RuntimeError(
                            f"Internal error: comp_mask slots ({mask_slots}) != expected slots ({slots}) for sample {i}."
                        )
                    v0, v1 = int(comp_offsets[i]), int(comp_offsets[i + 1])
                    vec = compression_vectors[v0:v1]
                    if int(vec.shape[0]) != slots:
                        raise RuntimeError(
                            f"Internal error: compression slice rows ({int(vec.shape[0])}) != slots ({slots}) for sample {i}."
                        )
                    pe[comp_mask_t] = vec.to(dtype=self._dtype)

                embeds[i] = pe
                comp_mask_list.append(list(comp_mask))
                prefix_lens.append(len(prefix))

            meta = {
                "n_spans": selected_spans_list,
                "flat_ctx_len": selected_flat_lens,
                "orig_n_spans": orig_spans_list,
                "orig_flat_ctx_len": orig_flat_lens,
                "slots": total_comp_slots_list,
                "comp_mask_list": comp_mask_list,
                "prefix_lens": prefix_lens,
                "fixed_len": fixed_lens_list,
                "avail_for_memory": avail_for_memory_list,
                "max_spans": max_spans_list,
                # Global config snapshot (helps debug span budgeting decisions).
                "span_len": int(span_len),
                "decoder_budget": int(decoder_budget),
                "vllm_max_model_len": int(getattr(self, "_vllm_max_model_len", 0) or 0),
                "decoder_memory_layout": str(decoder_memory_layout),
                "num_comp": int(num_comp),
                "gen_lens": [int(x) for x in gen_lens],
            }
            return (embeds, meta) if return_meta else embeds

        prompt_tokens_list: List[List[int]] = []
        selected_spans_list: List[int] = []
        selected_flat_lens: List[int] = []
        total_comp_slots_list: List[int] = []
        comp_offsets: List[int] = [0]
        comp_mask_list: List[List[bool]] = []
        prefix_lens: List[int] = []

        enc_tokens_mb: List[torch.Tensor] = []
        enc_ctx_mb: List[Dict[str, torch.Tensor]] = []
        verbose_compress = bool(getattr(self, "_verbose_compress", False))

        for i, (p, glen) in enumerate(zip(prompts, gen_lens)):
            p_tokens = prompt_tokens_override[i] if prompt_tokens_override is not None else self.tok_encode(p)
            prompt_tokens_list.append(p_tokens)

            # Split into raw spans.
            ctx_spans = [p_tokens[j : j + span_len] for j in range(0, len(p_tokens), span_len)]
            if not ctx_spans:
                ctx_spans = [[]]

            # Select tail spans to fit within decoder budget (without truncating generation budget).
            prompt_in_dec_len = len(p_tokens) if decoder_include_prompt_tokens else 0
            total_static = prompt_in_dec_len + boq_extra + bor_extra + (2 if decoder_memory_layout == "single" else 0)
            
            span_cost = num_comp if decoder_memory_layout == "single" else (num_comp + 2)
            reserve_cap = 256 if self._mode == "niah_generate" else 1024
            reserve_gen = min(int(glen), reserve_cap) if int(glen) > 0 else 0
            max_comp_tokens = max(0, int(decoder_budget) - int(total_static) - int(reserve_gen))
            
            max_chunks = max_comp_tokens // max(1, int(span_cost))
            if max_chunks <= 0:
                max_chunks = 0
            if max_chunks == 0:
                ctx_spans = []
            elif max_chunks < len(ctx_spans):
                ctx_spans = ctx_spans[-max_chunks:]
                if verbose_compress:
                    print("need chunk spans,", max_chunks, "total spans ", len(ctx_spans))
            else:
                if verbose_compress:
                    print(
                        "no need chunk spans,",
                        max_chunks,
                        "total spans ",
                        len(ctx_spans),
                        "available spaces",
                        max_comp_tokens - len(ctx_spans) * span_cost,
                    )

            if verbose_compress:
                print(
                    "max_length,",
                    decoder_budget,
                    "total_static,",
                    total_static,
                    "glen,",
                    glen,
                    "max_comp_tokens,",
                    max_comp_tokens,
                    "max_chunks,",
                    max_chunks,
                )


            n_spans = len(ctx_spans)
            selected_spans_list.append(n_spans)
            selected_flat_lens.append(sum(len(sp) for sp in ctx_spans))
            # breakpoint()
            slots = num_comp * n_spans
            total_comp_slots_list.append(slots)
            comp_offsets.append(comp_offsets[-1] + slots)

            # Build encoder micro-batches for each span. We intentionally do NOT pack multiple spans
            # into a single encoder sequence (no mixing across spans).
            for sp in ctx_spans:
                enc_seq = torch.tensor(list(sp) + ([placeholder_id] * num_comp), device=self.device, dtype=torch.long)
                clen = int(enc_seq.numel())
                mem_mask = torch.zeros(clen, device=self.device, dtype=torch.bool)
                mem_mask[-num_comp:] = True
                cu = torch.tensor([0, clen], device=self.device, dtype=torch.int32)
                enc_tokens_mb.append(enc_seq)
                enc_ctx_mb.append(
                    {
                        "cu_seqlens_q": cu,
                        "cu_seqlens_k": cu,
                        "max_seqlen_q": clen,
                        "max_seqlen_k": clen,
                        "positions": torch.arange(clen, device=self.device, dtype=torch.int32),
                        "encoder_mem_mask": mem_mask,
                    }
                )

        expected_total_slots = int(comp_offsets[-1])
        d_model = int(getattr(self.model.args, "d_model", 0))
        if expected_total_slots <= 0:
            compression_vectors = torch.empty((0, d_model), device=self.device, dtype=self._dtype)
        else:
            with torch.autocast(device_type="cuda", dtype=self._dtype):
                compression_vectors = self._compress_multi_batches_with_progress(
                    enc_tokens_mb, enc_ctx_mb, desc=f"compress[{self._mode}]"
                )
            if int(compression_vectors.shape[0]) != expected_total_slots:
                raise RuntimeError(
                    f"compress_multi_batches returned {int(compression_vectors.shape[0])} vectors, "
                    f"expected {expected_total_slots} (= num_comp * total_spans)."
                )
            if compression_vectors.dtype != self._dtype:
                compression_vectors = compression_vectors.to(dtype=self._dtype)

        # Build decoder prompt embeds and fill memory slot positions with compression vectors.
        for i, p_tokens in enumerate(prompt_tokens_list):
            slots = int(total_comp_slots_list[i])
            n_spans = int(selected_spans_list[i])

            prefix: List[int] = []
            comp_mask: List[bool] = []
            if decoder_memory_layout == "single":
                if slots > 0:
                    prefix = [BEGIN_OF_MEMORY_INDEX] + ([placeholder_id] * slots) + [END_OF_MEMORY_INDEX]
                    comp_mask = [False] + ([True] * slots) + [False]
            else:
                # per-span: each span has BOM + slots + EOM
                for _ in range(n_spans):
                    prefix.extend([BEGIN_OF_MEMORY_INDEX] + ([placeholder_id] * num_comp) + [END_OF_MEMORY_INDEX])
                    comp_mask.extend([False] + ([True] * num_comp) + [False])

            if decoder_include_prompt_tokens:
                prefix.extend(p_tokens)
                comp_mask.extend([False] * len(p_tokens))

            if add_boq:
                prefix.append(BEGIN_OF_QUERY_INDEX)
                comp_mask.append(False)

            if include_bor:
                prefix.append(BEGIN_OF_RECONSTRUCTION_INDEX)
                comp_mask.append(False)

            if not prefix:
                embeds[i] = None
                continue

            prefix_t = torch.tensor(prefix, device=self.device, dtype=torch.long)
            comp_mask_t = torch.tensor(comp_mask, device=self.device, dtype=torch.bool)
            
            pe = self.model.tok_embeddings(prefix_t).to(dtype=self._dtype)

            if slots > 0:
                mask_slots = int(comp_mask_t.sum().item())
                if mask_slots != slots:
                    raise RuntimeError(
                        f"Internal error: comp_mask slots ({mask_slots}) != expected slots ({slots}) "
                        f"for sample {i}."
                    )
                v0, v1 = int(comp_offsets[i]), int(comp_offsets[i + 1])
                vec = compression_vectors[v0:v1]
                if int(vec.shape[0]) != slots:
                    raise RuntimeError(
                        f"Internal error: compression slice rows ({int(vec.shape[0])}) != slots ({slots}) "
                        f"for sample {i}."
                    )
                pe[comp_mask_t] = vec.to(dtype=self._dtype)

            embeds[i] = pe
            comp_mask_list.append(list(comp_mask))
            prefix_lens.append(len(prefix))

        meta = {
            "n_spans": selected_spans_list,
            "flat_ctx_len": selected_flat_lens,
            "slots": total_comp_slots_list,
            "comp_mask_list": comp_mask_list,
            "prefix_lens": prefix_lens,
        }
        return (embeds, meta) if return_meta else embeds



        # ---- compression-aware scoring ----
    
    @torch.no_grad()
    def _loglikelihood_tokens_compress_answer(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: Optional[int] = None,
    ) -> List[Tuple[float, bool]]:
        """
        Compress the context (question) into memory slots, then score only the continuation
        (answer) tokens conditioned on those slots, without reconstructing the question.
        """
        bs = override_bs or self.batch_size
        try:
            bs = int(bs)
        except Exception:
            bs = 1

        num_comp = int(getattr(self.model.args, "num_compression_tokens", 0))
        placeholder_id = 0
        max_mem_span_len = int(getattr(self.model.args, "max_mem_span_len", self.max_length))
        include_bor = False
        decoder_budget = int(self.decoder_budget)

 
        
        
            
        
        def _split_ctx_for_compression(ctx_tokens: List[int], cont_len: int) -> Tuple[List[int], List[int]]:
            if not ctx_tokens:
                return [], []
            max_len = int(self.max_length)
            span_cost = num_comp + 2  # decoder cost per span: BOM + slots + EOM
            saving = max_mem_span_len - span_cost
            if saving <= 0:
                # No compression benefit; let the span selector drop old spans.
                return ctx_tokens, []

            raw_len = len(ctx_tokens)
            total_spans = math.ceil(raw_len / max_mem_span_len)
            # If even compressing all spans can't fit, keep no suffix and let the builder drop spans.
            if total_spans * span_cost + cont_len > max_len:
                return ctx_tokens, []

            # Need k spans compressed so that:
            #   raw_len + cont_len - k*(max_mem_span_len - (num_comp+2)) <= max_len
            need = raw_len + cont_len - max_len
            if need <= 0:
                k = raw_len // max_mem_span_len  # compress all full spans; keep remainder as suffix
            else:
                k = int(math.ceil(need / float(saving)))
            k = max(0, min(k, raw_len // max_mem_span_len))
            raw_comp_len = k * max_mem_span_len
            return ctx_tokens[:raw_comp_len], ctx_tokens[raw_comp_len:]
        
        

        res: List[Tuple[float, bool]] = []
        iterator = range(0, len(requests), bs)
        pbar = tqdm(iterator, disable=disable_tqdm or self.rank != 0, desc="native loglikelihood (compress)")
        for batch_start in pbar:
            chunk = requests[batch_start : batch_start + bs]
            # Build prefix prompt_embeds (memory blocks + optional BOQ) via the shared helper,
            # then score only the continuation tokens.
            prompt_tokens_override: List[List[int]] = [ctx for (_, ctx, _) in chunk]
            ctx_tokens_full_list: List[List[int]] = list(prompt_tokens_override)
            # In compress_answer, the decoder prefix is synthetic (memory blocks / chat scaffold),
            # so the continuation must be tokenized from the raw continuation string, not from
            # the (context+continuation) split performed in TemplateLM._encode_pair().
            cont_tokens_list = [
                self._tokenizer.encode(pair[1], bos=False, eos=False) if pair and len(pair) > 1 else []
                for (pair, _, _) in chunk
            ]
            prefix_tokens = self._get_likelihood_prefix_tokens("compress_answer")
            if prefix_tokens:
                cont_tokens_list = [prefix_tokens + list(cont) for cont in cont_tokens_list]
            suffix_tokens_list: List[List[int]] = [[] for _ in range(len(chunk))]

            
            # fill decoder prefix embeds, only compress old context and keep new context uncompressed
            # TODO temp close this feature
            if self._fill_decoder_prefix_embeds and not getattr(self, "_chat_use_template", False):
                prompt_tokens_list, suffix_tokens_list_t = zip(
                    *[_split_ctx_for_compression(p, len(c)) for p, c in zip(prompt_tokens_override, cont_tokens_list)]
                )
                prompt_tokens_override = list(prompt_tokens_list)
                suffix_tokens_list = list(suffix_tokens_list_t)
            elif getattr(self, "_chat_use_template", False):
                suffix_tokens_list = [[] for _ in range(len(chunk))]


            # prompt_tokens, suffix_tokens = compute_needed_comp_slots(prompt_tokens_override[0])
            # Fast path: empty continuations
            chunk_results: List[Tuple[float, bool]] = [(float("-inf"), False)] * len(chunk)
            nonempty_idxs = [i for i, c in enumerate(cont_tokens_list) if len(c) > 0]
            for i in range(len(chunk)):
                if len(cont_tokens_list[i]) == 0:
                    chunk_results[i] = (0.0, True)

            def _append_compress_debug_rows(
                prefix_embeds_list: Optional[List[Optional[torch.Tensor]]] = None,
                meta_n_spans: Optional[List[int]] = None,
                score_stats: Optional[Dict[int, Dict[str, Any]]] = None,
                skip_reasons: Optional[List[Optional[str]]] = None,
            ) -> None:
                if not self._save_loglikelihood_debug or self._distributed_args.rank != 0:
                    return
                rows: List[dict] = []
                for i in range(len(chunk_results)):
                    cont_len = len(cont_tokens_list[i])
                    logprob, greedy = chunk_results[i]
                    loss = None
                    ppl = None
                    score_info = score_stats.get(i) if score_stats else None
                    if score_info and score_info.get("tokens", 0) > 0:
                        loss = score_info.get("loss")
                        ppl = score_info.get("ppl")
                    elif cont_len > 0 and math.isfinite(logprob):
                        loss = -float(logprob) / float(cont_len)
                        try:
                            ppl = math.exp(loss)
                        except OverflowError:
                            ppl = float("inf")
                    row = {
                        "request_index": batch_start + i,
                        "mode": "compress_answer",
                        "cont_len": cont_len,
                        "logprob": logprob,
                        "greedy": greedy,
                        "ppl": ppl,
                    }
                    try:
                        raw_cont = chunk[i][0][1] if chunk[i] and chunk[i][0] and len(chunk[i][0]) > 1 else ""
                    except Exception:
                        raw_cont = ""
                    if raw_cont:
                        row["cont_str_len"] = int(len(raw_cont))
                        row["cont_str_preview"] = raw_cont[:200]
                    row["cont_tokens_len"] = int(cont_len)
                    if cont_len > 0:
                        row["cont_tokens_preview"] = list(cont_tokens_list[i][:20])
                        if cont_len <= 50:
                            row["cont_tokens"] = list(cont_tokens_list[i])
                        try:
                            decoded = self.tok_decode_w_special_tokens(cont_tokens_list[i])
                            if decoded:
                                row["cont_decoded_preview"] = decoded[:200]
                        except Exception:
                            pass
                    if loss is not None:
                        row["loss"] = loss
                    if score_info is not None:
                        row["score_tokens"] = int(score_info.get("tokens", 0))
                        if "windows" in score_info:
                            row["windows"] = int(score_info.get("windows") or 0)
                        if "rolled" in score_info:
                            row["rolled"] = bool(score_info.get("rolled"))
                    if skip_reasons is not None and skip_reasons[i]:
                        row["skip_reason"] = skip_reasons[i]
                    if meta_n_spans is not None:
                        n_spans = int(meta_n_spans[i]) if i < len(meta_n_spans) else 0
                        row["n_spans"] = n_spans
                        row["total_comp_slots"] = n_spans * int(num_comp)
                    if suffix_tokens_list is not None:
                        row["suffix_len"] = len(suffix_tokens_list[i])
                    if prefix_embeds_list is not None:
                        pe = prefix_embeds_list[i]
                        if pe is not None:
                            prefix_len = int(pe.shape[0]) + (len(suffix_tokens_list[i]) if suffix_tokens_list is not None else 0)
                            row["prefix_len"] = prefix_len
                    rows.append(row)
                self._append_loglikelihood_debug_rows(rows)

            if not nonempty_idxs:
                _append_compress_debug_rows()
                res.extend(chunk_results)
                continue

            dummy_prompts = [""] * len(chunk)
            gen_lens = [len(c) + (len(suffix_tokens_list[i]) if suffix_tokens_list is not None else 0) for i, c in enumerate(cont_tokens_list)]
            if not getattr(self, "_chat_use_template", False):
                # Prefer structured (context, query) splitting when available so the query
                # (e.g. "Question: ...\nAnswer:") is not compressed away. This is critical
                # for long-context multiple-choice tasks like InfiniteBench longbook_choice_eng.
                split_context_list: Optional[List[str]] = None
                split_query_list: Optional[List[str]] = None
                try:
                    doc_and_context = self._get_doc_and_context(
                        ctx_tokens_list=ctx_tokens_full_list, batch_start=batch_start
                    )
                    split_context_list = doc_and_context.get("context_list")
                    split_query_list = doc_and_context.get("query_list")
                except Exception:
                    split_context_list = None
                    split_query_list = None

                if (
                    split_context_list is not None
                    and split_query_list is not None
                    and len(split_context_list) == len(chunk)
                    and len(split_query_list) == len(chunk)
                ):
                    key_to_group: Dict[Tuple[str, str], int] = {}
                    group_contexts: List[str] = []
                    group_queries: List[str] = []
                    group_gen_lens: List[int] = []
                    group_indices: List[List[int]] = []
                    for i in range(len(chunk)):
                        key = (split_context_list[i], split_query_list[i])
                        gidx = key_to_group.get(key)
                        if gidx is None:
                            gidx = len(group_contexts)
                            key_to_group[key] = gidx
                            group_contexts.append(split_context_list[i])
                            group_queries.append(split_query_list[i])
                            group_gen_lens.append(int(gen_lens[i]))
                            group_indices.append([i])
                        else:
                            if int(gen_lens[i]) > group_gen_lens[gidx]:
                                group_gen_lens[gidx] = int(gen_lens[i])
                            group_indices[gidx].append(i)

                    group_dummy_prompts = [""] * len(group_contexts)
                    group_prefix_embeds, meta = self._build_compress_prompt_embeds_batch(
                        group_dummy_prompts,
                        group_gen_lens,
                        include_bor=False,
                        decoder_include_prompt_tokens=False,
                        decoder_memory_layout="per_span",
                        return_meta=True,
                        not_add_boq_index=False,
                        context_list=group_contexts,
                        query_list=group_queries,
                    )
                    meta_group_n_spans = (
                        meta.get("n_spans", [1] * len(group_contexts)) if meta else [1] * len(group_contexts)
                    )
                    meta_group_comp_masks = meta.get("comp_mask_list") if meta else None
                    prefix_embeds_list = [None] * len(chunk)
                    meta_n_spans = [1] * len(chunk)
                    prefix_comp_masks_list = [None] * len(chunk)
                    for gidx, idxs in enumerate(group_indices):
                        for i in idxs:
                            prefix_embeds_list[i] = group_prefix_embeds[gidx]
                            meta_n_spans[i] = int(meta_group_n_spans[gidx]) if gidx < len(meta_group_n_spans) else 1
                            if meta_group_comp_masks is not None and gidx < len(meta_group_comp_masks):
                                prefix_comp_masks_list[i] = list(meta_group_comp_masks[gidx])
                else:
                    key_to_group: Dict[Tuple[int, ...], int] = {}
                    group_prompt_tokens: List[List[int]] = []
                    group_gen_lens: List[int] = []
                    group_indices: List[List[int]] = []
                    for i in range(len(chunk)):
                        key = tuple(prompt_tokens_override[i])
                        gidx = key_to_group.get(key)
                        if gidx is None:
                            gidx = len(group_prompt_tokens)
                            key_to_group[key] = gidx
                            group_prompt_tokens.append(prompt_tokens_override[i])
                            group_gen_lens.append(int(gen_lens[i]))
                            group_indices.append([i])
                        else:
                            if int(gen_lens[i]) > group_gen_lens[gidx]:
                                group_gen_lens[gidx] = int(gen_lens[i])
                            group_indices[gidx].append(i)

                    group_dummy_prompts = [""] * len(group_prompt_tokens)
                    group_prefix_embeds, meta = self._build_compress_prompt_embeds_batch(
                        group_dummy_prompts,
                        group_gen_lens,
                        include_bor=False,
                        decoder_include_prompt_tokens=False,
                        decoder_memory_layout="per_span",
                        prompt_tokens_override=group_prompt_tokens,
                        return_meta=True,
                        # for do not add boq index for decoder prefix
                        not_add_boq_index=True,
                    )
                    meta_group_n_spans = (
                        meta.get("n_spans", [1] * len(group_prompt_tokens))
                        if meta
                        else [1] * len(group_prompt_tokens)
                    )
                    meta_group_comp_masks = meta.get("comp_mask_list") if meta else None
                    prefix_embeds_list = [None] * len(chunk)
                    meta_n_spans = [1] * len(chunk)
                    prefix_comp_masks_list = [None] * len(chunk)
                    for gidx, idxs in enumerate(group_indices):
                        for i in idxs:
                            prefix_embeds_list[i] = group_prefix_embeds[gidx]
                            meta_n_spans[i] = (
                                int(meta_group_n_spans[gidx]) if gidx < len(meta_group_n_spans) else 1
                            )
                            if meta_group_comp_masks is not None and gidx < len(meta_group_comp_masks):
                                prefix_comp_masks_list[i] = list(meta_group_comp_masks[gidx])
            else:
                doc_and_context = self._get_doc_and_context(ctx_tokens_list=ctx_tokens_full_list, batch_start=batch_start)
                context_list, question_list, query_list = (
                    doc_and_context["context_list"],
                    doc_and_context["question_list"],
                    doc_and_context["query_list"],
                )

                key_to_group: Dict[Tuple[str, str], int] = {}
                group_contexts: List[str] = []
                group_queries: List[str] = []
                group_gen_lens: List[int] = []
                group_indices: List[List[int]] = []
                for i in range(len(chunk)):
                    key = (context_list[i], query_list[i])
                    gidx = key_to_group.get(key)
                    if gidx is None:
                        gidx = len(group_contexts)
                        key_to_group[key] = gidx
                        group_contexts.append(context_list[i])
                        group_queries.append(query_list[i])
                        group_gen_lens.append(int(gen_lens[i]))
                        group_indices.append([i])
                    else:
                        if int(gen_lens[i]) > group_gen_lens[gidx]:
                            group_gen_lens[gidx] = int(gen_lens[i])
                        group_indices[gidx].append(i)

                group_dummy_prompts = [""] * len(group_contexts)
                group_prefix_embeds, meta = self._build_compress_prompt_embeds_batch(
                    group_dummy_prompts,
                    group_gen_lens,
                    include_bor=False,
                    decoder_include_prompt_tokens=False,
                    decoder_memory_layout="per_span",
                    return_meta=True,
                    not_add_boq_index=False,
                    context_list=group_contexts,
                    query_list=group_queries,
                )
                meta_group_n_spans = meta.get("n_spans", [1] * len(group_contexts)) if meta else [1] * len(group_contexts)
                meta_group_comp_masks = meta.get("comp_mask_list") if meta else None
                prefix_embeds_list = [None] * len(chunk)
                meta_n_spans = [1] * len(chunk)
                prefix_comp_masks_list = [None] * len(chunk)
                for gidx, idxs in enumerate(group_indices):
                    for i in idxs:
                        prefix_embeds_list[i] = group_prefix_embeds[gidx]
                        meta_n_spans[i] = int(meta_group_n_spans[gidx]) if gidx < len(meta_group_n_spans) else 1
                        if meta_group_comp_masks is not None and gidx < len(meta_group_comp_masks):
                            prefix_comp_masks_list[i] = list(meta_group_comp_masks[gidx])
            # breakpoint()
            
            # Suffix tokens (uncompressed tail of the prompt) can be ragged across the batch.
            # Embed per-sample to avoid creating a rectangular tensor.
            d_model = int(getattr(self.model.args, "d_model", 0))
            suffix_embeds_list: List[torch.Tensor] = []
            for suffix in suffix_tokens_list:
                if suffix:
                    st = torch.tensor(list(suffix), device=self.device, dtype=torch.long)
                    with torch.autocast(device_type="cuda", dtype=self._dtype):
                        se = self.model.tok_embeddings(st).to(dtype=self._dtype)
                    suffix_embeds_list.append(se)
                else:
                    suffix_embeds_list.append(torch.empty((0, d_model), device=self.device, dtype=self._dtype))

            # meta_n_spans is now pre-mapped per sample for grouped compression
            seq_embeds: List[torch.Tensor] = []
            seq_targets: List[torch.Tensor] = []
            seq_loss_masks: List[torch.Tensor] = []
            seq_comp_masks: List[torch.Tensor] = []
            prefix_lens: List[int] = []
            cont_lens: List[int] = []
            cont_targets: List[torch.Tensor] = []
            valid_map: List[int] = []
            skip_reasons: List[Optional[str]] = [None] * len(chunk)
            score_stats_by_idx: Dict[int, Dict[str, Any]] = {}

            for i in nonempty_idxs:
                cont = cont_tokens_list[i]
                cont_len = len(cont)
                if cont_len <= 0:
                    continue
                pe0 = prefix_embeds_list[i]
                if pe0 is None:
                    skip_reasons[i] = "no_prefix"
                    continue

                # Prefix consists of memory blocks + raw suffix tokens (kept uncompressed).
                suffix_e = suffix_embeds_list[i]
                pe = torch.cat([pe0, suffix_e], dim=0) if suffix_e.numel() else pe0
                prefix_len = int(pe.shape[0])

                if prefix_len + cont_len > decoder_budget:
                    skip_reasons[i] = "length_overflow"
                    continue

                # compression_token_mask: True for placeholder slots in memory blocks, False elsewhere.
                if prefix_comp_masks_list is not None and prefix_comp_masks_list[i] is not None:
                    comp_mask_prefix = prefix_comp_masks_list[i]
                else:
                    n_spans = int(meta_n_spans[i]) if i < len(meta_n_spans) else 1
                    comp_mask_prefix = ([False] + ([True] * num_comp) + [False]) * n_spans
                suffix_len = len(suffix_tokens_list[i]) if suffix_tokens_list[i] else 0
                base_mask = comp_mask_prefix + ([False] * suffix_len)
                base_mask_t = torch.tensor(base_mask, device=self.device, dtype=torch.bool)
                if int(base_mask_t.numel()) != prefix_len:
                    skip_reasons[i] = "comp_mask_mismatch"
                    continue

                cont_t = torch.tensor(cont, device=self.device, dtype=torch.long)
                with torch.autocast(device_type="cuda", dtype=self._dtype):
                    cont_e = self.model.tok_embeddings(cont_t).to(dtype=self._dtype)

                full_embeds = torch.cat([pe, cont_e], dim=0)
                total_len = int(full_embeds.shape[0])

                # Decoder token ids (for shifting/targets). Memory placeholders use a valid vocab id.
                prefix_mask = prefix_comp_masks_list[i] if prefix_comp_masks_list is not None else None
                if prefix_mask is not None:
                    prefix_ids = [placeholder_id] * len(prefix_mask)
                else:
                    n_spans = int(meta_n_spans[i]) if i < len(meta_n_spans) else 1
                    prefix_ids = ([BEGIN_OF_MEMORY_INDEX] + ([placeholder_id] * num_comp) + [END_OF_MEMORY_INDEX]) * n_spans
                suffix_ids = list(suffix_tokens_list[i]) if suffix_tokens_list[i] else []
                token_ids = prefix_ids + suffix_ids + list(cont)
                if len(token_ids) != total_len:
                    skip_reasons[i] = "token_len_mismatch"
                    continue

                targets_ids = token_ids[1:] + [int(self.eos_token_id)]
                targets_t = torch.tensor(targets_ids, device=self.device, dtype=torch.long)

                # Score only continuation: first continuation token is predicted at position prefix_len-1.
                score_start = prefix_len - 1
                score_end = score_start + cont_len
                if score_start < 0 or score_end > total_len:
                    skip_reasons[i] = "score_range_invalid"
                    continue
                loss_mask = torch.zeros(total_len, device=self.device, dtype=torch.bool)
                loss_mask[score_start:score_end] = True

                # compression_token_mask: True for placeholder slots in memory blocks, False elsewhere.
                if prefix_mask is not None:
                    comp_mask = list(prefix_mask) + ([False] * (total_len - len(prefix_mask)))
                else:
                    comp_mask_prefix = ([False] + ([True] * num_comp) + [False]) * n_spans
                    comp_mask = comp_mask_prefix + ([False] * (total_len - len(comp_mask_prefix)))
                comp_mask_t = torch.tensor(comp_mask, device=self.device, dtype=torch.bool)

                seq_embeds.append(full_embeds)
                seq_targets.append(targets_t)
                seq_loss_masks.append(loss_mask)
                seq_comp_masks.append(comp_mask_t)
                prefix_lens.append(prefix_len)
                cont_lens.append(cont_len)
                cont_targets.append(cont_t)
                valid_map.append(i)

            if not seq_embeds:
                _append_compress_debug_rows(prefix_embeds_list, meta_n_spans, score_stats_by_idx, skip_reasons)
                res.extend(chunk_results)
                continue

            rows_per_chunk = int(getattr(self.model.args, "cross_entropy_chunk", 8)) * 16
            rows_per_chunk = max(8, min(int(rows_per_chunk), 512))

            score_out = self._forward_score_continuations(
                seq_embeds=seq_embeds,
                cont_targets=cont_targets,
                prefix_lens=prefix_lens,
                comp_mask_list=seq_comp_masks,
                rows_per_chunk=rows_per_chunk,
            )
            per_sample = score_out.get("per_sample") or []
            for j, orig_idx in enumerate(valid_map):
                if j >= len(per_sample):
                    break
                ps = per_sample[j] or {}
                ll = float(ps.get("ll", float("-inf")))
                greedy = bool(ps.get("greedy", False))
                chunk_results[orig_idx] = (ll, greedy)
                score_stats_by_idx[orig_idx] = ps

            _append_compress_debug_rows(prefix_embeds_list, meta_n_spans, score_stats_by_idx, skip_reasons)
            res.extend(chunk_results)
        # Safety: if we somehow produced fewer responses than requests, pad with -inf.
        if len(res) < len(requests):
            missing = len(requests) - len(res)
            res.extend([(float("-inf"), False)] * missing)
        return res


        # Decide how much of the context to compress (prefix) vs keep raw (suffix),
        # without ever truncating continuation tokens.
    def _get_doc_and_context(self, ctx_tokens_list: List[List[int]], *, batch_start: int = 0) -> Dict[str, List[str]]:
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
        """
        Reconstruct-first loglikelihood for compression models.

        Differences vs `compress_answer`:
        - We **generate reconstruction once per unique context** (not once per option).
        - Then we score each continuation option conditioned on (memory + reconstruction [+ optional query suffix]).

        If `add_query_before_likelihood=True`:
        - We try to split the original prompt into a long document block and a short query/choices suffix
          using `<text> ... </text>` markers (LongBench-style).
        - Only the document block is compressed/reconstructed; the query suffix is appended before scoring.
        """
        self._ensure_vllm_manager(caller="reconstruct_first loglikelihood")
        if self._vllm_manager is None:
            raise RuntimeError("reconstruct_first loglikelihood requires vLLM but initialization failed.")

        num_comp = int(getattr(self.model.args, "num_compression_tokens", 0))
        max_mem_span_len = int(getattr(self.model.args, "max_mem_span_len", self.max_length))
        add_bor = bool(self._reconstruct_add_bor)
        max_bor = int(self._reconstruct_max_bor)
        add_query = bool(getattr(self, "_add_query_before_likelihood", False))
        chat_enabled = bool(getattr(self, "_chat_use_template", False))
        decoder_budget = int(self.decoder_budget)
        stop_id_set = set(self.stop_ids)
        newline_ids = self._tokenizer.encode("\n", bos=False, eos=False)
        newline_id = newline_ids[0] if newline_ids else None

        def _postprocess_recon(tokens: List[int], max_len: int) -> Tuple[List[int], dict]:
            out: List[int] = []
            bor_count = 0
            eor_count = 0
            stop_reason: Optional[str] = None
            eos_id = int(self.eos_token_id)
            eot_id = int(self.eot_token_id)
            for t in tokens:
                t = int(t)
                # Treat EOS/EOT as a hard stop (and do not include it).
                if t in (eos_id, eot_id):
                    stop_reason = "eos"
                    break
                if t == BEGIN_OF_RECONSTRUCTION_INDEX:
                    bor_count += 1
                    if max_bor > 0 and bor_count > max_bor:
                        stop_reason = "bor_limit"
                        break
                if t == END_OF_RECONSTRUCTION_INDEX:
                    eor_count += 1
                    out.append(t)
                    # If `max_bor` is disabled (<=0), stop at the first EOR.
                    if max_bor <= 0:
                        stop_reason = "eor"
                        break
                    if eor_count >= max_bor:
                        stop_reason = "max_eor"
                        break
                    continue
                out.append(t)
                if len(out) >= max_len:
                    stop_reason = "max_recon"
                    break
            if stop_reason is None:
                stop_reason = "max_recon"
            # Strip trailing EOS/EOT just in case.
            while out and out[-1] in (eos_id, eot_id):
                out.pop()
            # Strip trailing stop_ids.
            while out and (out[-1] in stop_id_set or (newline_id is not None and out[-1] == newline_id)):
                out.pop()

            
            return out, {"stop_reason": stop_reason, "bor_count": bor_count, "eor_count": eor_count}

        def _build_pre_output_tokens(
            gi: int,
            group_query_tokens: List[List[int]],
            group_suffix_lens: List[int],
            group_n_spans: List[int],
        ) -> Tuple[List[int], List[int]]:
            if chat_enabled:
                memory_start = self._tokenizer.encode("<|im_start|>memory\n", bos=False, eos=False)
                user_start = self._tokenizer.encode("<|im_start|>user\n", bos=False, eos=False)
                assistant_start = self._tokenizer.encode("<|im_start|>assistant\n", bos=False, eos=False)
                im_end = self._tokenizer.encode("<|im_end|>\n", bos=False, eos=False)
                span_cost = num_comp + 2
                query_tokens = group_query_tokens[gi]
                fixed_len = (
                    len(memory_start)
                    + len(im_end)
                    + len(user_start)
                    + (1 if getattr(self, "_add_boq_index", False) else 0)
                    + len(query_tokens)
                    + len(im_end)
                    + len(assistant_start)
                )
                avail_for_memory = int(decoder_budget) - int(fixed_len) - int(group_suffix_lens[gi])
                max_spans = avail_for_memory // max(1, span_cost)
                if max_spans < 0:
                    max_spans = 0
                ret = self._format_chat(
                    user_text=group_query_list[gi],
                    contexts=group_doc_list[gi],
                    max_spans=max_spans,
                )
                prefix_tokens = list(ret.get("decoder_prefix_tokens") or [])
            else:
                prefix_tokens = []
                for _ in range(int(group_n_spans[gi])):
                    prefix_tokens.extend([BEGIN_OF_MEMORY_INDEX] + ([0] * num_comp) + [END_OF_MEMORY_INDEX])
                if self._add_boq_index:
                    prefix_tokens.append(BEGIN_OF_QUERY_INDEX)
                if add_bor:
                    prefix_tokens.append(BEGIN_OF_RECONSTRUCTION_INDEX)

            pre_output_tokens = list(prefix_tokens) + list(group_recon_tokens[gi])
            if add_query:
                pre_output_tokens.extend(group_query_tokens[gi])
            return pre_output_tokens, list(prefix_tokens)

        res: List[Tuple[float, bool]] = []
        bs = max(1, int(self._vllm_reconstruct_batch_size))
        iterator = range(0, len(requests), bs)
        pbar = tqdm(iterator, disable=disable_tqdm or self.rank != 0, desc="native loglikelihood (reconstruct_first)")
        
        for batch_start in pbar:
            chunk = requests[batch_start : batch_start + bs]
            # ctx_str_list: List[str] = [ctx_str for (ctx_str, _, _) in chunk]
            ctx_tokens_list: List[List[int]] = [ctx for (_, ctx, _) in chunk]
            # Re-encode continuations from raw strings: reconstruct_first uses a synthetic prefix
            # (memory + reconstruction [+ optional query]) so TemplateLM._encode_pair() boundaries
            # are not applicable.
            cont_tokens_list = [
                self._tokenizer.encode(pair[1], bos=False, eos=False) if pair and len(pair) > 1 else []
                for (pair, _, _) in chunk
            ]
            prefix_tokens = self._get_likelihood_prefix_tokens("reconstruct_first")
            if prefix_tokens:
                cont_tokens_list = [prefix_tokens + list(cont) for cont in cont_tokens_list]

            # Per-request results in this chunk (keep order).
            chunk_results: List[Tuple[float, bool]] = [(float("-inf"), False)] * len(chunk)
            for i, cont in enumerate(cont_tokens_list):
                if not cont:
                    chunk_results[i] = (0.0, True)

            # Group by identical context tokens (e.g., multiple-choice options share the same prompt).
            groups: Dict[Tuple[int, ...], List[int]] = {}
            for i, ctx in enumerate(ctx_tokens_list):
                groups.setdefault(tuple(ctx), []).append(i)

            group_keys = list(groups.keys())
            # breakpoint()
            n_groups = len(group_keys)
            if n_groups == 0:
                res.extend(chunk_results)
                continue

            group_doc_list: List[str] = [""] * n_groups
            group_query_list: List[str] = [""] * n_groups
            group_query_tokens: List[List[int]] = [[] for _ in range(n_groups)]
            group_prompt_tokens: List[List[int]] = [[] for _ in range(n_groups)]
            group_max_cont_lens: List[int] = [0] * n_groups
            group_suffix_lens: List[int] = [0] * n_groups
            group_indices: List[List[int]] = [[] for _ in range(n_groups)]

            doc_and_context = self._get_doc_and_context(ctx_tokens_list=ctx_tokens_list, batch_start=batch_start)
            context_list = doc_and_context["context_list"]
            query_list = doc_and_context["query_list"]

            for group_idx, gk in enumerate(group_keys):
                idxs = groups[gk]
                rep = idxs[0]

                max_cont = max((len(cont_tokens_list[i]) for i in idxs), default=0)
                group_max_cont_lens[group_idx] = max_cont
                group_indices[group_idx] = idxs

                doc_text = context_list[rep]
                query_text = query_list[rep]
                group_doc_list[group_idx] = doc_text
                group_query_list[group_idx] = query_text

                # Always keep query tokens for debug; only append them for scoring in non-chat mode.
                qtoks = self._tokenizer.encode(query_text, bos=False, eos=False)
                group_query_tokens[group_idx] = qtoks

                if chat_enabled:
                    group_suffix_lens[group_idx] = int(max_cont)
                    if add_query:
                        group_suffix_lens[group_idx] += int(len(qtoks))
                    group_prompt_tokens[group_idx] = self._tokenizer.encode(doc_text, bos=False, eos=False)
                else:
                    group_suffix_lens[group_idx] = int(len(qtoks) + max_cont)
                    if add_query:
                        group_suffix_lens[group_idx] += int(len(qtoks))
                    # Compress the full prompt tokens (context + query).
                    group_prompt_tokens[group_idx] = list(ctx_tokens_list[rep])

            # Build compression prefix prompt_embeds per unique context (once per group).
            dummy_prompts = [""] * n_groups
            prefix_embeds_list, meta = self._build_compress_prompt_embeds_batch(
                dummy_prompts,
                group_suffix_lens,
                include_bor=add_bor,
                decoder_include_prompt_tokens=False,
                decoder_memory_layout="per_span",
                prompt_tokens_override=group_prompt_tokens,
                return_meta=True,
                not_add_boq_index=False,
                context_list=group_doc_list if chat_enabled else None,
                query_list=group_query_list if chat_enabled else None,
            )
            
            meta_n_spans = (meta or {}).get("n_spans", [0] * n_groups)
            meta_flat_ctx = (meta or {}).get("flat_ctx_len", [0] * n_groups)

            group_prefix_lens: List[int] = [0] * n_groups
            group_max_recon_lens: List[int] = [0] * n_groups
            group_n_spans: List[int] = [0] * n_groups
            group_n_slots: List[int] = [0] * n_groups

            for gi, pe in enumerate(prefix_embeds_list):
                if pe is None:
                    continue
                pl = int(pe.shape[0])
                group_prefix_lens[gi] = pl
                n_spans = int(meta_n_spans[gi]) if gi < len(meta_n_spans) else 1
                if n_spans <= 0:
                    n_spans = 1
                group_n_spans[gi] = n_spans
                group_n_slots[gi] = num_comp * int(n_spans)
                # Reserve room for query suffix + the longest continuation option for this context.
                budget_recon = max(0, decoder_budget - pl - int(group_suffix_lens[gi]))
                flat_len = int(meta_flat_ctx[gi]) if gi < len(meta_flat_ctx) else budget_recon
                group_max_recon_lens[gi] = min(flat_len, budget_recon)

            # Reconstruct ONCE per unique context (group).
            group_recon_tokens: List[List[int]] = [[] for _ in range(n_groups)]
            group_recon_infos: List[dict] = [{"stop_reason": None} for _ in range(n_groups)]
            valid_gis = [gi for gi in range(n_groups) if prefix_embeds_list[gi] is not None and group_max_recon_lens[gi] > 0]
            if valid_gis:
                sampling_params: Dict[str, Any] = {
                    "temperature": float(self._temperature),
                    "top_p": 1.0,
                    "max_tokens": int(max(group_max_recon_lens[gi] for gi in valid_gis)),
                }
                batch_embeds = [prefix_embeds_list[gi] for gi in valid_gis]  # type: ignore[list-item]
                outputs = self._vllm_manager.generate_from_embeddings(batch_embeds, sampling_params=sampling_params)
                for gi, out in zip(valid_gis, outputs):
                    if not out.outputs:
                        continue
                    token_ids = out.outputs[0].token_ids or []
                    recon_tokens, info = _postprocess_recon(token_ids, group_max_recon_lens[gi])
                    group_recon_tokens[gi] = recon_tokens
                    group_recon_infos[gi] = info

            # Pre-embed query tokens per group to avoid re-encoding for every option.
            d_model = int(getattr(self.model.args, "d_model", 0))
            group_query_embeds: List[torch.Tensor] = []
            for qtoks in group_query_tokens:
                if add_query and qtoks:
                    q = torch.tensor(qtoks, device=self.device, dtype=torch.long)
                    with torch.autocast(device_type="cuda", dtype=self._dtype):
                        qe = self.model.tok_embeddings(q).to(dtype=self._dtype)
                    group_query_embeds.append(qe)
                else:
                    group_query_embeds.append(torch.empty((0, d_model), device=self.device, dtype=self._dtype))

            # Build base embeds per group: prefix_embeds + recon_embeds (+ optional query suffix).
            group_base_embeds: List[Optional[torch.Tensor]] = [None] * n_groups
            group_base_lens: List[int] = [0] * n_groups

            for gi, pe in enumerate(prefix_embeds_list):
                if pe is None:
                    continue
                recon = group_recon_tokens[gi]
                with torch.autocast(device_type="cuda", dtype=self._dtype):
                    if recon:
                        rt = torch.tensor(recon, device=self.device, dtype=torch.long)
                        re = self.model.tok_embeddings(rt).to(dtype=self._dtype)
                    else:
                        re = torch.empty((0, d_model), device=self.device, dtype=self._dtype)
                qe = group_query_embeds[gi]
                base = torch.cat([pe, re, qe], dim=0) if (re.numel() or qe.numel()) else pe
                group_base_embeds[gi] = base
                group_base_lens[gi] = int(base.shape[0])

            # Reconstruct-first legacy path uses an all-false compression mask in decoder scoring.
            group_base_masks: List[Optional[torch.Tensor]] = [None] * n_groups
            for gi, base in enumerate(group_base_embeds):
                if base is None:
                    continue
                group_base_masks[gi] = torch.zeros(int(group_base_lens[gi]), device=self.device, dtype=torch.bool)

            rows_per_chunk = int(getattr(self.model.args, "cross_entropy_chunk", 8)) * 16
            rows_per_chunk = max(8, min(int(rows_per_chunk), 512))

            option_stats: Dict[int, Dict[str, Any]] = {}
            fit_pairs: List[Tuple[int, int]] = []

            for gi, idxs in enumerate(group_indices):
                base = group_base_embeds[gi]
                base_mask = group_base_masks[gi]
                base_len = int(group_base_lens[gi])
                if base is None or base_mask is None or base_len <= 0:
                    continue
                for req_idx in idxs:
                    cont = cont_tokens_list[req_idx]
                    if not cont:
                        option_stats[req_idx] = {
                            "ll": 0.0,
                            "greedy": True,
                            "tokens": 0,
                            "loss": 0.0,
                            "ppl": None,
                        }
                        continue
                    if base_len + len(cont) <= decoder_budget:
                        fit_pairs.append((req_idx, gi))

            # Batch score options that fit.
            score_bs = max(1, int(self._ppl_batch_size))
            for score_start in range(0, len(fit_pairs), score_bs):
                pairs = fit_pairs[score_start : score_start + score_bs]
                if bool(getattr(self, "_verbose_compress", False)):
                    print("pairs,", pairs, "score_start,", score_start, "score_bs,", score_bs)

                seq_embeds: List[torch.Tensor] = []
                pref_lens: List[int] = []
                cont_targets: List[torch.Tensor] = []
                comp_masks: List[torch.Tensor] = []
                orig_req_idxs: List[int] = []

                for req_idx, gi in pairs:
                    base = group_base_embeds[gi]
                    base_mask = group_base_masks[gi]
                    if base is None or base_mask is None:
                        continue
                    cont = cont_tokens_list[req_idx]
                    if not cont:
                        continue
                    cont_t = torch.tensor(cont, device=self.device, dtype=torch.long)
                    with torch.autocast(device_type="cuda", dtype=self._dtype):
                        cont_e = self.model.tok_embeddings(cont_t).to(dtype=self._dtype)
                    seq = torch.cat([base, cont_e], dim=0)
                    seq_embeds.append(seq)
                    base_len = int(group_base_lens[gi])
                    pref_lens.append(base_len)
                    cont_targets.append(cont_t)
                    comp_masks.append(torch.cat([base_mask, torch.zeros(len(cont), device=self.device, dtype=torch.bool)], dim=0))
                    orig_req_idxs.append(req_idx)

                if not seq_embeds:
                    continue

                score_out = self._forward_score_continuations(
                    seq_embeds=seq_embeds,
                    cont_targets=cont_targets,
                    prefix_lens=pref_lens,
                    comp_mask_list=comp_masks,
                    rows_per_chunk=rows_per_chunk,
                )
                per_sample = score_out.get("per_sample", [])
                for j, req_idx in enumerate(orig_req_idxs):
                    if j >= len(per_sample):
                        continue
                    ps = per_sample[j]
                    if ps.get("tokens", 0) <= 0:
                        continue
                    chunk_results[req_idx] = (ps["ll"], ps["greedy"])
                    option_stats[req_idx] = ps

            # Debug rows: interleave group summary with its options.
            debug_rows: List[dict] = []
            verbose_payload = bool(getattr(self, "_verbose_compress", False))
            for gi in range(n_groups):
                pre_tokens_full, _ = _build_pre_output_tokens(gi, group_query_tokens, group_suffix_lens, group_n_spans)
                pre_tokens_no_prefix = list(group_recon_tokens[gi])
                if add_query:
                    pre_tokens_no_prefix.extend(group_query_tokens[gi])
                info = group_recon_infos[gi]
                group_first_idx = group_indices[gi][0] if group_indices[gi] else 0

                group_row = {
                    "request_index": batch_start + group_first_idx,
                    "mode": "reconstruct_first",
                    "use_chat_template": chat_enabled,
                    "add_query_before_likelihood": add_query,
                    "reconstruct_add_bor": add_bor,
                    "reconstruct_max_bor": max_bor,
                    "num_comp": num_comp,
                    "max_mem_span_len": max_mem_span_len,
                    "n_spans": group_n_spans[gi],
                    "total_comp_slots": group_n_slots[gi],
                    "prefix_len": group_prefix_lens[gi],
                    "base_len": group_base_lens[gi],
                    "max_recon_len": group_max_recon_lens[gi],
                    "recon_tokens_len": len(group_recon_tokens[gi]),
                    "recon_stop_reason": info.get("stop_reason"),
                    "query_len": len(group_query_tokens[gi]),
                    "group_max_cont_len": group_max_cont_lens[gi],
                    "recon_text_preview": self.tok_decode_w_special_tokens(group_recon_tokens[gi]) if group_recon_tokens[gi] else "",
                    # "pre_output_text_preview": self.tok_decode_w_special_tokens(pre_tokens_full) if pre_tokens_full else "",
                    "pre_output_text_no_prefix_preview": self.tok_decode_w_special_tokens(pre_tokens_no_prefix)
                    if pre_tokens_no_prefix
                    else "",
                }
                if verbose_payload:
                    group_row["recon_text"] = self.tok_decode_w_special_tokens(group_recon_tokens[gi]) if group_recon_tokens[gi] else ""
                    group_row["pre_output_text"] = self.tok_decode_w_special_tokens(pre_tokens_full) if pre_tokens_full else ""
                    group_row["pre_output_text_no_prefix"] = (
                        self.tok_decode_w_special_tokens(pre_tokens_no_prefix) if pre_tokens_no_prefix else ""
                    )
                debug_rows.append(group_row)

                for req_idx in group_indices[gi]:
                    ps = option_stats.get(req_idx)
                    opt_row = {
                        "request_index": batch_start + req_idx,
                        "mode": "reconstruct_first_option",
                        "add_query_before_likelihood": add_query,
                        "cont_len": len(cont_tokens_list[req_idx]),
                        "logprob": chunk_results[req_idx][0],
                        "greedy": chunk_results[req_idx][1],
                    }
                    if ps is not None and int(ps.get("tokens", 0) or 0) > 0:
                        opt_row["loss"] = ps.get("loss")
                        opt_row["ppl"] = ps.get("ppl")
                        opt_row["score_tokens"] = int(ps.get("tokens", 0) or 0)
                        if "windows" in ps:
                            opt_row["windows"] = int(ps.get("windows") or 0)
                        if "rolled" in ps:
                            opt_row["rolled"] = bool(ps.get("rolled"))
                    debug_rows.append(opt_row)

            self._append_loglikelihood_debug_rows(debug_rows)

            res.extend(chunk_results)

        return res
