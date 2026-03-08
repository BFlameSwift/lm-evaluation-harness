"""Shared helpers for the `native` model backend.

This module holds small, reusable utilities that are used across multiple
subsystems (likelihood / generation / reconstruction) and do not need to live
in the large `NativeCausalLM` class implementation.

Design goals:
- Keep helpers pure where possible.
- Avoid importing GPU-heavy optional deps at import time.
- Reduce cross-module coupling (e.g., `reconstruct.py` should not import
  `model.py` just to get a tiny helper).
"""

from __future__ import annotations

import inspect
import json
import os
import re
from typing import Any, Dict, Mapping, Optional, Type, TypeVar

import torch

T = TypeVar("T")


def filter_kwargs_for(cls: Type[T], raw_kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Filter a kwargs mapping so it is safe to pass to ``cls(...)``.

    This is primarily used for "optional" kwargs that depend on the target
    function signature (e.g., some checkpoints accept `show_progress=True`,
    others do not). We inspect the signature and keep only keyword parameters.

    Notes:
    - If the class accepts `**kwargs`, this function still filters explicitly
      so call sites can be deterministic.
    """
    sig = inspect.signature(cls)
    valid_names = {
        name
        for name, param in sig.parameters.items()
        if param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and name != "self"
    }
    return {k: v for k, v in raw_kwargs.items() if k in valid_names}


def filter_kwargs_for_callable(fn: Any, raw_kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Filter a kwargs mapping so it is safe to call ``fn(**filtered)``.

    If ``fn`` has a `**kwargs` parameter, we return the raw mapping unchanged.
    Otherwise, we only keep recognized keyword arguments.
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


def str_to_dtype(name: Optional[str]) -> torch.dtype:
    """Parse a user-facing dtype string into a ``torch.dtype``.

    Accepted aliases:
    - bf16/bfloat16
    - fp16/float16/half
    - fp32/float32

    `None`/`"auto"` defaults to bf16 (matches most of our evaluation configs).
    """
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


def parse_mode(name: Optional[str]) -> str:
    """Validate/normalize the `native` execution mode string.

    This is used to parse `--model_args mode=...` from lm-eval CLI.
    """
    if name is None:
        return "decoder"
    name = name.lower()
    if name not in {
        "decoder",
        "compress_answer",
        "reconstruct_first",
        "vllm_decoding_with_compress",
        "niah_generate",
    }:
        raise ValueError(f"Unsupported native model mode: {name}")
    return name


def coerce_int(value: Optional[Any], default: Optional[int] = None) -> Optional[int]:
    """Best-effort convert a value into an int, with common string conventions.

    Examples:
    - `"123"` -> 123
    - `"none"` / `""` -> default
    - `True` -> 1
    """
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


def coerce_bool(value: Optional[Any], default: bool = False) -> bool:
    """Best-effort convert a value into a bool, with common string conventions.

    Examples:
    - `"1"`, `"true"`, `"yes"` -> True
    - `"0"`, `"false"`, `"no"` -> False
    - `""` / `"none"` -> default
    """
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"", "none"}:
            return bool(default)
        if raw in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if raw in {"0", "false", "f", "no", "n", "off"}:
            return False
    return bool(value)


def normalize_optional_text(value: Optional[Any]) -> str:
    """Normalize "optional string" style arguments.

    Convention:
    - `None`, `""`, `"none"` => `""`
    - otherwise return a string.

    Important: for backward compatibility, we return the original string (not
    the stripped one) when it is non-empty. Callers that need stripping should
    apply `str(...).strip()` themselves.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        raw = value.strip()
        if not raw or raw.lower() == "none":
            return ""
        return value
    return str(value)


def parse_json_dict_arg(value: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Best-effort parse a JSON object string into a dict.

    Returns `None` for empty values. Raises `ValueError` for invalid/non-dict JSON
    so callers fail loudly instead of silently ignoring a malformed long-context config.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        raise ValueError(f"Expected JSON object string or dict, got {type(value).__name__}")
    raw = value.strip()
    if not raw or raw.lower() == "none":
        return None
    try:
        loaded = json.loads(raw)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON object: {e}") from e
    if loaded is None:
        return None
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object, got {type(loaded).__name__}")
    return dict(loaded)


def build_rope_scaling_config(
    *,
    rope_scaling_json: Optional[Any] = None,
    rope_scaling_type: Optional[Any] = None,
    rope_scaling_factor: Optional[Any] = None,
    rope_scaling_original_max_position_embeddings: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Build a rope_scaling dict from either JSON or scalar CLI-friendly fields."""
    rope = parse_json_dict_arg(rope_scaling_json)
    rope_type = normalize_optional_text(rope_scaling_type).strip().lower()
    factor = rope_scaling_factor
    orig_max = coerce_int(rope_scaling_original_max_position_embeddings, None)
    if rope is None and not rope_type and factor is None and orig_max is None:
        return None
    rope = dict(rope or {})
    if rope_type:
        rope["rope_type"] = rope_type
        rope.setdefault("type", rope_type)
    if factor is not None:
        try:
            rope["factor"] = float(factor)
        except Exception as e:
            raise ValueError(f"Invalid rope_scaling_factor={factor!r}: {e}") from e
    if orig_max is not None and orig_max > 0:
        rope["original_max_position_embeddings"] = int(orig_max)
    return rope or None


def derive_lm_eval_output_dir(
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
    out_path = normalize_optional_text(output_path)
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


def token_embed(model: torch.nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
    """Lookup token embeddings for ``token_ids``.

    Our `arch` checkpoints expose a LLaMA/Qwen-style attribute `tok_embeddings`.
    For maximum compatibility we also support HF-style `get_input_embeddings()`.

    Args:
        model: The loaded model module.
        token_ids: 1D/2D tensor of token ids (dtype long) on the target device.

    Returns:
        The embedding tensor. Shape matches the input token shape with an extra
        hidden dimension (e.g., `[T, D]` for a 1D token list).
    """
    if hasattr(model, "tok_embeddings"):
        return model.tok_embeddings(token_ids)  # type: ignore[attr-defined]
    if hasattr(model, "get_input_embeddings"):
        emb_layer = model.get_input_embeddings()
        if emb_layer is not None:
            return emb_layer(token_ids)
    raise AttributeError("Model does not expose token embedding layer (tok_embeddings/get_input_embeddings)")
