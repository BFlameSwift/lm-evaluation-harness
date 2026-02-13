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
import os
import re
from typing import Any, Dict, Mapping, Optional, Type, TypeVar

import torch

T = TypeVar("T")


def filter_kwargs_for(cls: Type[T], raw_kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Filter kwargs for a class __init__ method."""
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
    """Filter kwargs for an arbitrary callable (bound method/function)."""
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
    if value is None:
        return ""
    if isinstance(value, str):
        raw = value.strip()
        if not raw or raw.lower() == "none":
            return ""
        return value
    return str(value)


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
    if hasattr(model, "tok_embeddings"):
        return model.tok_embeddings(token_ids)  # type: ignore[attr-defined]
    if hasattr(model, "get_input_embeddings"):
        emb_layer = model.get_input_embeddings()
        if emb_layer is not None:
            return emb_layer(token_ids)
    raise AttributeError("Model does not expose token embedding layer (tok_embeddings/get_input_embeddings)")

