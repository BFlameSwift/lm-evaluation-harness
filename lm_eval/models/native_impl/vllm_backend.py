"""
vLLM backend helpers for the `native` model.

This module isolates vLLM-specific concerns:
- exporting native checkpoints to a vLLM-loadable `safemodel/` directory
- patching `config.json` / `generation_config.json` to match `max_model_len`
- initializing local or remote vLLM engines
- best-effort shutdown to release GPU memory

Keeping this logic out of `model.py` makes the native adapter easier to read and
reduces the chance of accidental cross-coupling between likelihood/generation
paths and backend init/teardown.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
from typing import Any, List

import torch


def init_vllm_param(self) -> None:
    """Resolve/prepare a vLLM model directory (including safetensors export when needed)."""
    if getattr(self, "_use_remote_vllm", False):
        # A remote/persistent vLLM server owns model weights/config; skip local
        # safetensors conversion and config patching.
        return

    try:
        from eval_func.model2safetensors import convert_checkpoint, safemodel_needs_reconvert  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("vLLM safemodel export requires `eval_func.model2safetensors`.") from e

    model_path = self._vllm_model_path
    if model_path is None:
        # HF checkpoints: vLLM can load directly from checkpoint_dir (already a transformers directory).
        is_native_ckpt = self._vllm_checkpoint_dir is not None and os.path.exists(
            os.path.join(self._vllm_checkpoint_dir, "metadata.json")
        )
        if not is_native_ckpt:
            model_path = self._vllm_checkpoint_dir
        else:
            base_dir = self._vllm_output_root or self._vllm_checkpoint_dir
            if base_dir is None:
                raise ValueError(
                    "vLLM reconstruction requires vllm_model_path or checkpoint_dir (or vllm_output_root)."
                )

            def _safemodel_ready(safedir: str) -> bool:
                model_file = os.path.join(safedir, "model.safetensors")
                cfg_file = os.path.join(safedir, "config.json")
                if not (os.path.exists(model_file) and os.path.exists(cfg_file)):
                    return False
                try:
                    return not safemodel_needs_reconvert(safedir)
                except Exception:
                    return False

            convert_kwargs = {
                "checkpoint_dir": self._vllm_checkpoint_dir,
                "tokenizer_path": self._vllm_tokenizer_path,
                "dtype": self._dtype,
                "additional_kwargs": {
                    "max_position_embeddings": self._vllm_max_model_len,
                    "eos_token_id": self.eos_token_id,
                    "pad_token_id": self.pad_token_id,
                    "bos_token_id": getattr(self._tokenizer, "bos_id", None),
                    # Keep for backwards compatibility with older prompt templates,
                    # even though vLLM doesn't consume it directly.
                    "temperature": self._temperature,
                    "max_seq_len": self._max_seq_length,
                },
            }

            def _resolve_local_safedir() -> str:
                local_root = os.environ.get("NATIVE_VLLM_LOCAL_SAFEMODEL_ROOT") or "/tmp"
                local_root = os.path.abspath(os.path.expanduser(str(local_root)))
                key = "|".join(
                    [
                        f"ckpt={os.path.abspath(str(self._vllm_checkpoint_dir))}",
                        f"tok={os.path.abspath(str(self._vllm_tokenizer_path)) if self._vllm_tokenizer_path else ''}",
                        f"dtype={str(self._dtype)}",
                        f"maxlen={int(self._vllm_max_model_len or 0)}",
                    ]
                )
                digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
                return os.path.join(local_root, "native_vllm_safemodel", digest)

            # Optional hard switch to avoid blob/network filesystems entirely.
            prefer_local = str(os.environ.get("NATIVE_VLLM_FORCE_LOCAL_SAFEMODEL", "")).strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
            }
            if prefer_local:
                local_safedir = _resolve_local_safedir()
                os.makedirs(local_safedir, exist_ok=True)
                if not _safemodel_ready(local_safedir):
                    convert_checkpoint(output_dir=local_safedir, **convert_kwargs)
                ensure_vllm_config(self, local_safedir)
                self._vllm_model_dir = local_safedir
                model_path = local_safedir
                return

            # Prefer exporting safemodel into the lm-eval output directory so results
            # are colocated. Some blob/network filesystems fail during atomic renames,
            # so we fall back to a local temp directory (default: /tmp).
            remote_safedir = os.path.join(base_dir, "safemodel")
            try:
                if not _safemodel_ready(remote_safedir):
                    os.makedirs(remote_safedir, exist_ok=True)
                    convert_checkpoint(output_dir=remote_safedir, **convert_kwargs)
                ensure_vllm_config(self, remote_safedir)
                self._vllm_model_dir = remote_safedir
                model_path = remote_safedir
            except Exception as e:
                local_safedir = _resolve_local_safedir()
                print(
                    f"[native][warn] vLLM safemodel export to '{remote_safedir}' failed: {type(e).__name__}: {e}. "
                    f"Falling back to local safemodel at '{local_safedir}'.",
                    file=sys.stderr,
                )
                os.makedirs(local_safedir, exist_ok=True)

                # Best-effort cross-process lock to avoid duplicating huge exports.
                lock_path = os.path.join(local_safedir, ".export.lock")
                try:
                    import fcntl  # type: ignore
                except Exception:
                    fcntl = None
                with open(lock_path, "w", encoding="utf-8") as lock_f:
                    if fcntl is not None:
                        try:
                            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
                        except Exception:
                            pass
                    if not _safemodel_ready(local_safedir):
                        convert_checkpoint(output_dir=local_safedir, **convert_kwargs)
                    ensure_vllm_config(self, local_safedir)
                self._vllm_model_dir = local_safedir
                model_path = local_safedir

    if model_path is not None:
        # Propagate resolved model path so vLLM init uses it.
        self._vllm_model_dir = model_path


def init_vllm(self) -> None:
    """Initialize a vLLM engine wrapper (local or remote) and attach `self._vllm_manager`."""
    try:
        from eval_func.vllm_runner import (  # type: ignore
            VLLMDecoderManager,
            VLLMEngineConfig,
            VLLMEngineWrapper,
            VLLMRemoteEngineWrapper,
        )
    except Exception as e:  # pragma: no cover
        raise ImportError("vLLM initialization requires `eval_func.vllm_runner` (and vLLM).") from e

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

    try:
        cfg = VLLMEngineConfig(
            model_path=model_path,
            tensor_parallel_size=self._vllm_tensor_parallel,
            dtype=self._dtype,
            max_model_len=self._vllm_max_model_len or self._max_seq_length,
            enforce_eager=bool(getattr(self, "_vllm_enforce_eager", False)),
            enable_prompt_embeds=True,
            tokenizer=self._vllm_tokenizer_path or self._vllm_checkpoint_dir,
            additional_kwargs={"gpu_memory_utilization": self._vllm_gpu_memory_utilization},
        )
        engine = VLLMEngineWrapper(cfg)
        self._vllm_manager = VLLMDecoderManager(engine_wrapper=engine)
    except Exception as e:
        print(f"WARNING: Failed to init vLLM, falling back to torch backend. Error: {e}", file=sys.stderr)
        self._vllm_manager = None


def ensure_vllm_manager(self, *, caller: str) -> None:
    """
    Best-effort lazy vLLM initialization.

    Some eval modes only need vLLM for generation/reconstruction. Initializing vLLM eagerly can
    both slow down scoring-only tasks and introduce avoidable failures.
    """
    if getattr(self, "_vllm_manager", None) is not None:
        return
    if bool(getattr(self, "_vllm_init_attempted", False)):
        return
    setattr(self, "_vllm_init_attempted", True)
    try:
        init_vllm_param(self)
        init_vllm(self)
    except Exception as e:
        self._vllm_manager = None
        print(f"WARNING: Failed to init vLLM ({caller}). Error: {e}", file=sys.stderr)


def ensure_vllm_config(self, safedir: str) -> None:
    """Patch vLLM config files so generation respects `max_model_len`."""
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

