import os
import json
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
from eval_func.vllm_runner import VLLMEngineWrapper, VLLMEngineConfig, VLLMDecoderManager
from eval_func.utils import load_checkpoint_harness, _build_device_mesh
import sys
import math
from typing import Type, TypeVar, Mapping, Any, Dict
import inspect
import gc

T = TypeVar("T")



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
    if name not in {"decoder", "compress_answer", "reconstruct_first", "vllm_decoding_with_compress"}:
        raise ValueError(f"Unsupported native model mode: {name}")
    return name


def _default_distributed_args() -> DistributedArgs:
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return DistributedArgs(rank=rank, local_rank=local_rank, world_size=world_size)




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
    """

    backend = "causal"

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        pretrain_model_dir: Optional[str] = None,
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
        vllm_output_root: Optional[str] = None,
        # for reconstruction related
        vllm_reconstruct_batch_size: int = 20,
        ppl_batch_size: Optional[int] = 8,
        # for compression related
        compress_threshold: int = 8192,
        compress_chunk: int = 2048,
        max_cycles: int = 10,
        compress_start_tokens: Optional[str] = "<think>",
        temperature: float = 1.0,
        # chat template related， also load from yaml task
        use_chat_template: bool = False, 
        chat_add_generation_prompt: bool = True,
        add_thinking_tokens: bool = False,
        # reconstruct_first (loglikelihood) controls
        reconstruct_add_bor: bool = False,
        reconstruct_max_bor: int = 3,
        save_loglikelihood_debug: bool = True,
        loglikelihood_debug_path: Optional[str] = None,
        shutdown_vllm_after_loglikelihood: bool = False,
        shutdown_vllm_kill_children: bool = False,
        shutdown_vllm_verbose: bool = False,
        add_boq_index: bool = True,
        
        fill_decoder_prefix_embeds: bool = True,
    ) -> None:
        super().__init__()
        self._dtype = _str_to_dtype(dtype)
        self._device = torch.device("cuda")
        self._batch_size = int(batch_size) if isinstance(batch_size, (int, float)) or str(batch_size).isdigit() else 1
        self._mode = _parse_mode(mode)
        self._max_mem_span_len_override = max_mem_span_len
        self._use_vllm_reconstruct = use_vllm_reconstruct
        self._use_vllm_decoder = use_vllm_decoder
        self._use_vllm_answer = use_vllm_answer
        self._vllm_manager = None
        self._vllm_output_root = vllm_output_root
        self._last_generate_debug: List[dict] = []
        self._vllm_reconstruct_batch_size = max(1, int(vllm_reconstruct_batch_size))
        self._ppl_batch_size = max(1, int(ppl_batch_size)) if ppl_batch_size is not None else self._batch_size
        self._compress_threshold = max(1, int(compress_threshold))
        self._compress_chunk = max(1, int(compress_chunk))
        self._max_cycles = max(1, int(max_cycles))
        self._compress_start_tokens = []
        self._add_boq_index = bool(add_boq_index)
        self._fill_decoder_prefix_embeds = bool(fill_decoder_prefix_embeds)
        if compress_start_tokens:
            for t in compress_start_tokens.split(","):
                t = t.strip()
                if t:
                    self._compress_start_tokens.append(t)
        self._chat_use_template = bool(use_chat_template)
        self._chat_add_generation_prompt = bool(chat_add_generation_prompt)
        self._add_thinking_tokens = bool(add_thinking_tokens)
        self._temperature = temperature
        self._reconstruct_add_bor = bool(reconstruct_add_bor)
        self._reconstruct_max_bor = max(0, int(reconstruct_max_bor))
        self._save_loglikelihood_debug = bool(save_loglikelihood_debug)
        self._loglikelihood_debug_path = loglikelihood_debug_path
        self._last_loglikelihood_debug: List[dict] = []
        self._shutdown_vllm_after_loglikelihood = bool(shutdown_vllm_after_loglikelihood)
        self._shutdown_vllm_kill_children = bool(shutdown_vllm_kill_children)
        self._shutdown_vllm_verbose = bool(shutdown_vllm_verbose)
        distributed_args = _default_distributed_args()
        torch.cuda.set_device(distributed_args.local_rank)
        # Native supports tensor-parallel only; data-parallel (world_size > model_parallel_size) will duplicate work.
        self._distributed_args = distributed_args
        
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

        if max_seq_length is not None:
            self._max_seq_length = int(max_seq_length)
        else:
            if hasattr(self.model, "args"):
                self._max_seq_length = int(self.model.args.max_seq_len)
            else:
                self._max_seq_length = int(
                    getattr(getattr(self.model, "config", None), "max_position_embeddings", 2048) or 2048
                )
        if self._max_mem_span_len_override is not None:
            # Respect override for compression-aware paths
            if hasattr(self.model, "args"):
                self.model.args.max_mem_span_len = self._max_mem_span_len_override

        # Optional vLLM init for reconstruction speedup or decoder fast path (decoder-only, prompt_embeds)
        need_vllm = self._use_vllm_reconstruct or self._use_vllm_decoder or self._use_vllm_answer
        if self._mode == "vllm_decoding_with_compress":
            need_vllm = True
        # breakpoint()

        if need_vllm:
            # Prepare decoder-only safetensors if path not provided
            model_path = vllm_model_path
            # breakpoint()
            if model_path is None:
                # HF checkpoints: vLLM can load directly from checkpoint_dir (already a transformers directory).
                is_native_ckpt = checkpoint_dir is not None and os.path.exists(os.path.join(checkpoint_dir, "metadata.json"))
                if not is_native_ckpt:
                    model_path = checkpoint_dir
                else:
                    base_dir = self._vllm_output_root or checkpoint_dir
                    if base_dir is None:
                        raise ValueError("vLLM reconstruction requires vllm_model_path or checkpoint_dir (or vllm_output_root).")
                    safedir = os.path.join(base_dir, "safemodel")
                    model_path = safedir
                    need_convert = not os.path.exists(os.path.join(safedir, "model.safetensors")) or not os.path.exists(
                        os.path.join(safedir, "config.json")
                    )
                    if need_convert:
                        os.makedirs(safedir, exist_ok=True)
                        convert_checkpoint(
                            checkpoint_dir=checkpoint_dir,
                            output_dir=safedir,
                            tokenizer_path=tokenizer_path,
                            dtype=str(self._dtype).replace("torch.", ""),
                            additional_kwargs={
                                "max_position_embeddings": self._max_seq_length,
                                "eos_token_id": self.eos_token_id,
                                "pad_token_id": self.pad_token_id,
                                "bos_token_id": getattr(self._tokenizer, "bos_id", None),
                                "temperature": self._temperature,
                                "max_seq_len": self._max_seq_length,
                            },
                        )
            # breakpoint()
            try:
                # breakpoint()
                cfg = VLLMEngineConfig(
                    model_path=model_path,
                    tensor_parallel_size=vllm_tensor_parallel,
                    # dtype=str(self._dtype).replace("torch.", ""),
                    max_model_len= vllm_max_model_len or self._max_seq_length,
                    enable_prompt_embeds=True,
                    tokenizer=tokenizer_path or checkpoint_dir,
                    additional_kwargs={"gpu_memory_utilization": vllm_gpu_memory_utilization},
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

    def __del__(self) -> None:
        # Never raise from a destructor.
        try:
            self.shutdown_vllm_manager()
        except Exception as e:
            print(f"WARNING: Failed to shutdown vLLM in destructor, Error: {e}", file=sys.stderr)

    # ---- Required TemplateLM properties ----
    @property
    def eot_token_id(self) -> int:
        return self._tokenizer.eos_id

    @property
    def max_length(self) -> int:
        return self._max_seq_length

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
            if not self._vllm_output_root:
                return
            os.makedirs(self._vllm_output_root, exist_ok=True)
            out_path = os.path.join(self._vllm_output_root, f"loglikelihood_debug_{datetime_str}.jsonl")
            self._loglikelihood_debug_path = out_path
        else:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        
        with open(out_path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- Tokenization helpers ----
    def tok_encode(self, string: str, add_special_tokens: Optional[bool] = None,add_thinking_tokens: Optional[bool] = False, **kwargs) -> List[int]:
        # native tokenizer already includes BOS/EOS control; keep minimal
        if add_thinking_tokens:
            string = string + "<think>"
            return self._tokenizer.encode(string, bos=False, eos=False)
        else:
            return self._tokenizer.encode(string, bos=False, eos=False)

    def _format_chat(self, user_text: str, assistant_text: Optional[str] = None, add_generation_prompt: Optional[bool] = None) -> str:
        """
        Optionally wrap text with chat template. If assistant_text is None, will
        produce a prompt that expects model generation; otherwise returns a full
        conversation with assistant content included.
        """
        if not self._chat_use_template:
            return user_text if assistant_text is None else user_text + "\n" + assistant_text
        try:
            add_gen = self._chat_add_generation_prompt if add_generation_prompt is None else add_generation_prompt
            messages = [{"role": "user", "content": user_text}]
            if assistant_text is not None:
                messages.append({"role": "assistant", "content": assistant_text})
                add_gen = False if add_generation_prompt is None else add_generation_prompt
            return self._tokenizer.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_gen)
        except Exception:
            # fallback to raw text on any failure
            return user_text if assistant_text is None else user_text + "\n" + assistant_text

    def tok_decode(self, tokens: List[int]) -> str:
        return self._tokenizer.decode(tokens)
    def tok_decode_w_special_tokens(self, tokens: List[int]) -> str:
        return self._tokenizer.decode_w_special_tokens(tokens)

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
        total_lp = 0.0
        greedy_ok = True
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            h = hidden[s:e]
            t = targets[s:e]
            logits = self.model.output(h).float()
            if self._model_parallel_group is not None:
                from distributed.tensor_parallel import gather_from_model_parallel_region

                logits = gather_from_model_parallel_region(logits, self._model_parallel_group)
            logprobs = F.log_softmax(logits, dim=-1)
            total_lp += float(logprobs.gather(-1, t.unsqueeze(-1)).squeeze(-1).sum().item())
            if greedy_ok:
                greedy_ok = bool((logprobs.argmax(dim=-1) == t).all().item())
        return total_lp, greedy_ok
    
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

    def _compress_tokens(
        self,
        enc_tokens_flat: torch.Tensor,
        enc_cu: torch.Tensor,
        enc_mem_mask_flat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compress packed encoder tokens into compression vectors.

        Important:
        - We must never create a single encoder sequence whose `positions` exceed the model max length
          (e.g. 8192), otherwise rotary embedding indexing will fail.
        - We also must not split a (span_tokens + placeholders) unit across chunks, or the placeholder
          tokens would not attend to their corresponding span.

        Strategy:
        - For each packed sample, drop placeholder tokens (mask=True) to recover raw span tokens.
        - Chunk raw tokens by `max_mem_span_len`, append `num_comp` placeholders per chunk, and run
          `compress_multi_batches` (preferred) on those per-span micro-batches.
        """
        num_comp = int(getattr(self.model.args, "num_compression_tokens", 0))
        if num_comp <= 0:
            return torch.empty(0, device=self.device, dtype=self._dtype)

        span_limit = getattr(self.model.args, "max_mem_span_len", None)
        if span_limit is None or span_limit <= 0:
            span_limit = self._max_mem_span_len_override or self.max_length
        span_limit = max(1, int(span_limit))

        has_multi = hasattr(self.model, "compress_multi_batches")

        enc_tokens_mb: List[torch.Tensor] = []
        enc_ctx_mb: List[Dict[str, torch.Tensor]] = []

        # Build per-span micro-batches (each <= max_mem_span_len + num_comp tokens).
        for i in range(max(0, int(enc_cu.numel()) - 1)):
            s = int(enc_cu[i].item())
            e = int(enc_cu[i + 1].item())
            seq_tokens = enc_tokens_flat[s:e]
            seq_mask = enc_mem_mask_flat[s:e]

            # Recover raw tokens (drop placeholder positions).
            if seq_tokens.numel() == 0:
                raw = torch.empty(0, device=self.device, dtype=torch.long)
            else:
                raw = seq_tokens[~seq_mask]

            # Ensure at least one span so downstream slot accounting stays consistent.
            span_chunks: List[torch.Tensor]
            if raw.numel() == 0:
                span_chunks = [raw]
            else:
                span_chunks = [raw[j : j + span_limit] for j in range(0, raw.numel(), span_limit)]

            for chunk in span_chunks:
                # tokens + placeholders
                enc_seq = torch.cat(
                    [chunk.to(dtype=torch.long), torch.full((num_comp,), 0, device=self.device, dtype=torch.long)],
                    dim=0,
                )
                enc_mask_seq = torch.cat(
                    [
                        torch.zeros(int(chunk.numel()), device=self.device, dtype=torch.bool),
                        torch.ones(num_comp, device=self.device, dtype=torch.bool),
                    ],
                    dim=0,
                )
                clen = int(enc_seq.numel())
                cu = torch.tensor([0, clen], device=self.device, dtype=torch.int32)
                enc_tokens_mb.append(enc_seq)
                enc_ctx_mb.append(
                    {
                        "cu_seqlens_q": cu,
                        "cu_seqlens_k": cu,
                        "max_seqlen_q": clen,
                        "max_seqlen_k": clen,
                        "positions": torch.arange(clen, device=self.device, dtype=torch.int32),
                        "encoder_mem_mask": enc_mask_seq,
                    }
                )

        if not enc_tokens_mb:
            return torch.empty(0, device=self.device, dtype=self._dtype)

        with torch.autocast(device_type="cuda", dtype=self._dtype):
            if has_multi:
                return self.model.compress_multi_batches(enc_tokens_mb, enc_ctx_mb)

            # Fallback: run per-span compression sequentially to keep positions valid.
            outs: List[torch.Tensor] = []
            for t, ctx in zip(enc_tokens_mb, enc_ctx_mb):
                outs.append(self.model.compress(encoder_tokens=t, encoder_context=ctx))
            return torch.cat(outs, dim=0) if outs else torch.empty(0, device=self.device, dtype=self._dtype)

    def _compress_plain_sequence(self, tokens: torch.Tensor, num_comp: Optional[int] = None) -> torch.Tensor:
        """
        Compress a single token sequence without inserting slots/BOM/EOM.
        Useful for streaming/iterative compression where spans are managed externally.
        """
        if num_comp is None:
            num_comp = getattr(self.model.args, "num_compression_tokens", 0)
        span_limit = getattr(self.model.args, "max_mem_span_len", None)
        if span_limit is None or span_limit <= 0:
            max_len = self._max_seq_length
            span_limit = max_len - num_comp if max_len is not None else tokens.numel()
        span_limit = max(1, span_limit)
        has_multi = hasattr(self.model, "compress_multi_batches")

        if tokens.numel() == 0:
            return torch.empty(0, device=self.device, dtype=self._dtype)

        # we always append placeholders for compression tokens
        if num_comp <= 0:
            return torch.empty(0, device=self.device, dtype=self._dtype)

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
                return self.model.compress_multi_batches(enc_tokens_mb, enc_ctx_mb)

            # Fallback: run per-span compression sequentially to keep positions valid.
            outs: List[torch.Tensor] = []
            for t, ctx in zip(enc_tokens_mb, enc_ctx_mb):
                outs.append(self.model.compress(encoder_tokens=t, encoder_context=ctx))
            return torch.cat(outs, dim=0) if outs else torch.empty(0, device=self.device, dtype=self._dtype)

    # def _build_prompt_embeds(
    #     self,
    #     encoder_tokens: torch.Tensor,
    #     encoder_context: dict,
    #     decoder_prefix: torch.Tensor,
    #     compression_mask: torch.Tensor,
    # ) -> torch.Tensor:
    #     """Encode compression slots and return embeddings with slots filled."""
    #     with torch.inference_mode():
    #         compression_vectors = self._compress_tokens(encoder_tokens, encoder_context["cu_seqlens_q"], encoder_context["encoder_mem_mask"])
    #     prompt_embeds = self.model.tok_embeddings(decoder_prefix.to(self.device))
    #     prompt_embeds = prompt_embeds.to(compression_vectors.dtype)
    #     prompt_embeds[compression_mask] = compression_vectors
    #     return prompt_embeds

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
                logits = self.model(inps.flatten(), context=context, **kwargs)
        return logits.view(original_shape[0], original_shape[1], -1)

    @torch.no_grad()
    def _model_logits(self, logits: torch.Tensor) -> torch.Tensor:
        ctx = torch.autocast(device_type="cuda", dtype=self._dtype)
        with ctx:
            if hasattr(self.model, "compression_embeddings"):
                # logits here are hidden states [bs, seq, hidden]; project directly
                proj = self.model.output(logits)  # [bs, seq, vocab]
            else:
                proj = self.model.output(logits.unsqueeze(0))  # [1, seq, vocab]
        if self._model_parallel_group is not None:
            from distributed.tensor_parallel import gather_from_model_parallel_region

            proj = gather_from_model_parallel_region(proj, self._model_parallel_group)
        proj = F.log_softmax(proj.float(), dim=-1)
        return proj if hasattr(self.model, "compression_embeddings") else proj.squeeze(0)

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
                        logits = self.model.output(h).float()
                    nxt = int(torch.argmax(logits[-1]))
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

        final_output = torch.full((bsz, max_length), fill_value=self.pad_token_id, dtype=torch.long, device=self.device)
        final_output[:, :max_seqlen] = context
        for b in range(bsz):
            final_output[b, seqlens[b] : seqlens[b] + generation_length] = output[b]
        return final_output

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
            if getattr(self, "_shutdown_vllm_after_loglikelihood", False) and getattr(self, "_vllm_manager", None) is not None:
                self.shutdown_vllm_manager(
                    verbose=bool(getattr(self, "_shutdown_vllm_verbose", True)),
                    terminate_children=bool(getattr(self, "_shutdown_vllm_kill_children", False)),
                )
            return out
        if self._mode == "reconstruct_first" and hasattr(self.model, "compression_embeddings"):
            out = self._loglikelihood_tokens_reconstruct_first(requests, disable_tqdm)
            if getattr(self, "_shutdown_vllm_after_loglikelihood", False) and getattr(self, "_vllm_manager", None) is not None:
                self.shutdown_vllm_manager(
                    verbose=bool(getattr(self, "_shutdown_vllm_verbose", True)),
                    terminate_children=bool(getattr(self, "_shutdown_vllm_kill_children", False)),
                )
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

            for i, tgt in enumerate(tgt_batch):
                tgt_tensor = torch.tensor(tgt, device=self.device, dtype=torch.long)
                seq_logprobs = logprobs[i, -tgt_full_lens[i] :, :]
                cont_len = cont_lens[i]
                tail_logprobs = seq_logprobs[-cont_len:, :]
                token_logprobs = tail_logprobs.gather(-1, tgt_tensor[-cont_len:].unsqueeze(-1)).squeeze(-1)
                logprob = float(token_logprobs.sum().item())
                greedy = bool((tail_logprobs.argmax(dim=-1) == tgt_tensor[-cont_len:]).all().item())
                res.append((logprob, greedy))
        return res

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
        req_args = [req.args for req in requests]
        iterator = tqdm(range(0, len(req_args), self.batch_size), disable=disable_tqdm, desc=f"native generate ({self._mode})")
        for start in iterator:
            chunk = req_args[start : start + self.batch_size]
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
                    prompts = [self._format_chat(c, add_generation_prompt=True) for c, _ in chunk]
                else:
                    prompts = [c for c, _ in chunk]

                sampling_params = {
                    "max_tokens": max(g.get("max_generation_length", self.max_gen_toks) for g in gkwargs),
                    "temperature": gkwargs[0].get("temperature", self._temperature),
                    "top_p": gkwargs[0].get("top_p", 1.0),
                }
                outputs = self._vllm_manager.engine_wrapper.generate(prompts, sampling_params)
                for out, (_, gk) in zip(outputs, chunk):
                    text = out.outputs[0].text if out.outputs else ""
                    text = self._truncate_until(text, gk.get("until"))
                    results.append(text)
                continue

            if (
                self._mode == "decoder"
                and self._vllm_manager is not None
                and hasattr(self.model, "compression_embeddings")
            ):
                embeds = []
                gkwargs = []
                for c, gk in chunk:
                    if self._chat_use_template:
                        prompt_c = self._format_chat(c, add_generation_prompt=True)
                    else:
                        prompt_c = c
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
                    sampling_params = {
                        "max_tokens": max(gkwargs[i].get("max_generation_length", self.max_gen_toks) for i in valid_indices),
                        "temperature": gkwargs[valid_indices[0]].get("temperature", self._temperature),
                        "top_p": gkwargs[valid_indices[0]].get("top_p", 1.0),
                    }
                    batch_embeds = [embeds[i] for i in valid_indices]
                    outs = self._vllm_manager.generate_from_embeddings(batch_embeds, sampling_params=sampling_params)
                    for idx, out in zip(valid_indices, outs):
                        text = out.outputs[0].text if out.outputs else ""
                        outs_text[idx] = self._truncate_until(text, gkwargs[idx].get("until"))
                results.extend(outs_text)
                continue

            if (
                self._mode in {"compress_answer", "reconstruct_first", "vllm_decoding_with_compress"}
                and self._vllm_manager is not None
                and hasattr(self.model, "compression_embeddings")
            ):
                include_bor = self._mode == "reconstruct_first"
                if self._mode == "vllm_decoding_with_compress":
                    # new iterative vLLM decode with compression
                    # breakpoint()
                    for context_str, gk in chunk:

                        context_str = self._format_chat(context_str, add_generation_prompt=True)
                        text = self._generate_vllm_with_compress(
                            prompt=context_str,
                            max_gen_len=gk.get("max_generation_length", self.max_gen_toks),
                            temperature=gk.get("temperature", self._temperature),
                            top_p=gk.get("top_p", 1.0),
                            until=gk.get("until"),
                        )

                        results.append(text)
                    continue
                prompts = [self._format_chat(c, add_generation_prompt=True) for c, _ in chunk]
                gkwargs = [g for _, g in chunk]
                gen_lens = [g.get("max_generation_length", self.max_gen_toks) for g in gkwargs]
                embeds = self._build_compress_prompt_embeds_batch(prompts, gen_lens, include_bor)
                valid_indices = [i for i, e in enumerate(embeds) if e is not None]
                outs_text = [""] * len(chunk)
                if valid_indices:
                    sampling_params = {
                        "max_tokens": max(gkwargs[i].get("max_generation_length", self.max_gen_toks) for i in valid_indices),
                        "temperature": gkwargs[valid_indices[0]].get("temperature", self._temperature),
                        "top_p": gkwargs[valid_indices[0]].get("top_p", 1.0),
                    }
                    batch_embeds = [embeds[i] for i in valid_indices]
                    outs = self._vllm_manager.generate_from_embeddings(batch_embeds, sampling_params=sampling_params)
                    for idx, out in zip(valid_indices, outs):
                        text = out.outputs[0].text if out.outputs else ""
                        outs_text[idx] = self._truncate_until(text, gkwargs[idx].get("until"))
                results.extend(outs_text)
                continue

            # Fallback: torch paths, process one by one within chunk
            # generate for greedy decoding
            for context_str, gen_kwargs in chunk:
                until = gen_kwargs.get("until", None)
                max_gen_len = gen_kwargs.get("max_generation_length", self.max_gen_toks)
                temperature = gen_kwargs.get("temperature", self._temperature)
                top_p = gen_kwargs.get("top_p", 1.0)

                if self._mode in {"compress_answer", "reconstruct_first"} and hasattr(self.model, "compression_embeddings"):
                    text = self._generate_compress_answer(
                        prompt=context_str,
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                        until=until,
                        include_bor=(self._mode == "reconstruct_first"),
                    )
                    results.append(text)
                    continue

                context_str = self._format_chat(context_str, add_generation_prompt=True)
                ctx_tokens, _ = self.tok_batch_encode([context_str])
                # breakpoint()
                max_len = min(self.max_length, ctx_tokens.size(1) + max_gen_len)
                output_tokens = self._model_generate(
                    ctx_tokens.to(self.device),
                    max_len,
                    temperature=temperature,
                    top_p=top_p,
                )[0].tolist()
                gen_tokens = output_tokens[ctx_tokens.size(1) :]
                text = self.tok_decode(gen_tokens)

                if until:
                    stops = until if isinstance(until, list) else [until]
                    cutoff = len(text)
                    for s in stops:
                        idx = text.find(s)
                        if idx != -1:
                            cutoff = min(cutoff, idx)
                    text = text[:cutoff]
                results.append(text)
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
            output_tokens = self._model_generate(
                ctx_tokens.to(self.device),
                max_len,
                temperature=temperature,
                top_p=top_p,
            )[0].tolist()
            gen_tokens = output_tokens[ctx_tokens.size(1) :]
            text = self.tok_decode(gen_tokens)
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
    ) -> Tuple[List[Optional[torch.Tensor]], Optional[Dict[str, List[Any]]]]:
        """
        Batch build prompt embeddings for compress_answer/reconstruct_first with optional BOR.
        """
        num_comp = getattr(self.model.args, "num_compression_tokens", 0)
        embeds: List[Optional[torch.Tensor]] = [None] * len(prompts)
        if decoder_memory_layout not in {"single", "per_span"}:
            raise ValueError(f"Unsupported decoder_memory_layout: {decoder_memory_layout}")
        selected_spans_list: List[int] = []
        selected_flat_lens: List[int] = []

        # Fast path: no compression tokens
        if num_comp <= 0:
            # temp no BOQ index
                
            meta_n_spans = [0] * len(prompts)
            meta_flat_lens = [0] * len(prompts)
            for i, p in enumerate(prompts):
                tokens = prompt_tokens_override[i] if prompt_tokens_override is not None else self.tok_encode(p)
                if len(tokens) == 0:
                    continue
                prefix_tokens = []
                if self._add_boq_index:
                    prefix_tokens.append(BEGIN_OF_QUERY_INDEX)
                if decoder_include_prompt_tokens:
                    prefix_tokens = tokens[:] + ([BEGIN_OF_RECONSTRUCTION_INDEX] if include_bor else [])
                else:
                    # memory-only prefix degenerates to (optional) BOR
                    prefix_tokens = [BEGIN_OF_RECONSTRUCTION_INDEX] if include_bor else []
                if not prefix_tokens:
                    continue
                tok_tensor = torch.tensor(prefix_tokens, device=self.device, dtype=torch.long)
                embeds[i] = self.model.tok_embeddings(tok_tensor).to(dtype=self._dtype)
                meta_n_spans[i] = 0
                meta_flat_lens[i] = 0
            meta = {"n_spans": meta_n_spans, "flat_ctx_len": meta_flat_lens, "slots": [0] * len(prompts)}
            return (embeds, meta) if return_meta else embeds

        max_mem_span_len = getattr(self.model.args, "max_mem_span_len", self.max_length)
        placeholder_id = 0

        prompt_tokens_list: List[List[int]] = []
        total_comp_slots_list: List[int] = []
        enc_tokens: List[int] = []
        enc_mem_mask: List[bool] = []
        enc_lens: List[int] = []
        comp_offsets: List[int] = [0]
        positions_flat: List[int] = []
        model_max_len = getattr(self.model.args, "max_seq_len", self.max_length)

        for p, glen in zip(prompts, gen_lens):
            p_tokens = prompt_tokens_override[len(prompt_tokens_list)] if prompt_tokens_override is not None else self.tok_encode(p)
            prompt_tokens_list.append(p_tokens)
            ctx_spans = [p_tokens[i : i + max_mem_span_len] for i in range(0, len(p_tokens), max_mem_span_len)]
            if not ctx_spans:
                ctx_spans = [[]]

            bor_extra = 1 if include_bor else 0
            prompt_in_dec_len = len(p_tokens) if decoder_include_prompt_tokens else 0
            total_static = (2 if decoder_memory_layout == "single" else 0) + prompt_in_dec_len + bor_extra
            max_comp_tokens = max(0, self.max_length - total_static - glen)
            span_cost = num_comp if decoder_memory_layout == "single" else (num_comp + 2)
            max_chunks = max_comp_tokens // max(1, span_cost) if num_comp > 0 else 0
            if max_chunks <= 0:
                max_chunks = 1
            ctx_spans = ctx_spans[-max_chunks:]
            selected_spans_list.append(len(ctx_spans))
            selected_flat_lens.append(sum(len(sp) for sp in ctx_spans))

            slots = num_comp * len(ctx_spans)
            total_comp_slots_list.append(slots)
            comp_offsets.append(comp_offsets[-1] + slots)

            before_len = len(enc_tokens)
            for sp in ctx_spans:
                enc_tokens.extend(sp)
                enc_mem_mask.extend([False] * len(sp))
                enc_tokens.extend([placeholder_id] * num_comp)
                enc_mem_mask.extend([True] * num_comp)
            enc_lens.append(len(enc_tokens) - before_len)
            positions_flat.extend(range(enc_lens[-1]))

        if sum(enc_lens) == 0:
            compression_vectors = torch.empty(0, self.model.args.d_model, device=self.device, dtype=self._dtype)
            enc_cu = torch.tensor([0] * (len(prompts) + 1), device=self.device, dtype=torch.int32)
        else:
            enc_tokens_t = torch.tensor(enc_tokens, device=self.device, dtype=torch.long)
            enc_mem_mask_t = torch.tensor(enc_mem_mask, device=self.device, dtype=torch.bool)
            enc_cu = torch.tensor([0] + list(torch.tensor(enc_lens).cumsum(0).tolist()), device=self.device, dtype=torch.int32)
            compression_vectors = self._compress_tokens(enc_tokens_t, enc_cu, enc_mem_mask_t)

        for i, p_tokens in enumerate(prompt_tokens_list):
            slots = total_comp_slots_list[i]
            if decoder_memory_layout == "single":
                prefix = [BEGIN_OF_MEMORY_INDEX] + ([placeholder_id] * slots) + [END_OF_MEMORY_INDEX]
                comp_mask = [False] + ([True] * slots) + [False]
            else:
                n_spans = selected_spans_list[i]
                prefix = ([BEGIN_OF_MEMORY_INDEX] + ([placeholder_id] * num_comp) + [END_OF_MEMORY_INDEX]) * n_spans
                comp_mask = ([False] + ([True] * num_comp) + [False]) * n_spans
            if decoder_include_prompt_tokens:
                prefix = prefix + p_tokens
                comp_mask = comp_mask + ([False] * len(p_tokens))
                
            if self._add_boq_index and not not_add_boq_index:
                prefix = prefix + [BEGIN_OF_QUERY_INDEX]
                comp_mask = comp_mask + [False]
            
            # breakpoint()
            if include_bor:
                prefix.append(BEGIN_OF_RECONSTRUCTION_INDEX)
                comp_mask.append(False)
            # breakpoint() 

            prefix_t = torch.tensor(prefix, device=self.device, dtype=torch.long)
            comp_mask_t = torch.tensor(comp_mask, device=self.device, dtype=torch.bool)
            pe = self.model.tok_embeddings(prefix_t).to(dtype=self._dtype)
            if slots > 0 and compression_vectors.numel() > 0:
                pe[comp_mask_t] = compression_vectors[comp_offsets[i] : comp_offsets[i + 1]].to(dtype=self._dtype)
            embeds[i] = pe

        meta = {"n_spans": selected_spans_list, "flat_ctx_len": selected_flat_lens, "slots": total_comp_slots_list}
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

        # Decide how much of the context to compress (prefix) vs keep raw (suffix),
        # without ever truncating continuation tokens.
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
            cont_tokens_list: List[List[int]] = [cont for (_, _, cont) in chunk]
            suffix_tokens_list: List[List[int]] = [[] for _ in range(len(chunk))]
            
            if self._fill_decoder_prefix_embeds:
                prompt_tokens_list, suffix_tokens_list_t = zip(
                    *[_split_ctx_for_compression(p, len(c)) for p, c in zip(prompt_tokens_override, cont_tokens_list)]
                )
                prompt_tokens_override = list(prompt_tokens_list)
                suffix_tokens_list = list(suffix_tokens_list_t)


            # prompt_tokens, suffix_tokens = compute_needed_comp_slots(prompt_tokens_override[0])


            # Fast path: empty continuations
            chunk_results: List[Tuple[float, bool]] = [(float("-inf"), False)] * len(chunk)
            nonempty_idxs = [i for i, c in enumerate(cont_tokens_list) if len(c) > 0]
            for i in range(len(chunk)):
                if len(cont_tokens_list[i]) == 0:
                    chunk_results[i] = (0.0, True)

            if not nonempty_idxs:
                res.extend(chunk_results)
                continue

            dummy_prompts = [""] * len(chunk)
            gen_lens = [len(c) for c in cont_tokens_list]
            prefix_embeds_list, metainfo = self._build_compress_prompt_embeds_batch(
                dummy_prompts,
                gen_lens,
                include_bor=False,
                decoder_include_prompt_tokens=False,
                decoder_memory_layout="per_span",
                prompt_tokens_override=prompt_tokens_override,
                return_meta=True,
                # for do not add boq index for decoder prefix
                not_add_boq_index=True,
            )
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

            meta_n_spans = metainfo.get("n_spans", [1] * len(chunk)) if metainfo else [1] * len(chunk)

            seq_embeds: List[torch.Tensor] = []
            seq_targets: List[torch.Tensor] = []
            seq_loss_masks: List[torch.Tensor] = []
            seq_comp_masks: List[torch.Tensor] = []
            prefix_lens: List[int] = []
            cont_lens: List[int] = []
            cont_targets: List[torch.Tensor] = []
            valid_map: List[int] = []

            for i in nonempty_idxs:
                pe0 = prefix_embeds_list[i]
                if pe0 is None:
                    continue

                # Prefix consists of memory blocks + raw suffix tokens (kept uncompressed).
                suffix_e = suffix_embeds_list[i]
                pe = torch.cat([pe0, suffix_e], dim=0) if suffix_e.numel() else pe0

                cont = cont_tokens_list[i]
                cont_len = len(cont)
                if cont_len <= 0:
                    continue

                prefix_len = int(pe.shape[0])
                if prefix_len + cont_len > self.max_length:
                    # Cannot fit within decoder max length without truncating continuation.
                    continue

                cont_t = torch.tensor(cont, device=self.device, dtype=torch.long)
                with torch.autocast(device_type="cuda", dtype=self._dtype):
                    cont_e = self.model.tok_embeddings(cont_t).to(dtype=self._dtype)

                full_embeds = torch.cat([pe, cont_e], dim=0)
                total_len = int(full_embeds.shape[0])

                # Decoder token ids (for shifting/targets). Memory placeholders use a valid vocab id.
                n_spans = int(meta_n_spans[i]) if i < len(meta_n_spans) else 1
                prefix_ids = ([BEGIN_OF_MEMORY_INDEX] + ([placeholder_id] * num_comp) + [END_OF_MEMORY_INDEX]) * n_spans
                suffix_ids = list(suffix_tokens_list[i]) if suffix_tokens_list[i] else []
                token_ids = prefix_ids + suffix_ids + list(cont)
                if len(token_ids) != total_len:
                    continue

                targets_ids = token_ids[1:] + [int(self.eos_token_id)]
                targets_t = torch.tensor(targets_ids, device=self.device, dtype=torch.long)

                # Score only continuation: first continuation token is predicted at position prefix_len-1.
                score_start = prefix_len - 1
                score_end = score_start + cont_len
                if score_start < 0 or score_end > total_len:
                    continue
                loss_mask = torch.zeros(total_len, device=self.device, dtype=torch.bool)
                loss_mask[score_start:score_end] = True

                # compression_token_mask: True for placeholder slots in memory blocks, False elsewhere.
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
                res.extend(chunk_results)
                continue

            dec_lens = [int(t.shape[0]) for t in seq_embeds]
            dec_cu = torch.tensor([0] + list(torch.tensor(dec_lens).cumsum(0).tolist()), device=self.device, dtype=torch.int32)
            max_dec = max(dec_lens) if dec_lens else 0
            dec_positions = torch.cat([torch.arange(l, device=self.device, dtype=torch.int32) for l in dec_lens], dim=0)
            comp_mask_flat = torch.cat(seq_comp_masks, dim=0)
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
            # Avoid materializing [total_tokens, vocab] logits/logprobs which can OOM for large vocab.
            # Instead, project only the hidden states that correspond to continuation predictions.
            score_pos_chunks: List[torch.Tensor] = []
            score_tgt_chunks: List[torch.Tensor] = []
            score_ranges: List[Tuple[int, int, int]] = []  # (flat_start, flat_end, orig_idx)
            running = 0
            for j, orig_idx in enumerate(valid_map):
                start = int(dec_cu[j].item())
                dec_len = dec_lens[j]
                pref_len = prefix_lens[j]
                cont_len = cont_lens[j]
                rel_start = pref_len - 1
                rel_end = rel_start + cont_len
                if rel_start < 0 or rel_end > dec_len - 1 or cont_len <= 0:
                    score_ranges.append((running, running, orig_idx))
                    continue
                pos0 = start + rel_start
                pos1 = pos0 + cont_len
                score_pos_chunks.append(torch.arange(pos0, pos1, device=self.device, dtype=torch.long))
                score_tgt_chunks.append(cont_targets[j])
                score_ranges.append((running, running + cont_len, orig_idx))
                running += cont_len

            if running == 0:
                res.extend(chunk_results)
                continue

            score_pos = torch.cat(score_pos_chunks, dim=0)
            score_targets = torch.cat(score_tgt_chunks, dim=0)
            if score_targets.numel() != running or score_pos.numel() != running:
                res.extend(chunk_results)
                continue

            h_score = h.index_select(0, score_pos)
            del h
            del embeds_flat

            token_logprob = torch.empty(running, device=self.device, dtype=torch.float32)
            token_greedy_ok = torch.empty(running, device=self.device, dtype=torch.bool)

            rows_per_chunk = int(getattr(self.model.args, "cross_entropy_chunk", 8)) * 16
            rows_per_chunk = max(8, min(rows_per_chunk, 512))
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
                tgt_logits = logits_f.gather(-1, tgt_chunk.unsqueeze(-1)).squeeze(-1)
                token_logprob[off:off2] = (tgt_logits - lse)
                del logits_chunk, logits_f, lse, tgt_logits

            # Reduce back to per-request outputs.
            for (s0, s1, orig_idx) in score_ranges:
                if s1 <= s0:
                    continue
                lp_sum = float(token_logprob[s0:s1].sum().item())
                greedy = bool(token_greedy_ok[s0:s1].all().item())
                chunk_results[orig_idx] = (lp_sum, greedy)

            res.extend(chunk_results)
        # Safety: if we somehow produced fewer responses than requests, pad with -inf.
        if len(res) < len(requests):
            missing = len(requests) - len(res)
            res.extend([(float("-inf"), False)] * missing)
        return res



    @torch.no_grad()
    def _loglikelihood_tokens_reconstruct_first(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        """
        Reconstruct context via AE-style decoding (optionally batched vLLM prompt_embeds),
        then score the continuation tokens. Reconstruction is not scored; only continuation
        logprob is returned.
        """
        num_comp = getattr(self.model.args, "num_compression_tokens", 0)
        max_mem_span_len = getattr(self.model.args, "max_mem_span_len", self.max_length)
        placeholder_id = 0
        add_bor = bool(self._reconstruct_add_bor)

        max_bor = int(self._reconstruct_max_bor)
        def _postprocess_recon(
            tokens: List[int],
            max_len: int,
        ) -> Tuple[List[int], dict]:
            out: List[int] = []
            # bor_count = 0
            eor_count = 0
            stop_reason: Optional[str] = None
            for t in tokens:
                # if t == BEGIN_OF_RECONSTRUCTION_INDEX:
                #     bor_count >= max_bor:
                #         stop_reason = "bor_limit"
                #         break
                #     bor_count += 1
                if t == END_OF_RECONSTRUCTION_INDEX:
                    eor_count += 1
                    if eor_count >= max_bor:
                        stop_reason = "max_eor"
                        break

                out.append(int(t))
                if len(out) >= max_len:
                    stop_reason = "max_recon"
                    break
            if stop_reason is None:
                stop_reason = "max_recon"
            return out, {"stop_reason": stop_reason}

        res: List[Tuple[float, bool]] = []
        bs = self._vllm_reconstruct_batch_size if self._vllm_manager is not None else self.batch_size
        iterator = range(0, len(requests), bs)
        pbar = tqdm(iterator, disable=disable_tqdm or self.rank != 0, desc="native loglikelihood (reconstruct_first)")
        for batch_start in pbar:
            chunk = requests[batch_start : batch_start + bs]

            # Build encoder/decoder prefixes per sample
            enc_tokens_list: List[List[int]] = []
            enc_mem_masks: List[List[bool]] = []
            dec_prefix_list: List[List[int]] = []
            comp_mask_list: List[List[bool]] = []
            cont_list: List[List[int]] = []
            dec_prefix_lens: List[int] = []
            max_recon_lens: List[int] = []
            n_spans_list: List[int] = []
            n_slots_list: List[int] = []
            # stop_on_eor_list: List[bool] = []
            # enforce_bor_limit_list: List[bool] = []

            if self._vllm_manager is None:
                raise NotImplementedError("reconstruct_first loglikelihood requires vLLM in this harness.")

            # Use the shared prompt-embeds builder (with token override) so that:
            # - span selection respects decoder length (per-span BOM/slots/EOM layout)
            # - encoder length respects model_max_len / max_mem_span_len budget
            ctx_tokens_override = [context_enc for (_, context_enc, _) in chunk]
            cont_tokens_override = [continuation_enc for (_, _, continuation_enc) in chunk]
            # Reserve continuation + 1 EOT for scoring; reconstruction budget is computed later.
            gen_lens_for_slots = [len(c) + 1 for c in cont_tokens_override]
            dummy_prompts = [""] * len(chunk)
            prompt_embeds_for_recon, meta = self._build_compress_prompt_embeds_batch(
                dummy_prompts,
                gen_lens_for_slots,
                include_bor=add_bor,
                decoder_include_prompt_tokens=False,
                decoder_memory_layout="per_span",
                return_meta=True,
                prompt_tokens_override=ctx_tokens_override,
            )
            meta_n_spans = meta.get("n_spans", [1] * len(chunk))

            for i, (context_enc, continuation_enc) in enumerate(zip(ctx_tokens_override, cont_tokens_override)):
                # Split ctx into spans; select last n_spans chosen by the builder.
                ctx_spans_all = [context_enc[j : j + max_mem_span_len] for j in range(0, len(context_enc), max_mem_span_len)]
                if not ctx_spans_all:
                    ctx_spans_all = [[]]
                n_spans = int(meta_n_spans[i]) if i < len(meta_n_spans) else 1
                if n_spans <= 0:
                    n_spans = 1
                ctx_spans = ctx_spans_all[-n_spans:]

                # Decoder prefix: per-span BOM/slots/EOM blocks, then optional BOR.
                dec_prefix = ([BEGIN_OF_MEMORY_INDEX] + ([placeholder_id] * num_comp) + [END_OF_MEMORY_INDEX]) * len(ctx_spans)
                comp_prefix = ([False] + ([True] * num_comp) + [False]) * len(ctx_spans)
                if add_bor:
                    dec_prefix.append(BEGIN_OF_RECONSTRUCTION_INDEX)
                    comp_prefix.append(False)
                dec_prefix_len = len(dec_prefix)

                # Flatten spans for reconstruction-length budgeting and delimiter heuristics.
                flat_ctx = [t for sp in ctx_spans for t in sp]
                budget_recon = max(0, self.max_length - dec_prefix_len - len(continuation_enc) - 1)  # reserve one for EOT
                max_recon = min(len(flat_ctx), budget_recon)
                flat_ctx = flat_ctx[-max_recon:] if max_recon > 0 else []

                # If BOR/EOR show up in the reconstruction target, they are content delimiters and
                # should not be treated as early-stopping signals during reconstruction.


                # Encoder tokens with per-span slots
                enc_tokens, enc_mem_mask = [], []
                for sp in ctx_spans:
                    take_sp = sp[-max_mem_span_len:]
                    enc_tokens.extend(take_sp)
                    enc_mem_mask.extend([False] * len(take_sp))
                    if num_comp > 0:
                        enc_tokens.extend([placeholder_id] * num_comp)
                        enc_mem_mask.extend([True] * num_comp)

                cont_trim = continuation_enc  # defer truncation until after reconstruction

                enc_tokens_list.append(enc_tokens)
                enc_mem_masks.append(enc_mem_mask)
                dec_prefix_list.append(dec_prefix)
                comp_mask_list.append(comp_prefix)
                cont_list.append(cont_trim)
                dec_prefix_lens.append(dec_prefix_len)
                max_recon_lens.append(max_recon)
                n_spans_list.append(len(ctx_spans))
                n_slots_list.append(num_comp * len(ctx_spans))

            # Pack encoder
            enc_lens = [len(t) for t in enc_tokens_list]
            enc_cu = torch.tensor([0] + list(torch.tensor(enc_lens).cumsum(0).tolist()), device=self.device, dtype=torch.int32)
            max_enc = max(enc_lens)
            enc_positions = torch.cat([torch.arange(l, device=self.device, dtype=torch.int32) for l in enc_lens], dim=0)
            enc_tokens_flat = torch.tensor(sum(enc_tokens_list, []), device=self.device, dtype=torch.long)
            enc_mem_mask_flat = torch.tensor(sum(enc_mem_masks, []), device=self.device, dtype=torch.bool)
            enc_ctx = {
                "cu_seqlens_q": enc_cu,
                "cu_seqlens_k": enc_cu,
                "max_seqlen_q": max_enc,
                "max_seqlen_k": max_enc,
                "positions": enc_positions,
                "encoder_mem_mask": enc_mem_mask_flat,
            }

            # Prepare per-sample tensors
            dec_prefix_tensors = [torch.tensor(d, device=self.device, dtype=torch.long) for d in dec_prefix_list]
            comp_mask_tensors = [torch.tensor(c, device=self.device, dtype=torch.bool) for c in comp_mask_list]

            # Reconstruction: vLLM prompt_embeds batch if available, else per-sample greedy
            recon_tokens_list: List[List[int]] = [[] for _ in dec_prefix_list]
            recon_infos: List[dict] = [{"stop_reason": None} for _ in dec_prefix_list]
            
            breakpoint()
            if self._vllm_manager is not None:
                valid_indices = [
                    i
                    for i, (pe, mr) in enumerate(zip(prompt_embeds_for_recon, max_recon_lens))
                    if pe is not None and mr > 0
                ]
                if valid_indices:
                    sampling_params = {
                        "temperature": self._temperature,
                        "top_p": 1.0,
                        "max_tokens": max(max_recon_lens[i] for i in valid_indices),
                    }
                    # if all(stop_on_eor_list[i] for i in valid_indices):
                    #     sampling_params["stop_token_ids"] = [END_OF_RECONSTRUCTION_INDEX]
                    batch_embeds = [prompt_embeds_for_recon[i] for i in valid_indices]  # type: ignore[index]
                    outputs = self._vllm_manager.generate_from_embeddings(batch_embeds, sampling_params=sampling_params)
                    for sample_idx, out in zip(valid_indices, outputs):
                        if out.outputs:
                            token_ids = out.outputs[0].token_ids or []
                            recon_tokens, info = _postprocess_recon(
                                token_ids,
                                max_recon_lens[sample_idx],
                            )
                            recon_tokens_list[sample_idx] = recon_tokens
                            recon_infos[sample_idx] = info
            else:
                raise NotImplementedError("Greedy decoding is not supported for native harness.")
                for i, (dec_t, comp_t, max_recon) in enumerate(zip(dec_prefix_tensors, comp_mask_tensors, max_recon_lens)):
                    recon_tokens: List[int] = []
                    dec_tokens = dec_t
                    comp_mask = comp_t
                    bor_count = 0
                    stop_reason: Optional[str] = None
                    # stop_on_eor = stop_on_eor_list[i]
                    # enforce_bor_limit = enforce_bor_limit_list[i]
                    for _ in range(max_recon):
                        dec_cu = torch.tensor([0, dec_tokens.numel()], device=self.device, dtype=torch.int32)
                        dec_ctx = {
                            "cu_seqlens_q": dec_cu,
                            "cu_seqlens_k": dec_cu,
                            "max_seqlen_q": dec_tokens.numel(),
                            "max_seqlen_k": dec_tokens.numel(),
                            "positions": torch.arange(dec_tokens.numel(), device=self.device, dtype=torch.int32),
                            "compression_token_mask": comp_mask,
                        }
                        
                    # compute compression for this sample
                    comp_vec = self._compress_tokens(
                        enc_tokens_flat[enc_cu[i]:enc_cu[i+1]],
                        torch.tensor([0, enc_lens[i]], device=self.device, dtype=torch.int32),
                        torch.tensor(enc_mem_masks[i], device=self.device, dtype=torch.bool),
                    )
                    x_tokens = self.model.tok_embeddings(dec_tokens).to(self._dtype)
                    x_tokens[comp_mask] = comp_vec.to(x_tokens.dtype)
                    h_dec = x_tokens
                    for layer in self.model.layers:
                        h_dec = layer(h_dec, context=dec_ctx)
                        h_dec = self.model.norm(h_dec)
                        logits = self.model.output(h_dec).float()
                        next_token = int(torch.argmax(logits[-1]))
                        if stop_on_eor and next_token == END_OF_RECONSTRUCTION_INDEX:
                            stop_reason = "eor"
                            break
                        if next_token == BEGIN_OF_RECONSTRUCTION_INDEX:
                            # if enforce_bor_limit and bor_count >= max_bor:
                            #     stop_reason = "bor_limit"
                            #     break
                            # bor_count += 1
                            break
                    recon_tokens.append(next_token)
                    dec_tokens = torch.cat([dec_tokens, torch.tensor([next_token], device=self.device, dtype=torch.long)], dim=0)
                    comp_mask = torch.cat([comp_mask, torch.tensor([False], device=self.device, dtype=torch.bool)], dim=0)
                recon_tokens_list[i] = recon_tokens
                recon_infos[i] = {"bor_count": bor_count, "stop_reason": stop_reason or "max_recon"}


            if self._vllm_manager is not None:
                # self._vllm_manager.unload()
                # breakpoint()
                self.shutdown_vllm_manager()
                # breakpoint()
            # Build decoder with recon + continuation; defer scoring so we can batch PPL computation.
            dec_tokens_full: List[torch.Tensor] = []
            comp_masks_full: List[torch.Tensor] = []
            targets_full: List[torch.Tensor] = []
            cont_offsets: List[int] = []
            cont_lengths: List[int] = []
            cont_trim_list: List[List[int]] = []

            for i, (dec_t, comp_t, recon_tokens, cont_tokens) in enumerate(zip(dec_prefix_tensors, comp_mask_tensors, recon_tokens_list, cont_list)):
                dec_tokens = torch.cat([dec_t, torch.tensor(recon_tokens, device=self.device, dtype=torch.long)], dim=0)
                comp_mask = torch.cat([comp_t, torch.zeros(len(recon_tokens), device=self.device, dtype=torch.bool)], dim=0)

                cont_trim = cont_tokens[: max(0, self.max_length - dec_tokens.numel() - 1)]
                cont_tensor = torch.tensor(cont_trim, device=self.device, dtype=torch.long)
                dec_tokens = torch.cat([dec_tokens, cont_tensor], dim=0)
                comp_mask = torch.cat([comp_mask, torch.zeros(len(cont_trim), device=self.device, dtype=torch.bool)], dim=0)
                targets = torch.cat([dec_tokens[1:], torch.tensor([self.eot_token_id], device=self.device, dtype=torch.long)], dim=0)

                cont_start = dec_prefix_lens[i] + len(recon_tokens)

                dec_tokens_full.append(dec_tokens)
                comp_masks_full.append(comp_mask)
                targets_full.append(targets)
                cont_offsets.append(cont_start)
                cont_lengths.append(len(cont_trim))
                cont_trim_list.append(cont_trim)

            # Score continuation in small batches to reduce peak memory.
            chunk_results: List[Tuple[float, bool]] = [(float("-inf"), False)] * len(dec_tokens_full)
            debug_rows: List[dict] = []
            score_bs = max(1, self._ppl_batch_size)
            for score_start in range(0, len(dec_tokens_full), score_bs):
                idxs = list(range(score_start, min(score_start + score_bs, len(dec_tokens_full))))

                sub_enc_tokens = [enc_tokens_list[k] for k in idxs]
                sub_enc_masks = [enc_mem_masks[k] for k in idxs]
                enc_lens_sub = [len(t) for t in sub_enc_tokens]
                enc_cu_sub = torch.tensor(
                    [0] + list(torch.tensor(enc_lens_sub).cumsum(0).tolist()),
                    device=self.device,
                    dtype=torch.int32,
                )
                max_enc_sub = max(enc_lens_sub) if enc_lens_sub else 0
                enc_positions_sub = (
                    torch.cat([torch.arange(l, device=self.device, dtype=torch.int32) for l in enc_lens_sub], dim=0)
                    if enc_lens_sub
                    else torch.empty(0, device=self.device, dtype=torch.int32)
                )
                enc_tokens_flat_sub = (
                    torch.tensor(sum(sub_enc_tokens, []), device=self.device, dtype=torch.long)
                    if sub_enc_tokens
                    else torch.empty(0, device=self.device, dtype=torch.long)
                )
                enc_mem_mask_flat_sub = (
                    torch.tensor(sum(sub_enc_masks, []), device=self.device, dtype=torch.bool)
                    if sub_enc_masks
                    else torch.empty(0, device=self.device, dtype=torch.bool)
                )
                enc_ctx_sub = {
                    "cu_seqlens_q": enc_cu_sub,
                    "cu_seqlens_k": enc_cu_sub,
                    "max_seqlen_q": max_enc_sub,
                    "max_seqlen_k": max_enc_sub,
                    "positions": enc_positions_sub,
                    "encoder_mem_mask": enc_mem_mask_flat_sub,
                }

                sub_dec_tokens = [dec_tokens_full[k] for k in idxs]
                sub_comp_masks = [comp_masks_full[k] for k in idxs]
                dec_lens_sub = [int(t.numel()) for t in sub_dec_tokens]
                dec_cu_sub = torch.tensor(
                    [0] + list(torch.tensor(dec_lens_sub).cumsum(0).tolist()),
                    device=self.device,
                    dtype=torch.int32,
                )
                max_dec_sub = max(dec_lens_sub) if dec_lens_sub else 0
                dec_positions_sub = (
                    torch.cat([torch.arange(l, device=self.device, dtype=torch.int32) for l in dec_lens_sub], dim=0)
                    if dec_lens_sub
                    else torch.empty(0, device=self.device, dtype=torch.int32)
                )
                dec_tokens_flat_sub = (
                    torch.cat(sub_dec_tokens, dim=0) if sub_dec_tokens else torch.empty(0, device=self.device, dtype=torch.long)
                )
                comp_mask_flat_sub = (
                    torch.cat(sub_comp_masks, dim=0) if sub_comp_masks else torch.empty(0, device=self.device, dtype=torch.bool)
                )
                dec_ctx_sub = {
                    "cu_seqlens_q": dec_cu_sub,
                    "cu_seqlens_k": dec_cu_sub,
                    "max_seqlen_q": max_dec_sub,
                    "max_seqlen_k": max_dec_sub,
                    "positions": dec_positions_sub,
                    "compression_token_mask": comp_mask_flat_sub,
                }

                ctx = torch.autocast(device_type="cuda", dtype=self._dtype)
                with ctx:
                    if hasattr(self.model, "compression_embeddings"):
                        comp_vec = self._compress_tokens(enc_tokens_flat_sub, enc_cu_sub, enc_mem_mask_flat_sub)
                        x_tokens = self.model.tok_embeddings(dec_tokens_flat_sub).to(self._dtype)
                        if comp_mask_flat_sub.any():
                            x_tokens[comp_mask_flat_sub] = comp_vec.to(x_tokens.dtype)
                        h = x_tokens
                        for layer in self.model.layers:
                            h = layer(h, context=dec_ctx_sub)
                        h = self.model.norm(h)
                    else:
                        raise NotImplementedError("reconstruct_first loglikelihood only supports compression models in this harness.")

                # Build packed targets + loss masks to score only the continuation segment.
                sub_targets = [targets_full[k] for k in idxs]
                targets_flat_sub = torch.cat(sub_targets, dim=0) if sub_targets else torch.empty(0, device=self.device, dtype=torch.long)

                loss_masks_sub: List[torch.Tensor] = []
                for j, sample_idx in enumerate(idxs):
                    dec_len = int(dec_lens_sub[j])
                    cont_off = int(cont_offsets[sample_idx])
                    cont_len = int(cont_lengths[sample_idx])
                    mask = torch.zeros(dec_len, device=self.device, dtype=torch.bool)
                    if cont_len > 0 and cont_off > 0:
                        rel_start = cont_off - 1
                        rel_end = rel_start + cont_len
                        if 0 <= rel_start < rel_end <= dec_len:
                            mask[rel_start:rel_end] = True
                    loss_masks_sub.append(mask)
                loss_mask_flat_sub = torch.cat(loss_masks_sub, dim=0) if loss_masks_sub else torch.empty(0, device=self.device, dtype=torch.bool)

                # The fused CE kernel requires `n_rows % cross_entropy_chunk == 0`.
                ce_chunk = int(getattr(self.model.args, "cross_entropy_chunk", 1))
                pad = (-int(h.shape[0])) % ce_chunk if ce_chunk > 1 else 0
                if pad:
                    h = torch.cat([h, torch.zeros((pad, h.shape[1]), device=h.device, dtype=h.dtype)], dim=0)
                    targets_flat_sub = torch.cat(
                        [targets_flat_sub, torch.full((pad,), int(self.eot_token_id), device=self.device, dtype=torch.long)], dim=0
                    )
                    loss_mask_flat_sub = torch.cat(
                        [loss_mask_flat_sub, torch.zeros((pad,), device=self.device, dtype=torch.bool)], dim=0
                    )

                sample_ranges_sub: List[Tuple[int, int]] = [
                    (int(dec_cu_sub[j].item()), int(dec_cu_sub[j + 1].item())) for j in range(len(idxs))
                ]

                with torch.autocast(device_type="cuda", dtype=self._dtype):
                    loss_sum, n_tok, per_stats = self.model.compute_loss(
                        h,
                        targets_flat_sub,
                        loss_mask_flat_sub,
                        sample_ranges_sub,
                        process_group=self._model_parallel_group,
                    )

                per_losses = (per_stats or {}).get("losses", [])
                per_tokens = (per_stats or {}).get("tokens", [])

                # Best-effort greedy check (skip under tensor-parallel).
                greedy_flags: List[bool] = [False] * len(idxs)
                if self._model_parallel_group is None:
                    for j, sample_idx in enumerate(idxs):
                        cont_off = int(cont_offsets[sample_idx])
                        cont_len = int(cont_lengths[sample_idx])
                        if cont_len <= 0 or cont_off <= 0:
                            greedy_flags[j] = True
                            continue
                        rel_start = cont_off - 1
                        rel_end = rel_start + cont_len
                        start, end = sample_ranges_sub[j]
                        if rel_start < 0 or rel_end > (end - start):
                            greedy_flags[j] = False
                            continue
                        h_pred = h[start + rel_start : start + rel_end]
                        tgt_slice = targets_full[sample_idx][rel_start:rel_end]
                        ok = True
                        step = 64
                        for k in range(0, int(h_pred.shape[0]), step):
                            hk = h_pred[k : k + step]
                            tk = tgt_slice[k : k + step]
                            logits_k = self.model.output(hk).float()
                            pred_k = logits_k.argmax(dim=-1)
                            if not bool((pred_k == tk).all().item()):
                                ok = False
                                break
                        greedy_flags[j] = ok

                for j, sample_idx in enumerate(idxs):
                    if j >= len(per_losses):
                        continue
                    sample_loss = per_losses[j]
                    logprob = -float(sample_loss.item()) if hasattr(sample_loss, "item") else float("-inf")
                    chunk_results[sample_idx] = (logprob, bool(greedy_flags[j]))

            for i in range(len(dec_tokens_full)):
                logprob, greedy = chunk_results[i]
                recon_tokens = recon_tokens_list[i]
                info = recon_infos[i] if i < len(recon_infos) else {}
                debug_rows.append(
                    {
                        "request_index": batch_start + i,
                        "mode": "reconstruct_first",
                        "reconstruct_add_bor": add_bor,
                        "reconstruct_max_bor": max_bor,
                        "num_comp": num_comp,
                        "max_mem_span_len": max_mem_span_len,
                        "n_spans": n_spans_list[i],
                        "total_comp_slots": n_slots_list[i],
                        "dec_prefix_len": dec_prefix_lens[i],
                        "max_recon_len": max_recon_lens[i],
                        "recon_tokens_len": len(recon_tokens),
                        # "recon_tokens": recon_tokens,
                        "dec_tokens_full": dec_tokens_full[i].tolist(),
                        "recon_stop_reason": info.get("stop_reason"),
                        "recon_text_preview": self.tok_decode_w_special_tokens(recon_tokens) if recon_tokens else "",
                        "cont_len_scored": cont_lengths[i],
                        "cont_tokens_scored": cont_trim_list[i],
                        "logprob": logprob,
                        "greedy": greedy,
                    }
                )
            self._append_loglikelihood_debug_rows(debug_rows)

            res.extend(chunk_results)

        return res
