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



def _split_doc_and_query(active_lg_docs: List[Optional[dict]], active_tasks_names: List[Optional[str]],prompt_list: List[str] = [],doc_key: str = "context", question_key: str = "question") -> Dict[str, List[str]]:
    """
    Split a rendered prompt into:
        - doc_tokens: everything up to and including the closing `</text>` marker
        - query_tokens: everything after (question/choices/Answer:)

    We use tokenizer offset_mapping on the *full prompt string* to find the token boundary,
    because BPE boundaries mean that encoding `</text>` in isolation often does not match a
    subsequence of the full prompt token ids.
    """
    if not active_lg_docs or not active_tasks_names:
        return [], []
    
    ret_context_list = []
    ret_question_list = []
    ret_query_list = []
    # breakpoint()
    for doc, task_name, prompt in zip(active_lg_docs, active_tasks_names, prompt_list):
        if doc is None or task_name is None:
            continue
        if "longbench2" in task_name:
            context_text = doc[doc_key]
            question_text = doc[question_key]
            real_query_text = prompt.split("\n</text>\n\n")[1].strip()
            
            ret_question_list.append(question_text)
            ret_query_list.append(real_query_text)
            ret_context_list.append(context_text)

        else:
            raise ValueError(f"Unsupported task: {task_name}")

    return {
        "context_list": ret_context_list,
        "question_list": ret_question_list,
        "query_list": ret_query_list,
    }

def get_doc_query_keys_by_task_name(task_name: str) -> Dict[str, str]:
    if "longbench2" in task_name:
        return {
            "doc_key": "context",
            "question_key": "question",
        }
    else:
        print(f"Unsupported task: {task_name}")
        return {
            "doc_key": "context",
            "question_key": "question",
        }


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
        save_loglikelihood_debug: bool = True,
        loglikelihood_debug_path: Optional[str] = None,

        add_boq_index: bool = False,
        remove_eot_token: bool = True,
        fill_decoder_prefix_embeds: bool = False,
        
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
        self._reconstruct_add_bor = bool(reconstruct_add_bor)
        self._reconstruct_max_bor = max(0, int(reconstruct_max_bor))
        self._add_query_before_likelihood = bool(add_query_before_likelihood)
        self._save_loglikelihood_debug = bool(save_loglikelihood_debug)
        self._loglikelihood_debug_path = loglikelihood_debug_path
        self._last_loglikelihood_debug: List[dict] = []
        
        
        self._decoder_budget = max(max_seq_length, 8192)
        
        

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
                
        self._max_mem_span_len = self.model.args.max_mem_span_len if hasattr(self.model, "args") else self._max_mem_span_len_override
        self._num_compression_tokens = self.model.args.num_compression_tokens if hasattr(self.model, "args") else None

        # Optional vLLM init for reconstruction speedup or decoder fast path (decoder-only, prompt_embeds)
        need_vllm = self._use_vllm_reconstruct or self._use_vllm_decoder or self._use_vllm_answer
        
        
        if self._mode == "vllm_decoding_with_compress":
            need_vllm = True
        self._use_vllm = need_vllm
        self._vllm_model_path = vllm_model_path
        self._vllm_max_model_len = vllm_max_model_len
        self._vllm_tensor_parallel = vllm_tensor_parallel
        self._vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self._vllm_tokenizer_path = tokenizer_path
        self._vllm_checkpoint_dir = checkpoint_dir
        self._vllm_dtype = dtype
        # breakpoint()
        if need_vllm:
            self.init_vllm

        # if need_vllm:
    def init_vllm(self) -> None:
            # Prepare decoder-only safetensors if path not provided
            model_path = self._vllm_model_path
            # breakpoint()
            if model_path is None:
                # HF checkpoints: vLLM can load directly from checkpoint_dir (already a transformers directory).
                is_native_ckpt = self._vllm_checkpoint_dir is not None and os.path.exists(os.path.join(self._vllm_checkpoint_dir, "metadata.json"))
                if not is_native_ckpt:
                    model_path = self._vllm_checkpoint_dir
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
                            checkpoint_dir=self._vllm_checkpoint_dir,
                            output_dir=safedir,
                            tokenizer_path=self._vllm_tokenizer_path,
                            dtype=str(self._dtype).replace("torch.", ""),
                            additional_kwargs={
                                "max_position_embeddings": self._vllm_max_model_len,
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
        return budget or self._max_seq_length

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
        print(f"Saved loglikelihood debug rows to {out_path}")

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
    ) -> str:
        """
            Optionally wrap text with chat template. If assistant_text is None, will
            produce a prompt that expects model generation; otherwise returns a full
            conversation with assistant content included.
        """
        
        if not self._chat_use_template:
            return {"n_spans": 0, "available": 0,"decoder_prefix": user_text if assistant_text is None else user_text + "\n" + assistant_text}
            # return user_text if assistant_text is None else user_text + "\n" + assistant_text
        
        if self._chat_template_version == "v2":
            try:
                add_gen = self._chat_add_generation_prompt if add_generation_prompt is None else add_generation_prompt
                messages = [{"role": "user", "content": user_text}]
                if assistant_text is not None:
                    messages.append({"role": "assistant", "content": assistant_text})
                    add_gen = False if add_generation_prompt is None else add_generation_prompt
                return {"n_spans": 0, "available": 0,"decoder_prefix": self._tokenizer.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_gen)}
                # return self._tokenizer.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_gen)
            except Exception as e:
                print(f"WARNING: Failed to apply chat template, Error: {e}", file=sys.stderr)
                
                # fallback to raw text on any failure
                # return user_text if assistant_text is None else user_text + "\n" + assistant_text
            return {"n_spans": 0, "available": 0,"decoder_prefix": user_text if assistant_text is None else user_text + "\n" + assistant_text}
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
            spans = self._split_contexts_to_spans(contexts, self._max_mem_span_len)
            if max_spans is not None and max_spans > 0 and len(spans) > max_spans:
                if bool(getattr(self, "_verbose_compress", False)):
                    print(f"Truncating memory spans from {len(spans)} to {max_spans}")
                    print("budget:", self.decoder_budget, "span occupancy:", (2 + self._num_compression_tokens) * max_spans)
                spans = spans[-max_spans:]
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
            user_tokens = user_start + query_tokens + im_end
            comp_mask.extend([False] * len(user_tokens))
            
            ret_assistant_tokens = assistant_start 
            comp_mask.extend([False] * len(ret_assistant_tokens))
            
            
            total_comp_slots = n_spans * self._num_compression_tokens
            available = self.decoder_budget - 1 - n_spans * (self._num_compression_tokens + 2) - len(user_tokens) - len(ret_assistant_tokens)
            total_encoder_tokens = sum(len(sp) for sp in spans)
            comp_offsets = [i * self._num_compression_tokens for i in range(n_spans + 1)]
            
        # breakpoint()
            
        return {
            "n_spans": n_spans,
            "total_comp_slots": total_comp_slots,
            "total_encoder_tokens": total_encoder_tokens,
            "comp_offsets": comp_offsets,
            "available": available,
            "decoder_prefix": memory_tokens + user_tokens + ret_assistant_tokens,
            "comp_mask": comp_mask,
        }
            
            

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
            comp_mask_flat = torch.cat(comp_mask_list, dim=0).to(device=self.device, dtype=torch.bool)
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

        embeds_flat = torch.cat([e.to(device=self.device, dtype=self._dtype) for e in seq_embeds], dim=0)
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
                lse = torch.logsumexp(logits_f, dim=-1)
                tgt_logits = logits_f.gather(-1, tgt_chunk.unsqueeze(-1)).squeeze(-1)
                token_logprob[off:off2] = (tgt_logits - lse)

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
                ppl = float(math.exp(loss))
                total_ll += ll
                total_toks += nt
            per_sample.append({"ll": ll, "ll_norm": ll_norm, "loss": loss, "ppl": ppl, "tokens": nt, "greedy": greedy})

        if total_toks > 0:
            total_loss = -total_ll / float(total_toks)
            total_ppl = float(math.exp(total_loss))
        else:
            total_loss = None
            total_ppl = None

        return {"per_sample": per_sample, "total": {"ll": total_ll, "tokens": total_toks, "loss": total_loss, "ppl": total_ppl}}
    
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
                    prompts = [self._format_chat(c, add_generation_prompt=True)["decoder_prefix"] for c, _ in chunk]
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
                        prompt_c = self._format_chat(c, add_generation_prompt=True)["decoder_prefix"]
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

                        context_str = self._format_chat(context_str, add_generation_prompt=True)["decoder_prefix"]
                        text = self._generate_vllm_with_compress(
                            prompt=context_str,
                            max_gen_len=gk.get("max_generation_length", self.max_gen_toks),
                            temperature=gk.get("temperature", self._temperature),
                            top_p=gk.get("top_p", 1.0),
                            until=gk.get("until"),
                        )

                        results.append(text)
                    continue
                prompts = [self._format_chat(c, add_generation_prompt=True)["decoder_prefix"] for c, _ in chunk]
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

                context_str = self._format_chat(context_str, add_generation_prompt=True)["decoder_prefix"]  
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
        if chat_enabled and (context_list is None) != (query_list is None):
            raise ValueError("chat template requires both context_list and query_list when using chat contexts")
        if use_chat:
            if context_list is None or query_list is None:
                raise ValueError("chat template requires context_list and query_list in _build_compress_prompt_embeds_batch")
            if len(context_list) != len(prompts) or len(query_list) != len(prompts):
                raise ValueError(
                    f"context_list/query_list must match prompts length; got {len(context_list)}/{len(query_list)} vs {len(prompts)}"
                )
        decoder_budget = int(self.decoder_budget)

        # --------------------------
        # Fast path: no compression
        # --------------------------
        if num_comp <= 0:
            if use_chat:
                meta_n_spans: List[int] = []
                meta_flat_lens: List[int] = []
                meta_slots: List[int] = []
                for i in range(len(prompts)):
                    ret = self._format_chat(user_text=query_list[i], contexts=context_list[i])
                    prefix = ret["decoder_prefix"]
                    if not prefix:
                        embeds[i] = None
                        meta_n_spans.append(0)
                        meta_flat_lens.append(0)
                        meta_slots.append(0)
                        continue
                    tok_tensor = torch.tensor(prefix, device=self.device, dtype=torch.long)
                    embeds[i] = self.model.tok_embeddings(tok_tensor).to(dtype=self._dtype)
                    meta_n_spans.append(int(ret.get("n_spans", 0)))
                    meta_flat_lens.append(int(ret.get("total_encoder_tokens", 0)))
                    meta_slots.append(int(ret.get("total_comp_slots", 0)))
                meta = {"n_spans": meta_n_spans, "flat_ctx_len": meta_flat_lens, "slots": meta_slots}
                return (embeds, meta) if return_meta else embeds

            meta_n_spans = [0] * len(prompts)
            meta_flat_lens = [0] * len(prompts)
            meta_slots = [0] * len(prompts)
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
            meta = {"n_spans": meta_n_spans, "flat_ctx_len": meta_flat_lens, "slots": meta_slots}
            return (embeds, meta) if return_meta else embeds

        # --------------------------
        # Compression-aware path
        # --------------------------
        max_mem_span_len = int(getattr(self.model.args, "max_mem_span_len", decoder_budget))
        model_max_len = int(getattr(self.model.args, "max_seq_len", self._max_seq_length))
        if model_max_len <= num_comp:
            raise ValueError(
                f"Invalid config: model_max_len={model_max_len} <= num_compression_tokens={num_comp}. "
                "Encoder span cannot fit placeholders."
            )

        # Each encoder micro-batch is: span_tokens + num_comp placeholders.
        # Ensure positions never exceed model max len.
        span_len = max_mem_span_len
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

            enc_tokens_mb: List[torch.Tensor] = []
            enc_ctx_mb: List[Dict[str, torch.Tensor]] = []

            memory_start = self._tokenizer.encode("<|im_start|>memory\n", bos=False, eos=False)
            user_start = self._tokenizer.encode("<|im_start|>user\n", bos=False, eos=False)
            assistant_start = self._tokenizer.encode("<|im_start|>assistant\n", bos=False, eos=False)
            im_end = self._tokenizer.encode("<|im_end|>\n", bos=False, eos=False)
            span_cost = num_comp + 2

            for i, contexts in enumerate(context_list):
                spans = self._split_contexts_to_spans(contexts, span_len)
                query_tokens = self._tokenizer.encode(query_list[i], bos=False, eos=False)
                fixed_len = len(memory_start) + len(im_end) + len(user_start) + len(query_tokens) + len(im_end) + len(assistant_start)
                max_spans = (decoder_budget - fixed_len - int(gen_lens[i])) // max(1, span_cost)
                if max_spans <= 0:
                    max_spans = 1
                if max_spans < len(spans):
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
                if not hasattr(self.model, "compress_multi_batches"):
                    raise RuntimeError("Compression model missing compress_multi_batches; expected MassiveCompressedMemoryModel.")
                with torch.autocast(device_type="cuda", dtype=self._dtype):
                    compression_vectors = self.model.compress_multi_batches(enc_tokens_mb, enc_ctx_mb)
                if int(compression_vectors.shape[0]) != expected_total_slots:
                    raise RuntimeError(
                        f"compress_multi_batches returned {int(compression_vectors.shape[0])} vectors, "
                        f"expected {expected_total_slots} (= num_comp * total_spans)."
                    )
                if compression_vectors.dtype != self._dtype:
                    compression_vectors = compression_vectors.to(dtype=self._dtype)

            for i in range(len(prompts)):
                ret = self._format_chat(user_text=query_list[i], contexts=context_list[i], max_spans=max_spans_list[i])
                prefix = ret["decoder_prefix"]
                comp_mask = ret["comp_mask"]
                total_comp_slots = int(ret.get("total_comp_slots", 0))
                if total_comp_slots != total_comp_slots_list[i]:
                    raise RuntimeError(
                        f"Internal error: chat template slots ({total_comp_slots}) != encoder slots "
                        f"({total_comp_slots_list[i]}) for sample {i}."
                    )

                prefix_t = torch.tensor(prefix, device=self.device, dtype=torch.long)
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

            meta = {"n_spans": selected_spans_list, "flat_ctx_len": selected_flat_lens, "slots": total_comp_slots_list}
            return (embeds, meta) if return_meta else embeds

        prompt_tokens_list: List[List[int]] = []
        selected_spans_list: List[int] = []
        selected_flat_lens: List[int] = []
        total_comp_slots_list: List[int] = []
        comp_offsets: List[int] = [0]

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
            max_comp_tokens = max(0, int(decoder_budget) - int(total_static) - int(glen))
            
            max_chunks = max_comp_tokens // max(1, int(span_cost))
            if max_chunks <= 0:
                max_chunks = 1
            if max_chunks < len(ctx_spans):
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
            if not hasattr(self.model, "compress_multi_batches"):
                raise RuntimeError("Compression model missing compress_multi_batches; expected MassiveCompressedMemoryModel.")
            with torch.autocast(device_type="cuda", dtype=self._dtype):
                compression_vectors = self.model.compress_multi_batches(enc_tokens_mb, enc_ctx_mb)
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
            
            
            # fill decoder prefix embeds, only compress old context and keep new context uncompressed
            # TODO temp close this feature
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

            def _append_compress_debug_rows(
                prefix_embeds_list: Optional[List[Optional[torch.Tensor]]] = None,
                meta_n_spans: Optional[List[int]] = None,
            ) -> None:
                if not self._save_loglikelihood_debug or self._distributed_args.rank != 0:
                    return
                rows: List[dict] = []
                for i in range(len(chunk_results)):
                    cont_len = len(cont_tokens_list[i])
                    logprob, greedy = chunk_results[i]
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
                        "mode": "compress_answer",
                        "cont_len": cont_len,
                        "logprob": logprob,
                        "greedy": greedy,
                        "ppl": ppl,
                    }
                    if loss is not None:
                        row["loss"] = loss
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
            gen_lens = [len(c) for c in cont_tokens_list]
            if not getattr(self, "_chat_use_template", False):
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
                meta_group_n_spans = meta.get("n_spans", [1] * len(group_prompt_tokens)) if meta else [1] * len(group_prompt_tokens)
                prefix_embeds_list: List[Optional[torch.Tensor]] = [None] * len(chunk)
                meta_n_spans: List[int] = [1] * len(chunk)
                for gidx, idxs in enumerate(group_indices):
                    for i in idxs:
                        prefix_embeds_list[i] = group_prefix_embeds[gidx]
                        meta_n_spans[i] = int(meta_group_n_spans[gidx]) if gidx < len(meta_group_n_spans) else 1
            else:
                doc_and_context = self._get_doc_and_context(ctx_tokens_list=prompt_tokens_override)
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
                prefix_embeds_list = [None] * len(chunk)
                meta_n_spans = [1] * len(chunk)
                for gidx, idxs in enumerate(group_indices):
                    for i in idxs:
                        prefix_embeds_list[i] = group_prefix_embeds[gidx]
                        meta_n_spans[i] = int(meta_group_n_spans[gidx]) if gidx < len(meta_group_n_spans) else 1
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
                _append_compress_debug_rows(prefix_embeds_list, meta_n_spans)
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
                _append_compress_debug_rows(prefix_embeds_list, meta_n_spans)
                res.extend(chunk_results)
                continue

            score_pos = torch.cat(score_pos_chunks, dim=0)
            score_targets = torch.cat(score_tgt_chunks, dim=0)
            if score_targets.numel() != running or score_pos.numel() != running:
                _append_compress_debug_rows(prefix_embeds_list, meta_n_spans)
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

            _append_compress_debug_rows(prefix_embeds_list, meta_n_spans)
            res.extend(chunk_results)
        # Safety: if we somehow produced fewer responses than requests, pad with -inf.
        if len(res) < len(requests):
            missing = len(requests) - len(res)
            res.extend([(float("-inf"), False)] * missing)
        return res


        # Decide how much of the context to compress (prefix) vs keep raw (suffix),
        # without ever truncating continuation tokens.
    def _get_doc_and_context(self, ctx_tokens_list: List[List[int]]) -> Dict[str, List[str]]:
    
        doc_key, question_key = get_doc_query_keys_by_task_name(self._active_loglikelihood_task_names[0])["doc_key"], get_doc_query_keys_by_task_name(self._active_loglikelihood_task_names[0])["question_key"]
        
        # decoded original context
        prompt_list = [self._tokenizer.decode_w_special_tokens(ctx) for ctx in ctx_tokens_list]
            
        split_doc_and_query_results = _split_doc_and_query(active_lg_docs=self._active_loglikelihood_docs, active_tasks_names=self._active_loglikelihood_task_names,prompt_list=prompt_list,doc_key=doc_key, question_key=question_key)
        
        # len(context_list) == group_size * num_groups
        context_list, question_list, query_list = split_doc_and_query_results["context_list"], split_doc_and_query_results["question_list"], split_doc_and_query_results["query_list"]
        
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
        if self._vllm_manager is None:
            raise NotImplementedError("reconstruct_first loglikelihood requires vLLM in this harness.")

        num_comp = int(getattr(self.model.args, "num_compression_tokens", 0))
        max_mem_span_len = int(getattr(self.model.args, "max_mem_span_len", self.max_length))
        add_bor = bool(self._reconstruct_add_bor)
        max_bor = int(self._reconstruct_max_bor)
        add_query = bool(getattr(self, "_add_query_before_likelihood", False))
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
                    if max_bor > 0 and eor_count >= max_bor:
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
            chat_enabled = bool(getattr(self, "_chat_use_template", False))
            if chat_enabled:
                memory_start = self._tokenizer.encode("<|im_start|>memory\n", bos=False, eos=False)
                user_start = self._tokenizer.encode("<|im_start|>user\n", bos=False, eos=False)
                assistant_start = self._tokenizer.encode("<|im_start|>assistant\n", bos=False, eos=False)
                im_end = self._tokenizer.encode("<|im_end|>\n", bos=False, eos=False)
                span_cost = num_comp + 2
                query_tokens = group_query_tokens[gi]
                fixed_len = len(memory_start) + len(im_end) + len(user_start) + len(query_tokens) + len(im_end) + len(assistant_start)
                max_spans = (decoder_budget - fixed_len - int(group_suffix_lens[gi])) // max(1, span_cost)
                if max_spans <= 0:
                    max_spans = 1
                ret = self._format_chat(
                    user_text=group_query_list[gi],
                    contexts=group_doc_list[gi],
                    max_spans=max_spans,
                )
                prefix_tokens = ret["decoder_prefix"]
                if isinstance(prefix_tokens, str):
                    prefix_tokens = self._tokenizer.encode(prefix_tokens, bos=False, eos=False)
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
            cont_tokens_list: List[List[int]] = [cont for (_, _, cont) in chunk]

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

            group_doc_list: List[str] = [None for _ in range(n_groups)]
            group_query_list: List[str] = [None for _ in range(n_groups)]
            group_max_cont_lens: List[int] = [0 for _ in range(n_groups)]
            group_suffix_lens: List[int] = [0 for _ in range(n_groups)]
            group_indices: List[List[int]] = [[] for _ in range(n_groups)]
            
            doc_key, question_key = get_doc_query_keys_by_task_name(self._active_loglikelihood_task_names[0])["doc_key"], get_doc_query_keys_by_task_name(self._active_loglikelihood_task_names[0])["question_key"]
            
            split_doc_and_query_results = _split_doc_and_query(active_lg_docs=self._active_loglikelihood_docs, active_tasks_names=self._active_loglikelihood_task_names,prompt_list=[self._tokenizer.decode_w_special_tokens(ctx) for ctx in ctx_tokens_list],doc_key=doc_key, question_key=question_key)
            
            # len(context_list) == group_size * num_groups
            context_list, question_list, query_list = split_doc_and_query_results["context_list"], split_doc_and_query_results["question_list"], split_doc_and_query_results["query_list"]
            
            # context_list_group = [[] for _ in range(n_groups)]
            # question_list_group = [[] for _ in range(n_groups)]
            # query_list_group = [[] for _ in range(n_groups)]
            # ctx_tokens_list_group = [[] for _ in range(n_groups)]
            
            
            for group_idx,gk in enumerate(group_keys):
                idxs = groups[gk]
                # group_idx is the index of the group
                # idxs is the indices of the requests in the group
                rep = idxs[0]

                max_cont = max((len(cont_tokens_list[i]) for i in idxs), default=0)
                # group_max_cont_lens.append(max_cont)

                # group_indices.append(idxs)
                
                # context_list_group[group_idx] = context_list[rep]
                # question_list_group[group_idx] = question_list[rep]
                # query_list_group[group_idx] = query_list[rep]
                # ctx_tokens_list_group[group_idx] = ctx_tokens_list[rep]
                
                context = context_list[rep]
                question = question_list[rep]
                query = query_list[rep]
                ctx_tokens = ctx_tokens_list[rep]
                query_tokens = self._tokenizer.encode(query)
                
                group_doc_list[group_idx] = context
                group_query_list[group_idx] = query
                group_max_cont_lens[group_idx] = max_cont
                query_len = len(query_tokens)
                if self._chat_use_template:
                    group_suffix_lens[group_idx] = max_cont
                    if self._add_query_before_likelihood:
                        group_suffix_lens[group_idx] += query_len
                else:
                    group_suffix_lens[group_idx] = query_len + max_cont
                    if self._add_query_before_likelihood:
                        group_suffix_lens[group_idx] += query_len
                    
                group_indices[group_idx] = idxs
                
                
                #
            
            
            # for i, (all_prompt_tokens ,context, question, query) in enumerate(zip(ctx_tokens_list_group, context_list_group, question_list_group, query_list_group)):  
                
            #     group_idx = [i for i in range(len(group_keys)) if group_keys[i] == tuple(all_prompt_tokens)][0]
            #     if group_doc_tokens[group_idx] is not None:
            #         continue
                
                
            #     group_doc_tokens[group_idx] = self._tokenizer.encode(context)
            #     group_query_tokens[group_idx] = self._tokenizer.encode(query)
            #     # group_query_tokens[group_idx] = query
            #     group_max_cont_lens[group_idx] = max((len(cont_tokens_list[i]) for i in groups[group_keys[group_idx]]), default=0)
            #     group_suffix_lens[group_idx] = len(query) + group_max_cont_lens[group_idx]
                
            #     if self._add_query_before_likelihood:
            #         group_suffix_lens[group_idx] += len(query)

            group_query_tokens = [self._tokenizer.encode(query) for query in group_query_list]
            # breakpoint()
            

            # Build compression prefix prompt_embeds per unique context (once per group).
            dummy_prompts = [""] * n_groups
            prefix_embeds_list, meta = self._build_compress_prompt_embeds_batch(
                dummy_prompts,
                group_suffix_lens,
                include_bor=add_bor,
                decoder_include_prompt_tokens=False,
                decoder_memory_layout="per_span",
                prompt_tokens_override=[self._tokenizer.encode(doc) for doc in group_doc_list] if self._chat_use_template==True else ctx_tokens_list,
                return_meta=True,
                not_add_boq_index=False,
                context_list=group_doc_list,
                query_list=group_query_list,
            )
            
            meta_n_spans = (meta or {}).get("n_spans", [1] * n_groups)
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
                group_n_slots[gi] = num_comp * n_spans
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
                if qtoks and self._add_query_before_likelihood:
                    q = torch.tensor(qtoks, device=self.device, dtype=torch.long)
                    with torch.autocast(device_type="cuda", dtype=self._dtype):
                        qe = self.model.tok_embeddings(q).to(dtype=self._dtype)
                    group_query_embeds.append(qe)
                    
                # if not add_query_before_likelihood, just use the query tokens as the query suffix
                else:
                    group_query_embeds.append(torch.empty((0, d_model), device=self.device, dtype=self._dtype))

            # Build base embeds per group: prefix_embeds + recon_embeds (+ optional query suffix).
            group_base_embeds: List[Optional[torch.Tensor]] = [None] * n_groups
            group_base_lens: List[int] = [0] * n_groups
            group_dec_tokens: List[torch.Tensor] = [None] * n_groups
            
            
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
                group_dec_tokens[gi] = torch.cat([torch.tensor([0]*len(pe), device=self.device, dtype=torch.long), torch.tensor(recon, device=self.device, dtype=torch.long), torch.tensor(group_query_tokens[gi], device=self.device, dtype=torch.long)], dim=0)
                # print(group_dec_tokens[gi].shape)
                # print(group_base_embeds[gi].shape)
                # breakpoint()
                group_base_lens[gi] = int(base.shape[0])

            # Collect all (request_idx, group_idx) pairs that we need to score.
            score_pairs: List[Tuple[int, int]] = []
            for gi, idxs in enumerate(group_indices):
                base = group_base_embeds[gi]
                base_len = group_base_lens[gi]
                if base is None or base_len <= 0:
                    continue
                for req_idx in idxs:
                    cont = cont_tokens_list[req_idx]
                    if not cont:
                        continue
                    if base_len + len(cont) > decoder_budget:
                        # Cannot score without truncating continuation; return -inf for this option.
                        print("base_len + len(cont) > decoder_budget,",base_len + len(cont),">",decoder_budget)
                        print("we will skip this option")
                        continue
                    score_pairs.append((req_idx, gi))

            if not score_pairs:
                # Still write debug rows for reconstruction groups.
                debug_rows: List[dict] = []
                for gi in range(n_groups):
                    pre_tokens_full, _ = _build_pre_output_tokens(gi, group_query_tokens, group_suffix_lens, group_n_spans)
                    pre_tokens_no_prefix = list(group_recon_tokens[gi])
                    if add_query:
                        pre_tokens_no_prefix.extend(group_query_tokens[gi])
                    info = group_recon_infos[gi]
                    group_first_idx = group_indices[gi][0] if group_indices[gi] else 0
                    debug_rows.append(
                        {
                            "request_index": batch_start + group_first_idx,
                            "mode": "reconstruct_first",
                            "add_query_before_likelihood": add_query,
                            "reconstruct_add_bor": add_bor,
                            "reconstruct_max_bor": max_bor,
                            "num_comp": num_comp,
                            "max_mem_span_len": max_mem_span_len,
                            "n_spans": group_n_spans[gi],
                            "total_comp_slots": group_n_slots[gi],
                            "prefix_len": group_prefix_lens[gi],
                            "max_recon_len": group_max_recon_lens[gi],
                            "recon_tokens_len": len(group_recon_tokens[gi]),
                            "recon_stop_reason": info.get("stop_reason"),
                            "recon_text_preview": self.tok_decode_w_special_tokens(group_recon_tokens[gi][:512])
                            if group_recon_tokens[gi]
                            else "",
                            "recon_text": self.tok_decode_w_special_tokens(group_recon_tokens[gi]) if group_recon_tokens[gi] else "",
                            "pre_output_text": self.tok_decode_w_special_tokens(pre_tokens_full) if pre_tokens_full else "",
                            "pre_output_text_no_prefix": self.tok_decode_w_special_tokens(pre_tokens_no_prefix) if pre_tokens_no_prefix else "",
                            "query_len": len(group_query_tokens[gi]),
                        }
                    )
                    for req_idx in group_indices[gi]:
                        debug_rows.append(
                            {
                                "request_index": batch_start + req_idx,
                                "mode": "reconstruct_first_option",
                                "add_query_before_likelihood": add_query,
                                "cont_len": len(cont_tokens_list[req_idx]),
                                "logprob": chunk_results[req_idx][0],
                                "greedy": chunk_results[req_idx][1],
                            }
                        )
                self._append_loglikelihood_debug_rows(debug_rows)
                res.extend(chunk_results)
                continue


            score_bs = max(1, int(self._ppl_batch_size))
            debug_rows: List[dict] = []

            for score_start in range(0, len(score_pairs), score_bs):
                pairs = score_pairs[score_start : score_start + score_bs]
                
                print("pairs,",pairs,"score_start,",score_start,"score_bs,",score_bs)
                # breakpoint()
                
                seq_embeds: List[torch.Tensor] = []
                dec_lens: List[int] = []
                pref_lens: List[int] = []
                cont_lens: List[int] = []
                cont_targets: List[torch.Tensor] = []
                orig_req_idxs: List[int] = []
                group_for_req: List[int] = []
                
                dec_tokens: List[torch.Tensor] = []

                # Build packed batch embeddings.
                for req_idx, gi in pairs:
                    base = group_base_embeds[gi]
                    if base is None:
                        continue
                    cont = cont_tokens_list[req_idx]
                    cont_t = torch.tensor(cont, device=self.device, dtype=torch.long)
                    with torch.autocast(device_type="cuda", dtype=self._dtype):
                        cont_e = self.model.tok_embeddings(cont_t).to(dtype=self._dtype)
                    seq = torch.cat([base, cont_e], dim=0)
                    seq_embeds.append(seq)
                    

                    
                    dec_lens.append(int(seq.shape[0]))
                    pref_lens.append(int(group_base_lens[gi]))
                    cont_lens.append(int(len(cont)))
                    cont_targets.append(cont_t)
                    orig_req_idxs.append(req_idx)
                    group_for_req.append(gi)
                    
                    dec_token = torch.cat([group_dec_tokens[gi], cont_t], dim=0)
                    dec_tokens.append(dec_token)
                    
                    # breakpoint()
                    
                    

                if not seq_embeds:
                    continue

                dec_cu = torch.tensor([0] + list(torch.tensor(dec_lens).cumsum(0).tolist()), device=self.device, dtype=torch.int32)
                max_dec = max(dec_lens)
                dec_positions = torch.cat([torch.arange(l, device=self.device, dtype=torch.int32) for l in dec_lens], dim=0)
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

                # Build flattened positions/targets for continuation scoring.
                score_pos_chunks: List[torch.Tensor] = []
                score_tgt_chunks: List[torch.Tensor] = []
                score_ranges: List[Tuple[int, int, int]] = []
                running = 0
                for j, req_idx in enumerate(orig_req_idxs):
                    start = int(dec_cu[j].item())
                    pref_len = int(pref_lens[j])
                    cont_len = int(cont_lens[j])
                    if cont_len <= 0 or pref_len <= 0:
                        score_ranges.append((running, running, req_idx))
                        continue
                    rel_start = pref_len - 1
                    rel_end = rel_start + cont_len
                    if rel_start < 0 or rel_end > dec_lens[j]:
                        score_ranges.append((running, running, req_idx))
                        continue
                    pos0 = start + rel_start
                    pos1 = pos0 + cont_len
                    score_pos_chunks.append(torch.arange(pos0, pos1, device=self.device, dtype=torch.long))
                    score_tgt_chunks.append(cont_targets[j])
                    score_ranges.append((running, running + cont_len, req_idx))
                    running += cont_len
                    # breakpoint()

                if running == 0:
                    continue

                score_pos = torch.cat(score_pos_chunks, dim=0)
                score_targets = torch.cat(score_tgt_chunks, dim=0)
                h_score = h.index_select(0, score_pos)
                del h
                del embeds_flat

                token_logprob = torch.empty(running, device=self.device, dtype=torch.float32)
                token_greedy_ok = torch.empty(running, device=self.device, dtype=torch.bool)

                rows_per_chunk = int(getattr(self.model.args, "cross_entropy_chunk", 8)) * 16
                rows_per_chunk = max(16, min(rows_per_chunk, 512))
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

                for s0, s1, req_idx in score_ranges:
                    if s1 <= s0:
                        continue
                    lp_sum = float(token_logprob[s0:s1].sum().item())
                    greedy = bool(token_greedy_ok[s0:s1].all().item())
                    chunk_results[req_idx] = (lp_sum, greedy)

            # Debug rows: interleave group summary with its options.
            for gi in range(n_groups):
                pre_tokens_full, _ = _build_pre_output_tokens(gi, group_query_tokens, group_suffix_lens, group_n_spans)
                pre_tokens_no_prefix = list(group_recon_tokens[gi])
                if add_query:
                    pre_tokens_no_prefix.extend(group_query_tokens[gi])
                info = group_recon_infos[gi]
                group_first_idx = group_indices[gi][0] if group_indices[gi] else 0
                debug_rows.append(
                    {
                        "request_index": batch_start + group_first_idx,
                        "mode": "reconstruct_first",
                        "add_query_before_likelihood": add_query,
                        "reconstruct_add_bor": add_bor,
                        "reconstruct_max_bor": max_bor,
                        "num_comp": num_comp,
                        "max_mem_span_len": max_mem_span_len,
                        "n_spans": group_n_spans[gi],
                        "total_comp_slots": group_n_slots[gi],
                        "prefix_len": group_prefix_lens[gi],
                        "max_recon_len": group_max_recon_lens[gi],
                        "recon_tokens_len": len(group_recon_tokens[gi]),
                        "recon_stop_reason": info.get("stop_reason"),
                        "recon_text_preview": self.tok_decode_w_special_tokens(group_recon_tokens[gi])
                        if group_recon_tokens[gi]
                        else "",
                        "recon_text": self.tok_decode_w_special_tokens(group_recon_tokens[gi]) if group_recon_tokens[gi] else "",
                        "pre_output_text": self.tok_decode_w_special_tokens(pre_tokens_full) if pre_tokens_full else "",
                        "pre_output_text_no_prefix": self.tok_decode_w_special_tokens(pre_tokens_no_prefix) if pre_tokens_no_prefix else "",
                        "query_len": len(group_query_tokens[gi]),
                        "group_max_cont_len": group_max_cont_lens[gi],
                    }
                )
                for req_idx in group_indices[gi]:
                    debug_rows.append(
                        {
                            "request_index": batch_start + req_idx,
                            "mode": "reconstruct_first_option",
                            "add_query_before_likelihood": add_query,
                            "cont_len": len(cont_tokens_list[req_idx]),
                            "logprob": chunk_results[req_idx][0],
                            "greedy": chunk_results[req_idx][1],
                        }
                    )
            self._append_loglikelihood_debug_rows(debug_rows)

            res.extend(chunk_results)

        return res
