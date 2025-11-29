import os
import json
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model

from arch.model import ModelArgs, create_kv_cache
from arch.comp_mem import CompressedMemoryModel as Model
from config import DistributedArgs
from data.tokenizer import Tokenizer
from distributed import apply_tp
from torch.distributed.device_mesh import init_device_mesh
from data.ae_loader import (
    BEGIN_OF_MEMORY_INDEX,
    END_OF_MEMORY_INDEX,
    BEGIN_OF_RECONSTRUCTION_INDEX,
    END_OF_RECONSTRUCTION_INDEX,
)
from eval_func.model2safetensors import convert_checkpoint
from eval_func.vllm_runner import VLLMEngineWrapper, VLLMEngineConfig, VLLMDecoderManager
import sys


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
    if name not in {"decoder", "compress_answer", "reconstruct_then_ppl"}:
        raise ValueError(f"Unsupported native model mode: {name}")
    return name


def _default_distributed_args() -> DistributedArgs:
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return DistributedArgs(rank=rank, local_rank=local_rank, world_size=world_size)


def _build_device_mesh(world_size: int, mp_size: int):
    """Return None for single-process runs to avoid init_process_group when RANK is missing."""
    if world_size <= 1 and mp_size == 1:
        return None
    return init_device_mesh(
        "cuda",
        mesh_shape=(world_size // mp_size, mp_size),
        mesh_dim_names=["dp", "tp"],
    )


@register_model("native")
class NativeCausalLM(TemplateLM):
    """
    Minimal lm-evaluation-harness adapter for the native arch.Model checkpoints.

    Usage example:
    --model native --model_args checkpoint_dir=/path/to/ckpt,batch_size=4,max_seq_length=8192,mode=decoder

    mode:
      - decoder (default): vanilla causal scoring on decoder tokens only.
      - compress_answer: compress the context via encoder, then score the answer conditioned on memory.
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
        use_vllm_decoder: bool = False,
        use_vllm_reconstruct: bool = False,
        vllm_model_path: Optional[str] = None,
        vllm_max_model_len: Optional[int] = None,
        vllm_tensor_parallel: int = 1,
        vllm_gpu_memory_utilization: float = 0.5,
        vllm_output_root: Optional[str] = None,
        vllm_reconstruct_batch_size: int = 4000,
        ppl_batch_size: Optional[int] = 8,

    ) -> None:
        super().__init__()
        self._dtype = _str_to_dtype(dtype)
        self._device = torch.device("cuda")
        self._batch_size = int(batch_size) if isinstance(batch_size, (int, float)) or str(batch_size).isdigit() else 1
        self._mode = _parse_mode(mode)
        self._max_mem_span_len_override = max_mem_span_len
        self._use_vllm_reconstruct = use_vllm_reconstruct
        self._use_vllm_decoder = use_vllm_decoder
        self._vllm_manager = None
        self._vllm_output_root = vllm_output_root
        self._vllm_reconstruct_batch_size = max(1, int(vllm_reconstruct_batch_size))
        self._ppl_batch_size = max(1, int(ppl_batch_size)) if ppl_batch_size is not None else self._batch_size

        distributed_args = _default_distributed_args()
        torch.cuda.set_device(distributed_args.local_rank)
		
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
            model, tokenizer, _, device_mesh = self._load_checkpoint(checkpoint_dir, distributed_args, tokenizer_path)

        self.model = model.to(dtype=self._dtype, device=self._device)
        self.model.eval()
        self._tokenizer = tokenizer
        self._device_mesh = device_mesh
        self._model_parallel_group = device_mesh.get_group(1) if device_mesh is not None else None

        self._max_seq_length = max_seq_length or self.model.args.max_seq_len
        if self._max_mem_span_len_override is not None:
            # Respect override for compression-aware paths
            self.model.args.max_mem_span_len = self._max_mem_span_len_override

        # Optional vLLM init for reconstruction speedup or decoder fast path (decoder-only, prompt_embeds)
        need_vllm = self._use_vllm_reconstruct or self._use_vllm_decoder
        if need_vllm:
            if hasattr(self.model, "compression_embeddings") and self._use_vllm_decoder and self._mode == "decoder":
                # vLLM decoder-only path does not support compression model; fall back to torch
                print("WARNING: vLLM decoder path is disabled for compression models; using torch backend instead.", file=sys.stderr)
                need_vllm = False

        if need_vllm:
            # Prepare decoder-only safetensors if path not provided
            model_path = vllm_model_path
            if model_path is None:
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
                    )
            try:
                cfg = VLLMEngineConfig(
                    model_path=model_path,
                    tensor_parallel_size=vllm_tensor_parallel,
                    dtype=str(self._dtype).replace("torch.", ""),
                    max_model_len=vllm_max_model_len or self._max_seq_length,
                    enable_prompt_embeds=True,
                    tokenizer=tokenizer_path or getattr(self.model.args, "pretrain_model_dir", None),
                    additional_kwargs={"gpu_memory_utilization": vllm_gpu_memory_utilization},
                )
                self._vllm_manager = VLLMDecoderManager(
                    engine_wrapper=VLLMEngineWrapper(cfg),
                    tokenizer=self._tokenizer,
                )
            except Exception as e:
                print(f"WARNING: Failed to init vLLM, falling back to torch backend. Error: {e}", file=sys.stderr)
                self._vllm_manager = None

    # ---- Required TemplateLM properties ----
    @property
    def eot_token_id(self) -> int:
        return self._tokenizer.eos_id

    @property
    def max_length(self) -> int:
        return self._max_seq_length

    @property
    def max_gen_toks(self) -> int:
        return 1024

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
        return self._tokenizer.pad_id

    # ---- Tokenization helpers ----
    def tok_encode(self, string: str, add_special_tokens: Optional[bool] = None, **kwargs) -> List[int]:
        # native tokenizer already includes BOS/EOS control; keep minimal
        return self._tokenizer.encode(string, bos=False, eos=False)

    def tok_decode(self, tokens: List[int]) -> str:
        return self._tokenizer.decode(tokens)

    def tok_batch_encode(self, strings: List[str], left_truncate_len: Optional[int] = None, **kwargs):
        tokens = [self._tokenizer.encode(s, bos=False, eos=False) for s in strings]
        if left_truncate_len is not None:
            tokens = [t[-left_truncate_len:] for t in tokens]
        max_len = max(len(t) for t in tokens)
        padded = [t + [self.pad_token_id] * (max_len - len(t)) for t in tokens]
        tensor = torch.tensor(padded, device=self.device, dtype=torch.long)
        return tensor, tensor  # attention mask not used

    def _build_prompt_embeds(
        self,
        encoder_tokens: torch.Tensor,
        encoder_context: dict,
        decoder_prefix: torch.Tensor,
        compression_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode compression slots and return embeddings with slots filled."""
        with torch.inference_mode():
            compression_vectors = self.model.compress(
                encoder_tokens=encoder_tokens,
                encoder_context=encoder_context,
            )
        prompt_embeds = self.model.tok_embeddings(decoder_prefix.to(self.device))
        prompt_embeds = prompt_embeds.to(compression_vectors.dtype)
        prompt_embeds[compression_mask] = compression_vectors
        return prompt_embeds

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
                encoder_context = dict(context)
                encoder_context["encoder_mem_mask"] = torch.zeros_like(positions, dtype=torch.bool)
                decoder_context = dict(context)
                decoder_context["compression_token_mask"] = torch.zeros_like(positions, dtype=torch.bool)
                hidden = self.model(
                    encoder_tokens=inps.flatten(),
                    encoder_context=encoder_context,
                    decoder_tokens=inps.flatten(),
                    decoder_context=decoder_context,
                    last_hidden_only=True,
                )
                logits = hidden
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
                        logits = self.model(
                            encoder_tokens=enc,
                            encoder_context=enc_ctx,
                            decoder_tokens=enc,
                            decoder_context=dec_ctx,
                            last_hidden_only=False,
                        )
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
            return self._loglikelihood_tokens_compress_answer(requests, disable_tqdm, override_bs)
        if self._mode == "reconstruct_then_ppl" and hasattr(self.model, "compression_embeddings"):
            return self._loglikelihood_tokens_reconstruct_then_ppl(requests, disable_tqdm)
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
        for context_str, gen_kwargs in [req.args for req in requests]:
            until = gen_kwargs.get("until", None)
            max_gen_len = gen_kwargs.get("max_generation_length", self.max_gen_toks)
            temperature = gen_kwargs.get("temperature", 0)
            top_p = gen_kwargs.get("top_p", 1.0)

            # Fast path: decoder mode + vLLM + non-compression model
            if (
                self._mode == "decoder"
                and self._vllm_manager is not None
                and not hasattr(self.model, "compression_embeddings")
            ):
                text = self._generate_with_vllm_decoder(
                    prompt=context_str,
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                    until=until,
                )
                results.append(text)
                continue

            # compression-aware generation path for compress_answer / reconstruct_then_ppl
            if self._mode in {"compress_answer", "reconstruct_then_ppl"} and hasattr(self.model, "compression_embeddings"):
                text = self._generate_compress_answer(
                    prompt=context_str,
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                    until=until,
                )
                results.append(text)
                continue

            ctx_tokens, _ = self.tok_batch_encode([context_str])
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
        # Split prompt into spans for encoder compression
        ctx_spans = [prompt_tokens[i : i + max_mem_span_len] for i in range(0, len(prompt_tokens), max_mem_span_len)]
        if not ctx_spans:
            ctx_spans = [[]]

        # Budget: BOM + comp_slots + EOM + prompt + BOR + answer
        # Ensure we keep room for generation tokens
        max_comp_tokens = max(0, self.max_length - (len(prompt_tokens) + 3 + max_gen_len))
        max_chunks = max_comp_tokens // num_comp if num_comp > 0 else 0
        if max_chunks <= 0:
            max_chunks = 1
        ctx_spans = ctx_spans[-max_chunks:]

        total_comp_slots = num_comp * len(ctx_spans)
        # Build encoder packed tensors
        enc_tokens: List[int] = []
        enc_mem_mask: List[bool] = []
        for sp in ctx_spans:
            enc_tokens.extend(sp)
            enc_mem_mask.extend([False] * len(sp))
            enc_tokens.extend([placeholder_id] * num_comp)
            enc_mem_mask.extend([True] * num_comp)

        # Decoder prefix: BOM + slots + EOM + prompt + BOR
        dec_prefix = [BEGIN_OF_MEMORY_INDEX] + ([placeholder_id] * total_comp_slots) + [END_OF_MEMORY_INDEX] + prompt_tokens + [BEGIN_OF_RECONSTRUCTION_INDEX]
        comp_mask = [False] + ([True] * total_comp_slots) + [False] + ([False] * len(prompt_tokens)) + [False]

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

        num_comp = getattr(self.model.args, "num_compression_tokens", 0)
        # Use a valid vocab id for placeholder slots; pad_id is -100 in our tokenizer,
        # which would break embedding lookup.
        placeholder_id = 0
        # Use model-configured memory span if present, otherwise fall back to max_length.
        max_mem_span_len = getattr(self.model.args, "max_mem_span_len", self.max_length)
        # warning if not load from max_mem_span_len
        print(f"Using max_mem_span_len={max_mem_span_len} for compress_answer loglikelihood.")
        

        res: List[Tuple[float, bool]] = []
        iterator = range(0, len(requests), bs)
        pbar = tqdm(iterator, disable=disable_tqdm or self.rank != 0, desc="native loglikelihood (compress)")
        for batch_start in pbar:
            chunk = requests[batch_start : batch_start + bs]

            enc_tokens_list: List[List[int]] = []
            enc_mem_masks: List[List[bool]] = []
            dec_tokens_list: List[List[int]] = []
            comp_masks: List[List[bool]] = []
            targets_list: List[List[int]] = []
            ans_offsets: List[int] = []
            cont_lengths: List[int] = []

            for (_, context_enc, continuation_enc) in chunk:
                # Chunk context to respect max_mem_span_len; keep the most recent spans.
                ctx_spans = [context_enc[i : i + max_mem_span_len] for i in range(0, len(context_enc), max_mem_span_len)]
                # Determine how many chunks can fit given max_length budget.
                # Decoder will be: BOM + num_comp * n_chunks + EOM + continuation
                max_comp_tokens = max(0, self.max_length - (2 + len(continuation_enc)))
                max_chunks = max_comp_tokens // max(1, num_comp) if num_comp > 0 else 0
                if num_comp == 0:
                    # Fallback: no compression tokens, just trim context to fit
                    ctx_spans = [context_enc[-max_mem_span_len:]]
                    max_chunks = 0
                if max_chunks == 0 and num_comp > 0:
                    max_chunks = 1  # at least one span if compression is available
                ctx_spans = ctx_spans[-max_chunks:] if max_chunks > 0 else ctx_spans

                # Encoder: concatenate spans, each followed by its compression slots.
                enc_tokens, enc_mem_mask = [], []
                for span in ctx_spans:
                    enc_tokens.extend(span)
                    enc_mem_mask.extend([False] * len(span))
                    if num_comp > 0:
                        enc_tokens.extend([placeholder_id] * num_comp)
                        enc_mem_mask.extend([True] * num_comp)

                # Decoder: one BOM, then all compression slots, then EOM + answer.
                total_comp_slots = num_comp * len(ctx_spans) if num_comp > 0 else 0
                max_cont_len = max(0, self.max_length - (total_comp_slots + 2))
                cont_trim = continuation_enc[:max_cont_len]
                dec_tokens = [BEGIN_OF_MEMORY_INDEX] + ([placeholder_id] * total_comp_slots) + [END_OF_MEMORY_INDEX] + cont_trim
                comp_mask = [False] + ([True] * total_comp_slots) + [False] + ([False] * len(cont_trim))

                targets = dec_tokens[1:] + [self.eot_token_id]
                answer_start = total_comp_slots + 2  # skip BOM + all slots + EOM

                enc_tokens_list.append(enc_tokens)
                enc_mem_masks.append(enc_mem_mask)
                dec_tokens_list.append(dec_tokens)
                comp_masks.append(comp_mask)
                targets_list.append(targets)
                ans_offsets.append(answer_start)
                cont_lengths.append(len(cont_trim))

            # pack encoder sequences
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

            # pack decoder sequences
            dec_lens = [len(t) for t in dec_tokens_list]
            dec_cu = torch.tensor([0] + list(torch.tensor(dec_lens).cumsum(0).tolist()), device=self.device, dtype=torch.int32)
            max_dec = max(dec_lens)
            dec_positions = torch.cat([torch.arange(l, device=self.device, dtype=torch.int32) for l in dec_lens], dim=0)
            dec_tokens_flat = torch.tensor(sum(dec_tokens_list, []), device=self.device, dtype=torch.long)
            comp_mask_flat = torch.tensor(sum(comp_masks, []), device=self.device, dtype=torch.bool)
            dec_ctx = {
                "cu_seqlens_q": dec_cu,
                "cu_seqlens_k": dec_cu,
                "max_seqlen_q": max_dec,
                "max_seqlen_k": max_dec,
                "positions": dec_positions,
                "compression_token_mask": comp_mask_flat,
            }

            targets_flat = torch.tensor(sum(targets_list, []), device=self.device, dtype=torch.long)

            ctx = torch.autocast(device_type="cuda", dtype=self._dtype)
            with ctx:
                logits = self.model(
                    encoder_tokens=enc_tokens_flat,
                    encoder_context=enc_ctx,
                    decoder_tokens=dec_tokens_flat,
                    decoder_context=dec_ctx,
                    last_hidden_only=False,
                )
            logprobs = F.log_softmax(logits.float(), dim=-1)

            for i, (ans_off, cont_len) in enumerate(zip(ans_offsets, cont_lengths)):
                start, end = dec_cu[i].item(), dec_cu[i + 1].item()
                sample_logprobs = logprobs[start : end - 1, :]  # drop final padding target
                sample_targets = targets_flat[start : end - 1]
                answer_slice = slice(ans_off, ans_off + cont_len)
                if cont_len <= 0 or sample_logprobs.numel() == 0 or sample_targets.numel() == 0:
                    res.append((float("-inf"), False))
                    continue
                lp = sample_logprobs[answer_slice]
                tgt = sample_targets[answer_slice]
                if lp.numel() == 0 or tgt.numel() == 0:
                    res.append((float("-inf"), False))
                    continue
                token_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                logprob = float(token_lp.sum().item()) if token_lp.numel() > 0 else float("-inf")
                greedy = bool((lp.argmax(dim=-1) == tgt).all().item()) if token_lp.numel() > 0 else False
                res.append((logprob, greedy))
        # Safety: if we somehow produced fewer responses than requests, pad with -inf.
        if len(res) < len(requests):
            missing = len(requests) - len(res)
            res.extend([(float("-inf"), False)] * missing)
        return res

    @torch.no_grad()
    def _loglikelihood_tokens_reconstruct_then_ppl(
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

        res: List[Tuple[float, bool]] = []
        bs = self._vllm_reconstruct_batch_size if self._vllm_manager is not None else self.batch_size
        iterator = range(0, len(requests), bs)
        pbar = tqdm(iterator, disable=disable_tqdm or self.rank != 0, desc="native loglikelihood (reconstruct_then_ppl)")
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

            for (_, context_enc, continuation_enc) in chunk:
                # Split ctx into spans, keep as many as fit budget.
                ctx_spans = [context_enc[i : i + max_mem_span_len] for i in range(0, len(context_enc), max_mem_span_len)]
                if not ctx_spans:
                    ctx_spans = [[]]

                if num_comp > 0:
                    max_slots_tokens = max(0, self.max_length - len(continuation_enc) - 4)  # BOM/EOM/BOR + EOT
                    max_chunks = max_slots_tokens // num_comp if num_comp > 0 else 0
                    if max_chunks <= 0:
                        max_chunks = 1
                    ctx_spans = ctx_spans[-max_chunks:]
                else:
                    # no compression slots; keep last span only
                    ctx_spans = ctx_spans[-1:]

                total_comp_slots = num_comp * len(ctx_spans) if num_comp > 0 else 0
                dec_prefix = [BEGIN_OF_MEMORY_INDEX] + ([placeholder_id] * total_comp_slots) + [END_OF_MEMORY_INDEX] + [BEGIN_OF_RECONSTRUCTION_INDEX]
                comp_prefix = [False] + ([True] * total_comp_slots) + [False] + [False]
                dec_prefix_len = len(dec_prefix)

                # Flatten spans for encoder and reconstruction target
                flat_ctx = [t for sp in ctx_spans for t in sp]
                # Budget for reconstruction tokens
                budget_recon = max(0, self.max_length - dec_prefix_len - len(continuation_enc) - 1)  # reserve one for EOT
                max_recon = min(len(flat_ctx), budget_recon)
                flat_ctx = flat_ctx[-max_recon:] if max_recon > 0 else []

                # Encoder tokens with per-span slots
                enc_tokens, enc_mem_mask = [], []
                for sp in ctx_spans:
                    take_sp = sp[-max_mem_span_len:]
                    enc_tokens.extend(take_sp)
                    enc_mem_mask.extend([False] * len(take_sp))
                    if num_comp > 0:
                        enc_tokens.extend([placeholder_id] * num_comp)
                        enc_mem_mask.extend([True] * num_comp)

                cont_trim = continuation_enc[: max(0, self.max_length - dec_prefix_len - max_recon)]

                enc_tokens_list.append(enc_tokens)
                enc_mem_masks.append(enc_mem_mask)
                dec_prefix_list.append(dec_prefix)
                comp_mask_list.append(comp_prefix)
                cont_list.append(cont_trim)
                dec_prefix_lens.append(dec_prefix_len)
                max_recon_lens.append(max_recon)

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
            if self._vllm_manager is not None:
                # Batch compress once, then split per-sample to fill prompt embeds.
                with torch.inference_mode():
                    compression_vectors = self.model.compress(
                        encoder_tokens=enc_tokens_flat,
                        encoder_context=enc_ctx,
                    )
                comp_counts = [int(sum(mask)) for mask in enc_mem_masks]
                comp_offsets = [0]
                for c in comp_counts:
                    comp_offsets.append(comp_offsets[-1] + c)

                prompt_embeds = []
                for i in range(len(dec_prefix_list)):
                    comp_vec = compression_vectors[comp_offsets[i] : comp_offsets[i + 1]]
                    prompt = self.model.tok_embeddings(dec_prefix_tensors[i].to(self.device))
                    prompt = prompt.to(comp_vec.dtype)
                    if comp_vec.numel() > 0:
                        prompt[comp_mask_tensors[i]] = comp_vec
                    prompt_embeds.append(prompt)

                sampling_params = {"temperature": 0.0, "top_p": 1.0, "max_tokens": max(max_recon_lens) if max_recon_lens else 0}
                outputs = self._vllm_manager.generate_from_embeddings(prompt_embeds, sampling_params=sampling_params)
                for i, out in enumerate(outputs):
                    if out.outputs:
                        text = out.outputs[0].text
                        recon_tokens_list[i] = self.tok_encode(text)[: max_recon_lens[i]]
            else:
                for i, (dec_t, comp_t, max_recon) in enumerate(zip(dec_prefix_tensors, comp_mask_tensors, max_recon_lens)):
                    recon_tokens: List[int] = []
                    dec_tokens = dec_t
                    comp_mask = comp_t
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
                        logits = self.model(
                            encoder_tokens=enc_tokens_flat[enc_cu[i]:enc_cu[i+1]],
                            encoder_context={
                                "cu_seqlens_q": torch.tensor([0, enc_lens[i]], device=self.device, dtype=torch.int32),
                                "cu_seqlens_k": torch.tensor([0, enc_lens[i]], device=self.device, dtype=torch.int32),
                                "max_seqlen_q": enc_lens[i],
                                "max_seqlen_k": enc_lens[i],
                                "positions": torch.arange(enc_lens[i], device=self.device, dtype=torch.int32),
                                "encoder_mem_mask": torch.tensor(enc_mem_masks[i], device=self.device, dtype=torch.bool),
                            },
                            decoder_tokens=dec_tokens,
                            decoder_context=dec_ctx,
                            last_hidden_only=False,
                        )
                        next_token = int(torch.argmax(logits[-1]))
                        if next_token == END_OF_RECONSTRUCTION_INDEX:
                            break
                        recon_tokens.append(next_token)
                        dec_tokens = torch.cat([dec_tokens, torch.tensor([next_token], device=self.device, dtype=torch.long)], dim=0)
                        comp_mask = torch.cat([comp_mask, torch.tensor([False], device=self.device, dtype=torch.bool)], dim=0)
                    recon_tokens_list[i] = recon_tokens

            # Build decoder with recon + continuation; defer scoring so we can batch PPL computation.
            dec_tokens_full: List[torch.Tensor] = []
            comp_masks_full: List[torch.Tensor] = []
            targets_full: List[torch.Tensor] = []
            cont_offsets: List[int] = []
            cont_lengths: List[int] = []

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

            # Score continuation in small batches to reduce peak memory.
            score_bs = max(1, self._ppl_batch_size)
            for score_start in range(0, len(dec_tokens_full), score_bs):
                idxs = list(range(score_start, min(score_start + score_bs, len(dec_tokens_full))))

                sub_enc_tokens = [enc_tokens_list[i] for i in idxs]
                sub_enc_masks = [enc_mem_masks[i] for i in idxs]
                enc_lens_sub = [len(t) for t in sub_enc_tokens]
                enc_cu_sub = torch.tensor([0] + list(torch.tensor(enc_lens_sub).cumsum(0).tolist()), device=self.device, dtype=torch.int32)
                max_enc_sub = max(enc_lens_sub) if enc_lens_sub else 0
                enc_positions_sub = torch.cat([torch.arange(l, device=self.device, dtype=torch.int32) for l in enc_lens_sub], dim=0) if enc_lens_sub else torch.empty(0, device=self.device, dtype=torch.int32)
                enc_tokens_flat_sub = torch.tensor(sum(sub_enc_tokens, []), device=self.device, dtype=torch.long) if sub_enc_tokens else torch.empty(0, device=self.device, dtype=torch.long)
                enc_mem_mask_flat_sub = torch.tensor(sum(sub_enc_masks, []), device=self.device, dtype=torch.bool) if sub_enc_masks else torch.empty(0, device=self.device, dtype=torch.bool)
                # For scoring PPL on reconstruction only, erase encoder signal and compression masks.
                if enc_tokens_flat_sub.numel() > 0:
                    enc_tokens_flat_sub = torch.zeros_like(enc_tokens_flat_sub)
                if enc_mem_mask_flat_sub.numel() > 0:
                    enc_mem_mask_flat_sub = torch.zeros_like(enc_mem_mask_flat_sub)
                enc_ctx_sub = {
                    "cu_seqlens_q": enc_cu_sub,
                    "cu_seqlens_k": enc_cu_sub,
                    "max_seqlen_q": max_enc_sub,
                    "max_seqlen_k": max_enc_sub,
                    "positions": enc_positions_sub,
                    "encoder_mem_mask": enc_mem_mask_flat_sub,
                }

                sub_dec_tokens = [dec_tokens_full[i] for i in idxs]
                sub_comp_masks = [comp_masks_full[i] for i in idxs]
                dec_lens_sub = [t.numel() for t in sub_dec_tokens]
                dec_cu_sub = torch.tensor([0] + list(torch.tensor(dec_lens_sub).cumsum(0).tolist()), device=self.device, dtype=torch.int32)
                max_dec_sub = max(dec_lens_sub) if dec_lens_sub else 0
                dec_positions_sub = torch.cat([torch.arange(l, device=self.device, dtype=torch.int32) for l in dec_lens_sub], dim=0) if dec_lens_sub else torch.empty(0, device=self.device, dtype=torch.int32)
                dec_tokens_flat_sub = torch.cat(sub_dec_tokens, dim=0) if sub_dec_tokens else torch.empty(0, device=self.device, dtype=torch.long)
                comp_mask_flat_sub = torch.cat(sub_comp_masks, dim=0) if sub_comp_masks else torch.empty(0, device=self.device, dtype=torch.bool)
                if comp_mask_flat_sub.numel() > 0:
                    comp_mask_flat_sub = torch.zeros_like(comp_mask_flat_sub)
                dec_ctx_sub = {
                    "cu_seqlens_q": dec_cu_sub,
                    "cu_seqlens_k": dec_cu_sub,
                    "max_seqlen_q": max_dec_sub,
                    "max_seqlen_k": max_dec_sub,
                    "positions": dec_positions_sub,
                    "compression_token_mask": comp_mask_flat_sub,
                }

                logits = self.model(
                    encoder_tokens=enc_tokens_flat_sub,
                    encoder_context=enc_ctx_sub,
                    decoder_tokens=dec_tokens_flat_sub,
                    decoder_context=dec_ctx_sub,
                    last_hidden_only=False,
                )
                logprobs = F.log_softmax(logits.float(), dim=-1)

                for j, sample_idx in enumerate(idxs):
                    dec_start = int(dec_cu_sub[j].item())
                    cont_off = cont_offsets[sample_idx]
                    cont_len = cont_lengths[sample_idx]
                    tgt = targets_full[sample_idx]

                    if cont_len > 0 and cont_off > 0 and cont_off < tgt.numel() + 1:
                        lp_start = dec_start + cont_off - 1
                        lp_end = lp_start + cont_len
                        lp_slice = logprobs[lp_start:lp_end]
                        tgt_slice = tgt[cont_off - 1 : cont_off - 1 + cont_len]
                        if lp_slice.numel() == tgt_slice.numel() and lp_slice.numel() > 0:
                            token_lp = lp_slice.gather(-1, tgt_slice.unsqueeze(-1)).squeeze(-1)
                            logprob = float(token_lp.sum().item())
                            greedy = bool((lp_slice.argmax(dim=-1) == tgt_slice).all().item())
                        else:
                            logprob = float("-inf")
                            greedy = False
                    else:
                        logprob = float("-inf")
                        greedy = False
                    res.append((logprob, greedy))

        return res

    # ---- helpers ----
    @staticmethod
    def _load_checkpoint(checkpoint_dir: str, distributed_args: DistributedArgs, tokenizer_path: Optional[str] = None):
        with open(os.path.join(checkpoint_dir, "metadata.json")) as f:
            metadata = json.load(f)

        modelargs = ModelArgs(**metadata["modelargs"])
        device_mesh = _build_device_mesh(distributed_args.world_size, modelargs.model_parallel_size)
        model = Model(modelargs)
        if device_mesh is not None:
            apply_tp(model, device_mesh)
            tp_rank = device_mesh.get_local_rank("tp")
        else:
            tp_rank = 0
        device = torch.device(f"cuda:{distributed_args.local_rank}")

        model_path = os.path.join(checkpoint_dir, f"model_state_rank_{tp_rank}.pth")
        model_state = torch.load(model_path, map_location="cpu")
        # Be lenient to handle compressor weights or config drift
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        if unexpected:
            import warnings
            warnings.warn(f"Unexpected keys in checkpoint ignored: {list(unexpected)}")
        if missing:
            import warnings
            warnings.warn(f"Missing keys when loading checkpoint: {list(missing)}")
        model = model.to(torch.bfloat16).to(device)
        torch.set_autocast_gpu_dtype(torch.bfloat16)
        model.eval()

        from data.lm_loader import DataLoaderArgs  # lazy import to avoid cycles

        dataloader_args = DataLoaderArgs(**metadata["dataloader_args"])
        tokenizer = Tokenizer(tokenizer_path or dataloader_args.tokenizer_path)
        return model, tokenizer, metadata.get("updates", 0), device_mesh
