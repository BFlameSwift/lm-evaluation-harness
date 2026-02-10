"""Compression / reconstruction helpers for the `native` model.

This module contains the compression-aware helper methods used by:
- `compress_answer` (prompt_embeds-based scoring/generation)
- `reconstruct_first`
- NIAH/RULER long-context tasks (span budgeting, tail-span truncation)

The functions are written as `self`-style helpers so `NativeCausalLM` can simply
delegate to them.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from data.ae_loader import (
    BEGIN_OF_MEMORY_INDEX,
    END_OF_MEMORY_INDEX,
    BEGIN_OF_RECONSTRUCTION_INDEX,
    END_OF_RECONSTRUCTION_INDEX,
)
from data.retrieval_loader import BEGIN_OF_QUERY_INDEX

from .model import _coerce_int, _token_embed, filter_kwargs_for_callable


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
                embeds[i] = _token_embed(self.model, tok_tensor).to(dtype=self._dtype)
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
            embeds[i] = _token_embed(self.model, tok_tensor).to(dtype=self._dtype)
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

            pe = _token_embed(self.model, prefix_t).to(dtype=self._dtype)
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
            pe = _token_embed(self.model, prefix_t).to(dtype=self._dtype)

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
        
        pe = _token_embed(self.model, prefix_t).to(dtype=self._dtype)

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
