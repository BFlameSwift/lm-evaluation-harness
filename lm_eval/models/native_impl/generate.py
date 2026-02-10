"""
Generation implementations for the `native` model.

This module contains the heavy generation code paths that were previously
inline in `native_impl/model.py`.

The functions are written as `self`-style helpers so `NativeCausalLM` can simply
delegate to them.
"""

from __future__ import annotations

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
from lm_eval.models.native_doc_utils import get_doc_query_keys_by_task_name, split_doc_and_query

from .model import (
    _extract_niah_needles,
    _infer_niah_needle_type,
    _token_embed,
    resolve_generation_kwargs,
    resolve_max_gen_toks,
)

_split_doc_and_query = split_doc_and_query


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

    # If caller explicitly asked for vLLM in decoder path, require it here.
    if self._mode == "decoder" and (self._use_vllm_decoder or self._use_vllm_answer or self._use_vllm_reconstruct):
        self._ensure_vllm_manager(caller="generate_until(mode=decoder)")
        if self._vllm_manager is None:
            raise RuntimeError(
                "generate_until(mode=decoder) requested vLLM (use_vllm_*), but vLLM initialization failed. "
                "Check vLLM install/config or set vllm_max_model_len and vllm_gpu_memory_utilization."
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
            # vLLM has a hard max prompt length; for safety, tail-truncate if needed.
            vllm_max_len = int(getattr(self, "_vllm_max_model_len", 0) or 0)
            clip_notes: List[Optional[str]] = [None] * len(prompts)
            prompt_lens: List[int] = []
            prompt_caps: List[int] = []
            if vllm_max_len > 0:
                clipped_prompts: List[str] = []
                for i, prompt in enumerate(prompts):
                    try:
                        toks = self.tok_encode(prompt)
                    except Exception:
                        toks = []
                    reserve = 1
                    try:
                        reserve = max(1, int(_get_max_gen_tokens(gkwargs[i])))
                    except Exception:
                        reserve = 1
                    target = int(vllm_max_len) - int(reserve)
                    if target <= 0 and vllm_max_len > 1:
                        target = int(vllm_max_len) - 1
                    if target > 0 and len(toks) > target:
                        toks = toks[-target:]
                        clip_notes[i] = f"tail_truncated:{len(toks)}"
                        try:
                            prompt = self.tok_decode(toks)
                        except Exception:
                            pass
                    clipped_prompts.append(prompt)
                    prompt_lens.append(len(toks))
                    prompt_caps.append(max(0, int(vllm_max_len) - int(len(toks))))
                prompts = clipped_prompts

            resolved = resolve_generation_kwargs(
                gkwargs[0],
                default_temperature=self._temperature,
                default_top_p=1.0,
                override_do_sample=getattr(self, "_gen_do_sample_override", None),
                override_temperature=getattr(self, "_gen_temperature_override", None),
                override_top_p=getattr(self, "_gen_top_p_override", None),
            )
            max_req_toks = max(_get_max_gen_tokens(g) for g in gkwargs)
            if vllm_max_len > 0 and prompt_caps:
                max_req_toks = min(max_req_toks, max(1, min(prompt_caps)))
            sampling_params = {
                "max_tokens": max_req_toks,
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
                        "clip_note": clip_notes[i] if i < len(clip_notes) else None,
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
            vllm_max_len = int(getattr(self, "_vllm_max_model_len", 0) or 0)
            clip_notes: List[Optional[str]] = []
            prompt_caps: List[int] = []
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
                    clip_notes.append(None)
                    continue
                if vllm_max_len > 0:
                    reserve = 1
                    try:
                        reserve = max(1, int(_get_max_gen_tokens(gk)))
                    except Exception:
                        reserve = 1
                    target = int(vllm_max_len) - int(reserve)
                    if target <= 0 and vllm_max_len > 1:
                        target = int(vllm_max_len) - 1
                    if target > 0 and len(tokens) > target:
                        tokens = tokens[-target:]
                        clip_notes.append(f"tail_truncated:{len(tokens)}")
                    else:
                        clip_notes.append(None)
                    prompt_caps.append(max(0, int(vllm_max_len) - int(len(tokens))))
                else:
                    clip_notes.append(None)
                tok_tensor = torch.tensor(tokens, device=self.device, dtype=torch.long)
                embeds.append(_token_embed(self.model, tok_tensor).to(dtype=self._dtype))
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
                max_req_toks = max(_get_max_gen_tokens(gkwargs[i]) for i in valid_indices)
                if vllm_max_len > 0 and prompt_caps:
                    max_req_toks = min(max_req_toks, max(1, min(prompt_caps)))
                sampling_params = {
                    "max_tokens": max_req_toks,
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
                        "clip_note": clip_notes[i] if i < len(clip_notes) else None,
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
        embeds = _token_embed(self.model, tok_tensor).to(self._dtype)
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

    # Build prompt_embeds for a single prompt. Use `max_gen_len` to reserve some
    # decoder budget for generation so vLLM won't skip due to 0 remaining tokens.
    build_ret = self._build_compress_prompt_embeds_batch(
        [prompt],
        [max_gen_len],
        include_bor,
        decoder_include_prompt_tokens=False,
        return_meta=False,
    )
    if isinstance(build_ret, tuple):
        embeds_list = build_ret[0]
    else:
        embeds_list = build_ret
    prompt_embeds = embeds_list[0] if embeds_list else None
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
