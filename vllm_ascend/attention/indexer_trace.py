# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight DeepSeek-V4 indexer threshold tracing.

The optimized Ascend quant lightning-indexer operator returns indices only.
For diagnostics, this module gathers the already-selected keys and recomputes
their scores.  This keeps the extra work proportional to ``index_topk`` rather
than the full context length.
"""

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger

logger = init_logger(__name__)

_LAYER_NUMBER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def compute_selected_indexer_scores(
    *,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    weights: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale_cache: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute scores for indices selected by the A3 quant indexer.

    This helper deliberately supports one request at a time.  That makes the
    logical-to-physical paged-cache mapping unambiguous and is the mode used by
    the accompanying experiment script.

    Returns:
        A pair ``(scores, valid)`` with shapes ``[T, K]``. Invalid ``-1``
        indexer outputs have a score of zero and ``valid=False``.
    """
    if block_table.ndim != 2 or block_table.shape[0] != 1:
        raise ValueError(
            "Indexer tracing requires exactly one active request; start vLLM "
            "with --max-num-seqs 1 and do not send concurrent requests."
        )
    if query.ndim != 3:
        raise ValueError(f"Expected query shape [T, H, D], got {tuple(query.shape)}")

    if topk_indices.ndim == 3:
        if topk_indices.shape[1] != 1:
            raise ValueError(f"Expected one key head in top-k output, got {tuple(topk_indices.shape)}")
        logical_indices = topk_indices[:, 0, :]
    elif topk_indices.ndim == 2:
        logical_indices = topk_indices
    else:
        raise ValueError(f"Expected top-k indices with 2 or 3 dimensions, got {tuple(topk_indices.shape)}")

    if logical_indices.shape[0] != query.shape[0]:
        raise ValueError(f"Top-k/query token count mismatch: {logical_indices.shape[0]} != {query.shape[0]}")

    valid = logical_indices >= 0
    safe_indices = logical_indices.clamp_min(0).to(torch.long)
    block_size = key_cache.shape[1]
    logical_blocks = safe_indices // block_size
    block_offsets = safe_indices % block_size

    physical_blocks = block_table[0].to(torch.long)[logical_blocks]
    physical_slots = physical_blocks * block_size + block_offsets

    head_dim = key_cache.shape[-1]
    flat_key_cache = key_cache.reshape(-1, head_dim)
    selected_keys = flat_key_cache.index_select(0, physical_slots.reshape(-1)).reshape(*safe_indices.shape, head_dim)

    flat_key_scales = key_scale_cache.reshape(key_cache.shape[0] * block_size, -1)[:, 0]
    selected_key_scales = flat_key_scales.index_select(0, physical_slots.reshape(-1)).reshape_as(safe_indices)

    num_tokens, num_heads, _ = query.shape
    query_scales = query_scale.reshape(num_tokens, num_heads, -1)[..., 0]
    indexer_weights = weights.reshape(num_tokens, num_heads, -1)[..., 0]

    dequant_query = query.float() * query_scales.float().unsqueeze(-1)
    dequant_keys = selected_keys.float() * selected_key_scales.float().unsqueeze(-1)
    correlations = torch.matmul(dequant_query, dequant_keys.transpose(1, 2))
    scores = (correlations.relu() * indexer_weights.float().unsqueeze(-1)).sum(dim=1)
    return scores.masked_fill(~valid, 0.0), valid


class DSV4IndexerTrace:
    """Write per-decode, per-layer indexer threshold statistics as JSONL."""

    def __init__(self, config: dict[str, Any] | None, data_parallel_rank: int = 0):
        config = config or {}
        self.requested = bool(config.get("enabled", False))
        self.enabled = self.requested
        self.strict = bool(config.get("strict", True))
        self.output_dir = Path(config.get("output_dir", "/tmp/dsv4-indexer-trace"))
        self.layers = {int(layer) for layer in config.get("layers", [])}
        self.sample_every = max(1, int(config.get("sample_every", 1)))
        self._decode_calls = 0

        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.data_parallel_rank = data_parallel_rank
        if dist.is_available() and dist.is_initialized():
            try:
                self.tensor_parallel_rank = get_tensor_model_parallel_rank()
            except AssertionError:
                self.tensor_parallel_rank = self.rank
        else:
            self.tensor_parallel_rank = 0
        if self.tensor_parallel_rank != 0:
            self.enabled = False
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_vllm_config(cls, vllm_config) -> "DSV4IndexerTrace":
        additional_config = vllm_config.additional_config or {}
        return cls(
            additional_config.get("dsv4_indexer_trace"),
            data_parallel_rank=vllm_config.parallel_config.data_parallel_rank,
        )

    def _layer_is_enabled(self, layer_name: str) -> bool:
        if not self.layers:
            return True
        match = _LAYER_NUMBER_RE.search(layer_name)
        return match is not None and int(match.group(1)) in self.layers

    def record_decode(
        self,
        *,
        layer_name: str,
        topk_indices: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        weights: torch.Tensor,
        query_scale: torch.Tensor,
        key_scale_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        compress_ratio: int,
    ) -> None:
        if not self.enabled or not self._layer_is_enabled(layer_name):
            return

        self._decode_calls += 1
        if (self._decode_calls - 1) % self.sample_every != 0:
            return

        try:
            if query.shape[0] != 1:
                raise ValueError(
                    "Indexer tracing expects one query token per decode. Disable speculative decoding "
                    "for this experiment."
                )
            scores, valid = compute_selected_indexer_scores(
                topk_indices=topk_indices,
                query=query,
                key_cache=key_cache,
                weights=weights,
                query_scale=query_scale,
                key_scale_cache=key_scale_cache,
                block_table=block_table,
            )

            valid_count = valid.sum(dim=-1)
            valid_float = valid.float()
            denominator = valid_count.clamp_min(1).float()
            mean = (scores * valid_float).sum(dim=-1) / denominator
            variance = (((scores - mean.unsqueeze(-1)) ** 2) * valid_float).sum(dim=-1) / denominator
            cutoff = scores.masked_fill(~valid, float("inf")).amin(dim=-1)
            top1 = scores.masked_fill(~valid, float("-inf")).amax(dim=-1)
            cutoff = torch.where(valid_count > 0, cutoff, torch.full_like(cutoff, float("nan")))
            top1 = torch.where(valid_count > 0, top1, torch.full_like(top1, float("nan")))

            context_len = seq_lens.reshape(-1)[-1].float()
            stats = torch.stack(
                [
                    cutoff[0],
                    top1[0],
                    mean[0],
                    variance[0].sqrt(),
                    valid_count[0].float(),
                    context_len,
                ]
            ).cpu()
            cutoff_value, top1_value, mean_value, std_value, valid_count_value, context_len_value = stats.tolist()
            topk_width = topk_indices.shape[-1]
            record = {
                "wall_time_ns": time.time_ns(),
                "pid": os.getpid(),
                "rank": self.rank,
                "data_parallel_rank": self.data_parallel_rank,
                "tensor_parallel_rank": self.tensor_parallel_rank,
                "layer": layer_name,
                "decode_call": self._decode_calls,
                "context_len": int(context_len_value),
                "compressed_len": int(context_len_value) // compress_ratio,
                "topk": topk_width,
                "valid_selected": int(valid_count_value),
                "has_full_topk": int(valid_count_value) == topk_width,
                "score_source": "recomputed_selected_quantized_keys",
                "cutoff": cutoff_value,
                "top1": top1_value,
                "mean_selected": mean_value,
                "std_selected": std_value,
                "cutoff_over_top1": (
                    cutoff_value / top1_value if math.isfinite(top1_value) and abs(top1_value) > 1e-12 else None
                ),
            }
            output_path = self.output_dir / (f"indexer-trace-dp{self.data_parallel_rank}-rank{self.rank}.jsonl")
            with output_path.open("a", encoding="utf-8") as output_file:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("Failed to record DeepSeek-V4 indexer threshold for %s", layer_name)
            if self.strict:
                raise
            self.enabled = False
