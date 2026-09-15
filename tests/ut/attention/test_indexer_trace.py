# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch

from vllm_ascend.attention.indexer_trace import (
    DSV4IndexerTrace,
    compute_selected_indexer_scores,
)


def trace_inputs():
    # Logical cache blocks [0, 1] map to physical blocks [1, 0].
    key_cache = torch.tensor(
        [
            [[[9, 9]], [[3, 4]]],
            [[[1, 2]], [[8, 8]]],
        ],
        dtype=torch.int8,
    )
    return {
        "topk_indices": torch.tensor([[[0, 3]]], dtype=torch.int32),
        "query": torch.tensor([[[1, 0], [0, 1]]], dtype=torch.int8),
        "key_cache": key_cache,
        "weights": torch.tensor([[1, 2]], dtype=torch.float16),
        "query_scale": torch.ones((1, 2), dtype=torch.float16),
        "key_scale_cache": torch.ones((2, 2, 1), dtype=torch.float16),
        "block_table": torch.tensor([[1, 0]], dtype=torch.int32),
    }


def test_compute_selected_indexer_scores_maps_paged_cache():
    scores, valid = compute_selected_indexer_scores(**trace_inputs())

    # score([1, 2]) = 1 * 1 + 2 * 2 = 5
    # score([3, 4]) = 1 * 3 + 2 * 4 = 11
    torch.testing.assert_close(scores, torch.tensor([[5.0, 11.0]]))
    assert valid.tolist() == [[True, True]]


def test_compute_selected_indexer_scores_masks_invalid_indices():
    inputs = trace_inputs()
    inputs["topk_indices"] = torch.tensor([[[0, -1]]], dtype=torch.int32)

    scores, valid = compute_selected_indexer_scores(**inputs)

    torch.testing.assert_close(scores, torch.tensor([[5.0, 0.0]]))
    assert valid.tolist() == [[True, False]]


def test_trace_writes_cutoff_statistics(tmp_path):
    tracer = DSV4IndexerTrace(
        {
            "enabled": True,
            "output_dir": str(tmp_path),
            "layers": [7],
        }
    )
    inputs = trace_inputs()
    tracer.record_decode(
        layer_name="model.layers.7.self_attn",
        **inputs,
        seq_lens=torch.tensor([9], dtype=torch.int32),
        compress_ratio=4,
    )

    record = json.loads((tmp_path / "indexer-trace-dp0-rank0.jsonl").read_text())
    assert record["data_parallel_rank"] == 0
    assert record["tensor_parallel_rank"] == 0
    assert record["context_len"] == 9
    assert record["compressed_len"] == 2
    assert record["valid_selected"] == 2
    assert record["has_full_topk"] is True
    assert record["score_source"] == "recomputed_selected_quantized_keys"
    assert record["cutoff"] == pytest.approx(5.0)
    assert record["top1"] == pytest.approx(11.0)
    assert record["mean_selected"] == pytest.approx(8.0)
    assert record["std_selected"] == pytest.approx(3.0)
