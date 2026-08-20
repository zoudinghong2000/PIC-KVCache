from types import SimpleNamespace

import pytest
import torch

from cacheblend_vllm.ascend import (
    PagedBlendLoader,
    Qwen3BlendExecutor,
    relocate_key,
    validate_rope_support,
)
from cacheblend_vllm.config import CacheBlendConfig
from cacheblend_vllm.storage import LocalPinnedCPUStore
from cacheblend_vllm.types import BlendPlan, BlendSegment


class FakeRope:
    def __init__(self, neox: bool):
        self.head_size = 4
        self.rotary_dim = 4
        self.is_neox_style = neox

    def __call__(self, positions, query, key):
        def rotate(value):
            shaped = value.reshape(value.shape[0], -1, 4)
            angle = positions.float().reshape(-1, 1, 1) * 0.17
            cosine, sine = torch.cos(angle), torch.sin(angle)
            if self.is_neox_style:
                first, second = torch.chunk(shaped, 2, dim=-1)
                result = torch.cat(
                    (first * cosine - second * sine, second * cosine + first * sine),
                    dim=-1,
                )
            else:
                first, second = shaped[..., ::2], shaped[..., 1::2]
                result = torch.stack(
                    (first * cosine - second * sine, second * cosine + first * sine),
                    dim=-1,
                ).reshape_as(shaped)
            return result.reshape_as(value)

        return rotate(query), rotate(key)


def supported_rope_cache() -> torch.Tensor:
    positions = torch.arange(8, dtype=torch.float32).reshape(-1, 1)
    angles = positions * torch.tensor([[0.1, 0.2]])
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)


@pytest.mark.parametrize("neox", [False, True])
def test_relocate_key_matches_fresh_rope(neox):
    rope = FakeRope(neox)
    raw = torch.randn(3, 8)
    old_positions = torch.tensor([2, 3, 4])
    new_positions = torch.tensor([9, 10, 11])
    _, encoded_old = rope(old_positions, raw, raw)
    _, expected = rope(new_positions, raw, raw)
    actual = relocate_key(rope, old_positions, new_positions, encoded_old)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_rope_validation_accepts_only_full_unscaled_rotary_embedding():
    model = SimpleNamespace(
        config=SimpleNamespace(rope_scaling=None, partial_rotary_factor=1.0)
    )
    rope = SimpleNamespace(head_size=4, rotary_dim=4, cos_sin_cache=supported_rope_cache())
    validate_rope_support(model, rope)

    model.config.rope_scaling = {"type": "linear", "factor": 2.0}
    with pytest.raises(ValueError, match="rope_scaling"):
        validate_rope_support(model, rope)
    model.config.rope_scaling = None

    model.config.partial_rotary_factor = 0.5
    with pytest.raises(ValueError, match="partial_rotary_factor"):
        validate_rope_support(model, rope)
    model.config.partial_rotary_factor = 1.0

    rope.rotary_dim = 2
    with pytest.raises(ValueError, match="rotary_dim"):
        validate_rope_support(model, rope)


def test_rope_validation_rejects_non_unit_custom_cache():
    model = SimpleNamespace(config=SimpleNamespace(rope_scaling=None))
    rope = SimpleNamespace(
        head_size=4,
        rotary_dim=4,
        cos_sin_cache=torch.ones(8, 4),
    )
    with pytest.raises(ValueError, match="unit-norm"):
        validate_rope_support(model, rope)


def test_loader_preserves_apc_prefix_zeroes_gaps_and_scatter_suffix():
    config = CacheBlendConfig.from_extra_config(
        {
            "chunk_size": 2,
            "local_cpu_gb": 0.001,
            "event_pipeline": False,
            "strict_version_check": False,
        }
    )
    paged = torch.arange(2 * 2 * 4, dtype=torch.float32).reshape(2, 2, 4, 1, 1)
    original_prefix = paged[:, 0, :2].clone()
    store = LocalPinnedCPUStore(2, 4096, pin_memory=False)
    segment_id = store.begin_put("m", 4, [7, 8], num_layers=1)
    host_parent = torch.tensor(
        [
            [[[0.0]], [[70.0]], [[80.0]], [[0.0]]],
            [[[0.0]], [[700.0]], [[800.0]], [[0.0]]],
        ]
    )
    host = host_parent[:, 1:3]
    assert not host.is_contiguous()
    store.put_layer_host(segment_id, 0, host)
    segment = BlendSegment(segment_id, target_start=4)
    store.pin("r", [segment])
    plan = BlendPlan(apc_prefix_tokens=2, allocation_end=6, hit_tokens=2, segments=(segment,))
    loader = PagedBlendLoader(
        [paged],
        store,
        "r",
        plan,
        torch.arange(6),
        block_size=4,
        config=config,
    )
    rope = FakeRope(True)
    loader.prefetch(0, rope)
    contiguous = loader.get(0)
    assert torch.equal(contiguous[:, :2], paged[:, 0, :2])
    assert torch.count_nonzero(contiguous[:, 2:4]) == 0
    assert torch.equal(contiguous[:, 4:6], host)

    contiguous[:, 2:6].fill_(-1)
    loader.commit(0, contiguous)
    assert torch.equal(paged[:, 0, :2], original_prefix)
    assert torch.count_nonzero(paged[:, 0, 2:4] + 1) == 0
    assert torch.count_nonzero(paged[:, 1, :2] + 1) == 0


def test_selection_always_keeps_gaps_and_top_cached_scores():
    executor = object.__new__(Qwen3BlendExecutor)
    executor.tp_group = None
    executor.config = CacheBlendConfig.from_extra_config(
        {"recompute_ratios": [0.5], "strict_version_check": False}
    )
    selected = executor._select(
        torch.tensor([1.0, 10.0, 2.0, 9.0]),
        torch.tensor([True, False, True, False]),
    )
    assert selected.tolist() == [0, 1, 2]


def test_sparse_q_scores_match_chunked_causal_attention():
    executor = object.__new__(Qwen3BlendExecutor)
    executor.tp_group = None
    executor.block_size = 2
    executor.config = CacheBlendConfig.from_extra_config(
        {
            "selection_strategy": "sparse_q",
            "check_layers": [1],
            "query_score_chunk_size": 1,
            "overflow_blocks": 0,
            "tail_query_tokens": 2,
            "strict_version_check": False,
        }
    )
    query_one_head = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[1.0, 1.0]],
            [[-1.0, 1.0]],
        ]
    )
    query = torch.cat((query_one_head, query_one_head.flip(-1)), dim=1)
    key = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[1.0, 1.0]],
            [[-1.0, 0.0]],
            [[0.0, -1.0]],
            [[-1.0, 1.0]],
        ]
    )
    absolute = torch.tensor([2, 3, 4, 5])
    gaps = torch.tensor([False, True, False, True])

    scores, forced, stats = executor._sparse_q_scores(
        query,
        key,
        absolute,
        gaps,
        scale=0.5,
    )

    sparse_query = query[gaps].permute(1, 0, 2)
    sparse_positions = absolute[gaps]
    logits = torch.matmul(sparse_query, key[:, 0].T) * 0.5
    logits.masked_fill_(
        sparse_positions[None, :, None] < torch.arange(key.shape[0])[None, None, :],
        torch.finfo(logits.dtype).min,
    )
    expected = torch.softmax(logits, dim=-1).sum((0, 1))[absolute]
    assert torch.allclose(scores, expected)
    assert torch.equal(forced, gaps)
    assert stats["sparse_query_tokens"] == 2
    assert not stats["tail_fallback"]


def test_sparse_q_forces_overflow_and_tail_fallback():
    executor = object.__new__(Qwen3BlendExecutor)
    executor.tp_group = None
    executor.block_size = 2
    executor.config = CacheBlendConfig.from_extra_config(
        {
            "chunk_size": 2,
            "selection_strategy": "sparse_q",
            "check_layers": [1],
            "query_score_chunk_size": 2,
            "overflow_blocks": 1,
            "tail_query_tokens": 2,
            "strict_version_check": False,
        }
    )
    query = torch.ones(6, 1, 2)
    key = torch.ones(6, 1, 2)
    absolute = torch.arange(6)

    _, forced, stats = executor._sparse_q_scores(
        query,
        key,
        absolute,
        torch.tensor([False, False, True, False, False, False]),
        scale=1.0,
    )
    assert forced.tolist() == [True, True, True, True, True, False]
    assert not stats["tail_fallback"]

    _, forced, stats = executor._sparse_q_scores(
        query,
        key,
        absolute,
        torch.zeros(6, dtype=torch.bool),
        scale=1.0,
    )
    assert forced.tolist() == [False, False, False, False, True, True]
    assert stats["sparse_query_tokens"] == 2
    assert stats["tail_fallback"]


def test_forced_selection_does_not_consume_query_topk_budget():
    executor = object.__new__(Qwen3BlendExecutor)
    executor.tp_group = None
    executor.config = CacheBlendConfig.from_extra_config(
        {"recompute_ratios": [0.5], "strict_version_check": False}
    )
    selected = executor._select(
        torch.tensor([0.0, 1.0, 10.0, 9.0]),
        torch.tensor([True, False, False, False]),
        forced_mask=torch.tensor([False, True, False, False]),
    )
    assert selected.tolist() == [0, 1, 2]
