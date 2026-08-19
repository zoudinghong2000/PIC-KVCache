import pytest
import torch

from cacheblend_vllm.kv_layout import gather_paged_kv, scatter_paged_kv


@pytest.mark.parametrize("layout", ["merged_first", "merged_block", "separate"])
def test_gather_and_scatter_round_trip(layout):
    block_size = 2
    base = torch.arange(2 * 3 * block_size * 2, dtype=torch.float32).reshape(2, 3, block_size, 2)
    if layout == "merged_first":
        paged = base.clone()
    elif layout == "merged_block":
        paged = base.permute(1, 0, 2, 3).contiguous()
    else:
        paged = [base[0].clone(), base[1].clone()]

    slots = torch.tensor([0, 3, 4])
    gathered = gather_paged_kv(paged, slots, block_size)
    assert torch.equal(gathered, base[:, [0, 1, 2], [0, 1, 0]])

    replacement = torch.full_like(gathered, -1)
    scatter_paged_kv(replacement, paged, slots, block_size)
    assert torch.equal(gather_paged_kv(paged, slots, block_size), replacement)


def test_rejects_unknown_layout():
    with pytest.raises(ValueError, match="unsupported"):
        gather_paged_kv(torch.empty(3, 4, 5), torch.tensor([0]), 2)
