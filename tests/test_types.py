import pytest

from cacheblend_vllm.types import BlendPlan, BlendSegment, SegmentId


def segment(source: int, target: int, length: int = 4) -> BlendSegment:
    return BlendSegment(SegmentId("model", source + 7, source, length), target)


def test_blend_plan_round_trip_and_gaps():
    plan = BlendPlan.from_segments(4, [segment(0, 4), segment(8, 12)])
    assert plan.allocation_end == 16
    assert plan.hit_tokens == 8
    assert plan.gap_tokens == 4
    assert BlendPlan.from_dict(plan.to_dict()) == plan
    assert plan.passes_gate(4, 0.5, 0)
    assert not plan.passes_gate(4, 0.75, 0)


def test_blend_plan_rejects_overlaps():
    with pytest.raises(ValueError, match="overlap"):
        BlendPlan(0, 8, 8, (segment(0, 0), segment(4, 2)))
