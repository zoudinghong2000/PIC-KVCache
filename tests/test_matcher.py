from cacheblend_vllm.hashing import content_hash, rolling_hashes
from cacheblend_vllm.matcher import FingerprintRecord, TokenRangeMatcher
from cacheblend_vllm.types import SegmentId


def test_rolling_hash_matches_direct_hash_at_every_offset():
    tokens = [3, 1, 4, 1, 5, 9, 2]
    actual = list(rolling_hashes(tokens, 3))
    expected = [(start, content_hash(tokens[start : start + 3])) for start in range(5)]
    assert actual == expected


def test_matcher_finds_arbitrary_offsets_and_skips_overlaps():
    matcher = TokenRangeMatcher(chunk_size=4)
    matcher.register(matcher.make_record("m", 0, [10, 11, 12, 13]))
    matcher.register(matcher.make_record("m", 8, [20, 21, 22, 23]))

    query = [99, 10, 11, 12, 13, 10, 11, 12, 13, 0, 20, 21, 22, 23]
    matches = matcher.match("m", query, start_offset=1)
    assert [(value.target_start, value.source_start) for value in matches] == [
        (1, 0),
        (5, 0),
        (10, 8),
    ]


def test_hash_collision_is_verified_with_tokens():
    matcher = TokenRangeMatcher(chunk_size=2)
    fake_id = SegmentId("m", content_hash([7, 8]), 0, 2)
    matcher.register(FingerprintRecord(fake_id, (100, 101)))
    assert matcher.match("m", [7, 8]) == ()
