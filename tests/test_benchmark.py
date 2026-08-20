import json

import pytest

from benchmarks.cacheblend.analyze import (
    measured_trace_rows,
    sparse_selection_rows,
    summarize_trace,
    write_timeline,
)
from benchmarks.cacheblend.benchmark import (
    build_document,
    build_orders,
    percentile,
    summarize_records,
    token_digest,
)


class _CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]


def test_document_has_exact_deterministic_token_length():
    tokenizer = _CharacterTokenizer()
    first = build_document(tokenizer, document_id=3, target_tokens=1025)
    second = build_document(tokenizer, document_id=3, target_tokens=1025)
    other = build_document(tokenizer, document_id=4, target_tokens=1025)

    assert len(first) == 1025
    assert first == second
    assert token_digest(first) == token_digest(second)
    assert token_digest(first) != token_digest(other)

    quality = build_document(
        tokenizer,
        document_id=3,
        target_tokens=1025,
        include_retrieval_code=True,
    )
    assert [ord(character) for character in "ZXQ000003"] in [
        quality[index : index + len("ZXQ000003")]
        for index in range(len(quality) - len("ZXQ000003") + 1)
    ]


def test_blend_orders_are_reproducible_and_non_contiguous():
    orders = build_orders(4, 3, 4, seed=7)

    assert orders == build_orders(4, 3, 4, seed=7)
    assert all(len(set(order)) == 3 for order in orders)
    for index, order in enumerate(orders):
        contiguous = tuple((index + offset) % 4 for offset in range(3))
        assert order[0] == index % 4
        assert order != contiguous


def test_invalid_document_selection_is_rejected():
    with pytest.raises(ValueError, match="within the document set"):
        build_orders(2, 3, 1, seed=0)
    with pytest.raises(ValueError, match="must not exceed"):
        build_orders(4, 3, 5, seed=0)


def test_summary_groups_phases_and_computes_gain():
    records = [
        {
            "phase": "blend",
            "prompt_tokens": 100,
            "ttft_seconds": 2.0,
            "total_seconds": 2.1,
            "metrics_delta": {"external": 80},
            "expected_text": "ZXQ000001",
            "correct": True,
        },
        {
            "phase": "blend",
            "prompt_tokens": 100,
            "ttft_seconds": 4.0,
            "total_seconds": 4.1,
            "metrics_delta": {"external": 70},
            "expected_text": "ZXQ000002",
            "correct": False,
        },
        {
            "phase": "cold",
            "prompt_tokens": 100,
            "ttft_seconds": 9.0,
            "total_seconds": 9.1,
            "metrics_delta": {},
        },
    ]

    summary = summarize_records(records)

    assert summary["errors"] == 0
    assert summary["phases"]["blend"]["ttft_mean_seconds"] == 3.0
    assert summary["phases"]["blend"]["metrics_delta"]["external"] == 150
    assert summary["phases"]["blend"]["quality_accuracy"] == 0.5
    assert summary["gains"]["cold_over_blend"] == 3.0
    assert percentile([4.0, 1.0, 2.0, 3.0], 0.90) == 4.0


def test_trace_summary_and_request_timeline(tmp_path):
    rows = [
        {
            "timestamp_ns": 1_000_000,
            "event": "lookup_finished",
            "request_id": "r1",
            "duration_us": 1000,
            "benchmark_phase": "blend",
            "benchmark_name": "blend-0",
            "client_started_ns": 500_000,
            "ttft_seconds": 0.5,
        },
        {
            "timestamp_ns": 3_000_000,
            "event": "lookup_finished",
            "request_id": "r1",
            "duration_us": 3000,
            "benchmark_phase": "blend",
            "benchmark_name": "blend-0",
            "client_started_ns": 500_000,
            "ttft_seconds": 0.5,
        },
    ]

    summary = summarize_trace(rows)
    write_timeline(tmp_path, rows)
    timeline = json.loads((tmp_path / "pipeline_timeline.jsonl").read_text())

    assert summary == [
        {
            "phase": "blend",
            "event": "lookup_finished",
            "count": 2,
            "mean_ms": 2.0,
            "p90_ms": 3.0,
            "max_ms": 3.0,
        }
    ]
    assert timeline["phase"] == "blend"
    assert timeline["name"] == "blend-0"
    assert [event["relative_ms"] for event in timeline["events"]] == [0.0, 2.0]
    assert [event["client_relative_ms"] for event in timeline["events"]] == [0.5, 2.5]


def test_trace_join_accepts_vllm_internal_request_suffix():
    rows = [{"request_id": "cmpl-123-0-worker", "event": "blend_started"}]
    records = {
        "cmpl-123": {
            "phase": "blend",
            "name": "blend-0",
            "client_started_ns": 1,
            "ttft_seconds": 0.5,
        }
    }

    measured = measured_trace_rows(rows, records)

    assert measured[0]["benchmark_phase"] == "blend"
    assert measured[0]["benchmark_name"] == "blend-0"


def test_sparse_selection_report_preserves_pipeline_counts():
    rows = [
        {
            "event": "selection_finished",
            "benchmark_phase": "blend",
            "benchmark_name": "blend-0",
            "layer_id": 6,
            "active_tokens": 1024,
            "cached_tokens": 900,
            "gap_tokens": 124,
            "sparse_query_tokens": 124,
            "selected_tokens": 387,
            "tail_fallback": False,
        }
    ]

    assert sparse_selection_rows(rows) == [
        ["blend", "blend-0", "6", "1024", "900", "124", "124", "387", "False"]
    ]
