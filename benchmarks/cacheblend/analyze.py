"""Compare benchmark arms and summarize optional CacheBlend pipeline traces."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .benchmark import percentile


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must have NAME=PATH form")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not (path / "summary.json").is_file():
        raise argparse.ArgumentTypeError(f"missing {path / 'summary.json'}")
    return name, path


def format_number(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def phase_value(summary: dict[str, Any], phase: str, field: str) -> Any:
    return summary.get("phases", {}).get(phase, {}).get(field)


def build_arm_table(runs: list[tuple[str, Path]]) -> tuple[list[str], list[list[str]]]:
    headers = [
        "Arm",
        "Populate mean",
        "Blend mean",
        "Blend accuracy",
        "APC repeat",
        "Cold mean",
        "Cold/Blend",
        "Blend external span",
        "Errors",
    ]
    rows = []
    for name, path in runs:
        summary = load_json(path / "summary.json")
        blend_metrics = phase_value(summary, "blend", "metrics_delta") or {}
        rows.append(
            [
                name,
                format_number(phase_value(summary, "populate", "ttft_mean_seconds")),
                format_number(phase_value(summary, "blend", "ttft_mean_seconds")),
                format_number(phase_value(summary, "blend", "quality_accuracy"), 2),
                format_number(phase_value(summary, "apc_repeat", "ttft_mean_seconds")),
                format_number(phase_value(summary, "cold", "ttft_mean_seconds")),
                format_number(summary.get("gains", {}).get("cold_over_blend"), 2),
                str(
                    int(
                        blend_metrics.get("vllm:external_prefix_cache_hits_total", 0)
                    )
                ),
                str(summary.get("errors", 0)),
            ]
        )
    return headers, rows


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def load_trace_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trace_dir = path / "traces"
    if not trace_dir.is_dir():
        return rows
    for trace_file in sorted(trace_dir.glob("*.jsonl")):
        for line in trace_file.read_text().splitlines():
            if line:
                rows.append(json.loads(line))
    return sorted(rows, key=lambda row: int(row["timestamp_ns"]))


def load_request_index(path: Path) -> dict[str, dict[str, Any]]:
    records_path = path / "records.jsonl"
    if not records_path.is_file():
        return {}
    result = {}
    for line in records_path.read_text().splitlines():
        if not line:
            continue
        record = json.loads(line)
        response_id = record.get("response_id")
        if response_id:
            result[str(response_id)] = record
    return result


def measured_trace_rows(
    rows: list[dict[str, Any]],
    request_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join server request IDs to client phases and exclude warmup/process noise."""
    result = []
    for row in rows:
        request_id = row.get("request_id")
        internal_id = str(request_id) if request_id else ""
        record = request_index.get(internal_id)
        if record is None and internal_id:
            # vLLM appends a prompt index and engine-local suffix to the public
            # OpenAI response ID (for example ``cmpl-...-0-...``).
            record = request_index.get(internal_id.rsplit("-", 2)[0])
        if record is None:
            continue
        result.append(
            {
                **row,
                "benchmark_phase": record["phase"],
                "benchmark_name": record["name"],
                "client_started_ns": record["client_started_ns"],
                "ttft_seconds": record["ttft_seconds"],
            }
        )
    return result


def summarize_trace(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        phase = str(row.get("benchmark_phase", "-"))
        event = str(row["event"])
        key = (phase, event)
        counts[key] += 1
        duration = row.get("duration_us")
        if duration is not None:
            grouped[key].append(float(duration) / 1000)
    result = []
    for phase, event in sorted(counts):
        durations = grouped[(phase, event)]
        result.append(
            {
                "phase": phase,
                "event": event,
                "count": counts[(phase, event)],
                "mean_ms": statistics.fmean(durations) if durations else None,
                "p90_ms": percentile(durations, 0.90),
                "max_ms": max(durations) if durations else None,
            }
        )
    return result


def write_timeline(run_path: Path, rows: list[dict[str, Any]]) -> None:
    by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        request_id = row.get("request_id")
        if request_id:
            by_request[str(request_id)].append(row)
    output_path = run_path / "pipeline_timeline.jsonl"
    with output_path.open("w", encoding="utf-8") as output:
        for request_id, events in sorted(by_request.items()):
            origin = min(int(event["timestamp_ns"]) for event in events)
            client_started_ns = int(events[0]["client_started_ns"])
            output.write(
                json.dumps(
                    {
                        "request_id": request_id,
                        "phase": events[0]["benchmark_phase"],
                        "name": events[0]["benchmark_name"],
                        "ttft_seconds": events[0]["ttft_seconds"],
                        "events": [
                            {
                                **event,
                                "relative_ms": (int(event["timestamp_ns"]) - origin) / 1e6,
                                "client_relative_ms": (
                                    int(event["timestamp_ns"]) - client_started_ns
                                )
                                / 1e6,
                            }
                            for event in events
                        ],
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def lookup_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    result = []
    for row in rows:
        if row["event"] != "lookup_finished":
            continue
        result.append(
            [
                str(row["benchmark_phase"]),
                str(row["benchmark_name"]),
                str(row.get("outcome", "failed")),
                str(row.get("apc_prefix_tokens", 0)),
                str(row.get("hit_tokens", 0)),
                str(row.get("gap_tokens", 0)),
                str(row.get("allocation_tokens", 0)),
                format_number(float(row.get("duration_us", 0)) / 1000),
            ]
        )
    return result


def selection_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    result = []
    for row in rows:
        if row["event"] != "selection_compared":
            continue
        result.append(
            [
                str(row["benchmark_phase"]),
                str(row["benchmark_name"]),
                str(row.get("layer_id", "-")),
                str(row.get("deviation_tokens", 0)),
                str(row.get("sparse_q_tokens", 0)),
                str(row.get("overlap_tokens", 0)),
                format_number(row.get("jaccard"), 3),
                str(row.get("sparse_query_tokens", 0)),
                str(bool(row.get("tail_fallback", False))),
            ]
        )
    return result


def sparse_selection_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    result = []
    for row in rows:
        if row["event"] != "selection_finished":
            continue
        result.append(
            [
                str(row["benchmark_phase"]),
                str(row["benchmark_name"]),
                str(row.get("layer_id", "-")),
                str(row.get("active_tokens", 0)),
                str(row.get("cached_tokens", 0)),
                str(row.get("gap_tokens", 0)),
                str(row.get("sparse_query_tokens", 0)),
                str(row.get("selected_tokens", 0)),
                str(bool(row.get("tail_fallback", False))),
            ]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    arm_headers, arm_rows = build_arm_table(args.run)
    report = [
        "# CacheBlend benchmark report",
        "",
        (
            "TTFT values are seconds. `Cold/Blend` greater than one means the blend "
            "phase is faster than an equal-size uncached request."
        ),
        "",
        markdown_table(arm_headers, arm_rows),
    ]

    trace_payload: dict[str, Any] = {}
    for name, path in args.run:
        trace_rows = measured_trace_rows(load_trace_rows(path), load_request_index(path))
        if not trace_rows:
            continue
        stages = summarize_trace(trace_rows)
        comparisons = selection_rows(trace_rows)
        sparse_selections = sparse_selection_rows(trace_rows)
        trace_payload[name] = stages
        write_timeline(path, trace_rows)
        report.extend(
            [
                "",
                f"## Pipeline trace: {name}",
                "",
                (
                    "Most values are host scheduling/enqueue durations; "
                    "`save_store_wait_finished` includes synchronized Store completion. "
                    "Overlapping stages must not be added together."
                ),
                "",
                "### Lookup plans",
                "",
                markdown_table(
                    [
                        "Phase",
                        "Request",
                        "Outcome",
                        "APC",
                        "Exact hits",
                        "Gaps",
                        "Span",
                        "Lookup ms",
                    ],
                    lookup_rows(trace_rows),
                ),
            ]
        )
        if comparisons:
            report.extend(
                [
                    "",
                    "### Query-aware selection comparison",
                    "",
                    markdown_table(
                        [
                            "Phase",
                            "Request",
                            "Layer",
                            "K deviation",
                            "Sparse-Q",
                            "Overlap",
                            "Jaccard",
                            "Sparse queries",
                            "Tail fallback",
                        ],
                        comparisons,
                    ),
                ]
            )
        if sparse_selections:
            report.extend(
                [
                    "",
                    "### Sparse-Q selections",
                    "",
                    markdown_table(
                        [
                            "Phase",
                            "Request",
                            "Layer",
                            "Active",
                            "Cached",
                            "Gaps",
                            "Sparse queries",
                            "Selected",
                            "Tail fallback",
                        ],
                        sparse_selections,
                    ),
                ]
            )
        report.extend(
            [
                "",
                "### Stage events",
                "",
                markdown_table(
                    ["Phase", "Event", "Count", "Mean ms", "p90 ms", "Max ms"],
                    [
                        [
                            stage["phase"],
                            stage["event"],
                            str(stage["count"]),
                            format_number(stage["mean_ms"]),
                            format_number(stage["p90_ms"]),
                            format_number(stage["max_ms"]),
                        ]
                        for stage in stages
                    ],
                ),
            ]
        )

    rendered = "\n".join(report) + "\n"
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        args.output.with_suffix(".trace.json").write_text(
            json.dumps(trace_payload, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
