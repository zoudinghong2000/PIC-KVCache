"""Deterministic multi-document CacheBlend benchmark.

The populate phase stores every document independently.  Blend requests then
reorder and combine those documents so native APC cannot represent the reuse,
while CacheBlend can match their fixed-size interior ranges at new positions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

METRICS = (
    "vllm:external_prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:prompt_tokens_total",
    "vllm:time_to_first_token_seconds_count",
    "vllm:time_to_first_token_seconds_sum",
)


@dataclass(frozen=True, slots=True)
class RequestSpec:
    phase: str
    name: str
    prompt_token_ids: list[int]
    document_ids: tuple[int, ...]
    expected_text: str | None = None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    phases: dict[str, dict[str, Any]] = {}
    for phase in dict.fromkeys(record["phase"] for record in records):
        selected = [record for record in records if record["phase"] == phase]
        successful = [record for record in selected if "error" not in record]
        ttfts = [float(record["ttft_seconds"]) for record in successful]
        totals = [float(record["total_seconds"]) for record in successful]
        metrics: dict[str, float] = {}
        for record in successful:
            for name, value in record.get("metrics_delta", {}).items():
                metrics[name] = metrics.get(name, 0.0) + float(value)
        phases[phase] = {
            "requests": len(selected),
            "errors": len(selected) - len(successful),
            "prompt_tokens": sum(int(record["prompt_tokens"]) for record in selected),
            "ttft_total_seconds": sum(ttfts),
            "ttft_mean_seconds": statistics.fmean(ttfts) if ttfts else None,
            "ttft_median_seconds": statistics.median(ttfts) if ttfts else None,
            "ttft_p90_seconds": percentile(ttfts, 0.90),
            "ttft_p99_seconds": percentile(ttfts, 0.99),
            "latency_mean_seconds": statistics.fmean(totals) if totals else None,
            "metrics_delta": dict(sorted(metrics.items())),
        }
        quality = [record for record in successful if record.get("expected_text")]
        if quality:
            phases[phase]["quality_requests"] = len(quality)
            phases[phase]["quality_accuracy"] = sum(
                bool(record.get("correct")) for record in quality
            ) / len(quality)
    gains: dict[str, float] = {}
    cold = phases.get("cold", {}).get("ttft_mean_seconds")
    blend = phases.get("blend", {}).get("ttft_mean_seconds")
    apc = phases.get("apc_repeat", {}).get("ttft_mean_seconds")
    if cold and blend:
        gains["cold_over_blend"] = cold / blend
    if blend and apc:
        gains["blend_over_apc_floor"] = blend / apc
    return {
        "requests": len(records),
        "errors": sum("error" in record for record in records),
        "phases": phases,
        "gains": gains,
    }


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def retrieval_code(document_id: int) -> str:
    return f"ZXQ{document_id:06d}"


def build_document(
    tokenizer: Any,
    document_id: int,
    target_tokens: int,
    include_retrieval_code: bool = False,
) -> list[int]:
    """Build deterministic, position-varying text and trim it to an exact size."""
    if target_tokens <= 0:
        raise ValueError("document_tokens must be positive")
    sentences = [f"Document {document_id} contains reproducible benchmark facts. "]
    if include_retrieval_code:
        sentences.append(
            f"The exact retrieval code for document {document_id} is "
            f"{retrieval_code(document_id)}. "
        )
    fact = 0
    token_ids: list[int] = []
    while len(token_ids) < target_tokens:
        for _ in range(512):
            value = (document_id + 17) * 1_000_003 + fact * 97
            sentences.append(
                f"Fact {fact} for document {document_id} has deterministic value {value}. "
            )
            fact += 1
        token_ids = _tokenize(tokenizer, "".join(sentences))
    return token_ids[:target_tokens]


def token_digest(token_ids: list[int]) -> str:
    payload = ",".join(str(token) for token in token_ids).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def build_orders(
    num_documents: int,
    documents_per_query: int,
    num_requests: int,
    seed: int,
) -> list[tuple[int, ...]]:
    if not 1 <= documents_per_query <= num_documents:
        raise ValueError("documents_per_query must be within the document set")
    if num_requests > num_documents:
        raise ValueError("blend_requests must not exceed num_documents")
    rng = random.Random(seed)
    orders: list[tuple[int, ...]] = []
    for request_index in range(num_requests):
        selected = [(request_index + offset) % num_documents for offset in range(documents_per_query)]
        # Keep a different leading document for each request in the first
        # corpus cycle. Native APC can then consume that populated document,
        # but cannot accidentally consume a multi-document prefix produced by
        # an earlier blend request. Shuffle only the remaining documents.
        shuffled = selected[1:]
        if len(shuffled) > 1:
            for _ in range(100):
                rng.shuffle(shuffled)
                candidate = (selected[0], *shuffled)
                if candidate != tuple(selected) and candidate not in orders:
                    break
        orders.append((selected[0], *shuffled))
    return orders


def build_prompt(
    prefix: list[int],
    separator: list[int],
    documents: dict[int, list[int]],
    order: tuple[int, ...],
    question: list[int],
) -> list[int]:
    prompt = list(prefix)
    for document_id in order:
        prompt.extend(separator)
        prompt.extend(documents[document_id])
    prompt.extend(separator)
    prompt.extend(question)
    return prompt


def fetch_metrics(url: str, timeout: float) -> dict[str, float]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        lines = response.read().decode().splitlines()
    result = {name: 0.0 for name in METRICS}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        try:
            label, raw_value = line.rsplit(None, 1)
            name = label.split("{", 1)[0]
            if name in result:
                result[name] += float(raw_value)
        except (ValueError, TypeError):
            continue
    return result


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {
        name: after.get(name, 0.0) - before.get(name, 0.0)
        for name in sorted(set(before) | set(after))
    }


def stream_completion(
    api_base: str,
    model: str,
    prompt_token_ids: list[int],
    output_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt_token_ids,
            "max_tokens": output_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": True,
        }
    ).encode()
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_at = None
    response_id = None
    output_fragments: list[str] = []
    usage: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            response_id = response_id or chunk.get("id")
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                text = choice.get("text") or ""
                if text and first_token_at is None:
                    first_token_at = time.perf_counter()
                output_fragments.append(text)
    finished = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("stream completed without a generated token")
    return {
        "response_id": response_id,
        "ttft_seconds": first_token_at - started,
        "total_seconds": finished - started,
        "output_text": "".join(output_fragments),
        "usage": usage,
    }


def run_request(args: argparse.Namespace, spec: RequestSpec) -> dict[str, Any]:
    before = fetch_metrics(args.metrics_url, args.timeout)
    started_ns = time.time_ns()
    try:
        result = stream_completion(
            args.api_base,
            args.model,
            spec.prompt_token_ids,
            args.output_tokens,
            args.timeout,
        )
        record = {
            "phase": spec.phase,
            "name": spec.name,
            "document_ids": list(spec.document_ids),
            "prompt_tokens": len(spec.prompt_token_ids),
            "client_started_ns": started_ns,
            **result,
        }
        if spec.expected_text is not None:
            record["expected_text"] = spec.expected_text
            record["correct"] = spec.expected_text.casefold() in result[
                "output_text"
            ].casefold()
    except Exception as error:  # noqa: BLE001 - record each failed request and continue
        record = {
            "phase": spec.phase,
            "name": spec.name,
            "document_ids": list(spec.document_ids),
            "prompt_tokens": len(spec.prompt_token_ids),
            "client_started_ns": started_ns,
            "error": f"{type(error).__name__}: {error}",
        }
    after = fetch_metrics(args.metrics_url, args.timeout)
    record["metrics_delta"] = metric_delta(before, after)
    return record


def create_workload(args: argparse.Namespace, tokenizer: Any) -> tuple[list[RequestSpec], dict]:
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    prefix = bos + _tokenize(tokenizer, "You are a precise document analysis assistant.")
    separator = _tokenize(tokenizer, "\n\n--- CACHEBLEND DOCUMENT BOUNDARY ---\n\n")
    populate_question = _tokenize(tokenizer, "\nAcknowledge this document with one token.")
    documents = {
        index: build_document(
            tokenizer,
            index,
            args.document_tokens,
            include_retrieval_code=args.quality,
        )
        for index in range(args.num_documents)
    }
    orders = build_orders(
        args.num_documents,
        args.documents_per_query,
        args.blend_requests,
        args.seed,
    )

    specs: list[RequestSpec] = []
    for document_id in range(args.num_documents):
        specs.append(
            RequestSpec(
                "populate",
                f"populate-{document_id}",
                build_prompt(
                    prefix,
                    separator,
                    documents,
                    (document_id,),
                    populate_question,
                ),
                (document_id,),
            )
        )
    blend_specs = []
    for index, order in enumerate(orders):
        target = order[-1]
        expected = retrieval_code(target) if args.quality else None
        question = (
            f"\nWhat is the exact retrieval code for document {target}? "
            "Return only that code."
            if args.quality
            else "\nCompare the supplied documents with one token."
        )
        blend_specs.append(
            RequestSpec(
                "blend",
                f"blend-{index}",
                build_prompt(
                    prefix,
                    separator,
                    documents,
                    order,
                    _tokenize(tokenizer, question),
                ),
                order,
                expected,
            )
        )
    specs.extend(blend_specs)
    for index in range(args.apc_repeats):
        source = blend_specs[index % len(blend_specs)]
        specs.append(
            RequestSpec(
                "apc_repeat",
                f"apc-repeat-{index}",
                source.prompt_token_ids,
                source.document_ids,
                source.expected_text,
            )
        )

    cold_documents: dict[int, list[int]] = {}
    for cold_index in range(args.cold_requests * args.documents_per_query):
        document_id = args.num_documents + cold_index
        cold_documents[document_id] = build_document(
            tokenizer,
            document_id,
            args.document_tokens,
            include_retrieval_code=args.quality,
        )
    for request_index in range(args.cold_requests):
        begin = request_index * args.documents_per_query + args.num_documents
        order = tuple(range(begin, begin + args.documents_per_query))
        target = order[-1]
        expected = retrieval_code(target) if args.quality else None
        question = (
            f"\nWhat is the exact retrieval code for document {target}? "
            "Return only that code."
            if args.quality
            else "\nCompare the supplied documents with one token."
        )
        specs.append(
            RequestSpec(
                "cold",
                f"cold-{request_index}",
                build_prompt(
                    prefix,
                    separator,
                    cold_documents,
                    order,
                    _tokenize(tokenizer, question),
                ),
                order,
                expected,
            )
        )

    manifest = {
        "seed": args.seed,
        "document_tokens": args.document_tokens,
        "num_documents": args.num_documents,
        "documents_per_query": args.documents_per_query,
        "quality": args.quality,
        "blend_orders": [list(order) for order in orders],
        "documents": {
            str(index): {"tokens": len(tokens), "digest": token_digest(tokens)}
            for index, tokens in {**documents, **cold_documents}.items()
        },
    }
    return specs, manifest


def create_engine_warmup(args: argparse.Namespace, tokenizer: Any) -> RequestSpec:
    """Create an equal-shape miss that cannot overlap the measured corpus."""
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    prefix = bos + _tokenize(tokenizer, "You are a precise document analysis assistant.")
    separator = _tokenize(tokenizer, "\n\n--- CACHEBLEND DOCUMENT BOUNDARY ---\n\n")
    question = _tokenize(tokenizer, "\nCompare the supplied documents with one token.")
    first_id = 1_000_000
    documents = {
        first_id + index: build_document(tokenizer, first_id + index, args.document_tokens)
        for index in range(args.documents_per_query)
    }
    order = tuple(documents)
    return RequestSpec(
        "engine_warmup",
        "engine-warmup",
        build_prompt(prefix, separator, documents, order, question),
        order,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="http://127.0.0.1:8123/v1")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8123/metrics")
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer path or model ID")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-documents", type=int, default=4)
    parser.add_argument("--document-tokens", type=int, default=4096)
    parser.add_argument("--documents-per-query", type=int, default=4)
    parser.add_argument("--blend-requests", type=int, default=4)
    parser.add_argument("--apc-repeats", type=int, default=2)
    parser.add_argument("--cold-requests", type=int, default=2)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Ask deterministic needle questions and report exact-code accuracy",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle-seconds", type=float, default=0.50)
    parser.add_argument("--timeout", type=float, default=1800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.blend_requests <= 0:
        raise ValueError("blend_requests must be positive")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    specs, manifest = create_workload(args, tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest["arguments"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    # Use the measured long-prompt shape so the first populate/blend request is
    # not charged for lazy kernels or transfer-thread startup. The document IDs
    # are disjoint from both the populate and cold corpora.
    warmup = create_engine_warmup(args, tokenizer)
    manifest["engine_warmup"] = {
        "prompt_tokens": len(warmup.prompt_token_ids),
        "digest": token_digest(warmup.prompt_token_ids),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    warmup_record = run_request(args, warmup)
    if "error" in warmup_record:
        raise RuntimeError(warmup_record["error"])
    print(
        f"[warmup] prompt_tokens={len(warmup.prompt_token_ids)} "
        f"ttft={warmup_record['ttft_seconds']:.4f}s",
        flush=True,
    )
    time.sleep(args.settle_seconds)

    records: list[dict[str, Any]] = []
    records_path = args.output_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8", buffering=1) as output:
        previous_phase = None
        for index, spec in enumerate(specs, 1):
            if previous_phase is not None and spec.phase != previous_phase:
                time.sleep(args.settle_seconds)
            record = run_request(args, spec)
            records.append(record)
            output.write(json.dumps(record, sort_keys=True) + "\n")
            status = record.get("error", f"ttft={record['ttft_seconds']:.4f}s")
            print(f"[{index}/{len(specs)}] {spec.phase}/{spec.name} {status}", flush=True)
            if spec.phase == "populate":
                time.sleep(args.settle_seconds)
            previous_phase = spec.phase

    summary = summarize_records(records)
    summary["manifest"] = manifest
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
