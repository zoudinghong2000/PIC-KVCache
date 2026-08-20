import json

from cacheblend_vllm.trace import PipelineTracer


def test_pipeline_trace_is_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("CACHEBLEND_TRACE_DIR", raising=False)
    tracer = PipelineTracer("worker")
    tracer.emit("ignored", request_id="r")
    tracer.close()
    assert not list(tmp_path.iterdir())


def test_pipeline_trace_writes_structured_events(monkeypatch, tmp_path):
    monkeypatch.setenv("CACHEBLEND_TRACE_DIR", str(tmp_path))
    tracer = PipelineTracer("scheduler")
    tracer.emit("lookup_done", request_id="r", duration_us=12.5, hit_tokens=256)
    tracer.close()

    rows = [json.loads(line) for line in next(tmp_path.iterdir()).read_text().splitlines()]
    assert rows[0]["event"] == "trace_started"
    assert rows[1]["component"] == "scheduler"
    assert rows[1]["request_id"] == "r"
    assert rows[1]["duration_us"] == 12.5
    assert rows[1]["hit_tokens"] == 256
