"""Out-of-tree CacheBlend V1 KV connector for vLLM 0.18."""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import init_logger

from .ascend import PagedBlendLoader, Qwen3BlendExecutor
from .compat import verify_runtime
from .config import CacheBlendConfig
from .ipc import LookupServer, TensorParallelLookup, endpoint_uri
from .kv_layout import gather_paged_kv
from .registry import get_model
from .storage import LocalPinnedCPUStore
from .trace import PipelineTracer
from .types import BlendPlan, SegmentId

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _slot_mapping(block_ids: list[int], block_size: int, token_count: int) -> torch.Tensor:
    blocks = torch.tensor(block_ids, dtype=torch.long)
    offsets = torch.arange(block_size, dtype=torch.long)
    return (blocks[:, None] * block_size + offsets[None, :]).flatten()[:token_count]


def _layer_sort_key(name: str) -> tuple[int, str]:
    matches = re.findall(r"(?:layers?\.)?(\d+)", name)
    return (int(matches[-1]) if matches else 1 << 30, name)


@dataclass(slots=True)
class RequestMetadata:
    request_id: str
    token_ids: list[int]
    slot_mapping: torch.Tensor
    plan: BlendPlan | None
    save: bool
    save_end: int


@dataclass(slots=True)
class _SchedulerRequestState:
    request: Request
    block_ids: list[int]
    cacheable: bool


@dataclass(slots=True)
class CacheBlendConnectorMetadata(KVConnectorMetadata):
    requests: list[RequestMetadata] = field(default_factory=list)


class CacheBlendConnectorV1(KVConnectorBase_V1, SupportsHMA):
    """APC-first, non-contiguous CacheBlend without an LMCache dependency."""

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        # Prompt prefill is not captured by FULL_DECODE_ONLY. With the default
        # prompt-only store, decode graph replay never needs the layerwise save
        # callbacks. Decode-cache storage does require Python between pieces.
        return bool(extra_config.get("save_decode_cache", False))

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        transfer = vllm_config.kv_transfer_config
        assert transfer is not None
        self.config = CacheBlendConfig.from_extra_config(transfer.kv_connector_extra_config)
        verify_runtime(self.config.strict_version_check)
        self._recompute_on_failure = transfer.kv_load_failure_policy == "recompute"
        self.engine_id = str(transfer.engine_id)
        self.block_size = int(vllm_config.cache_config.block_size)
        self.tp_size = int(vllm_config.parallel_config.tensor_parallel_size)
        if int(vllm_config.parallel_config.pipeline_parallel_size) != 1:
            raise ValueError("CacheBlendConnectorV1 currently requires pipeline_parallel_size=1")
        self.model_scope = self.config.model_scope or self._make_model_scope(vllm_config)

        self._lookup: TensorParallelLookup | None = None
        self._cancelled_lookup_ids: set[str] = set()
        self._plans: dict[str, BlendPlan] = {}
        self._load_requests: dict[str, Request] = {}
        self._request_objects: dict[str, Request] = {}
        self._scheduler_requests: dict[str, _SchedulerRequestState] = {}

        self._store: LocalPinnedCPUStore | None = None
        self._server: LookupServer | None = None
        self._kv_caches: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = []
        self._layer_ids: dict[str, int] = {}
        self._executor: Qwen3BlendExecutor | None = None
        self._load_errors: set[int] = set()
        self._save_pool: ThreadPoolExecutor | None = None
        self._save_futures: set[Future[bool]] = set()
        self._save_segments: dict[str, list[tuple[int, SegmentId]]] = {}
        self._pending_store_request_ids: set[str] = set()
        self._save_lock = threading.Lock()
        self._store_stream: Any | None = None
        # A Qwen3 request emits one transfer per layer.  Keep enough pinned
        # buffers for two full layer stacks so Store can overlap Serving while
        # retaining every staging allocation until its D2H event completes.
        self._save_slots = threading.Semaphore(128)
        component = "scheduler" if role is KVConnectorRole.SCHEDULER else "worker"
        self._trace = PipelineTracer(component)
        self._trace.emit(
            "connector_initialized",
            engine_id=self.engine_id,
            model_scope=self.model_scope,
            role=str(role),
            tp_size=self.tp_size,
        )

        if role is KVConnectorRole.SCHEDULER:
            uris = [
                endpoint_uri(self.config.ipc_root, self.engine_id, rank)
                for rank in range(self.tp_size)
            ]
            self._lookup = TensorParallelLookup(uris)

    @staticmethod
    def _make_model_scope(vllm_config: VllmConfig) -> str:
        model_config = vllm_config.model_config
        model = str(getattr(model_config, "model", "unknown-model"))
        revision = str(getattr(model_config, "revision", ""))
        dtype = str(getattr(model_config, "dtype", "unknown"))
        tp = int(vllm_config.parallel_config.tensor_parallel_size)
        return f"{model}@{revision}|{dtype}|tp={tp}"

    # Worker-side API -----------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        if self.role is not KVConnectorRole.WORKER:
            return
        ordered = sorted(kv_caches.items(), key=lambda item: _layer_sort_key(item[0]))
        self._layer_ids = {name: index for index, (name, _) in enumerate(ordered)}
        self._kv_caches = [value for _, value in ordered]
        if not self._kv_caches:
            raise RuntimeError("CacheBlendConnectorV1 received no KV caches")
        try:
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

            rank = int(get_tensor_model_parallel_rank())
        except (AssertionError, ImportError):
            rank = 0
        self._store = LocalPinnedCPUStore(
            chunk_size=self.config.chunk_size,
            max_bytes=self.config.max_local_cpu_bytes,
        )
        self._server = LookupServer(self.config.ipc_root, self.engine_id, rank, self._store)
        self._server.start()
        self._save_pool = ThreadPoolExecutor(
            max_workers=8,
            thread_name_prefix=f"cacheblend-save-rank-{rank}",
        )
        first_cache = self._kv_caches[0]
        first_tensor = first_cache if isinstance(first_cache, torch.Tensor) else first_cache[0]
        if first_tensor.device.type == "npu":
            # Match LMCache's layerwise writeback topology: one default-priority
            # Store stream owns both paged-KV gather and the following D2H copy.
            # The separate CacheBlend loader continues to own its Load stream.
            self._store_stream = torch.npu.Stream()
        model = get_model(self.engine_id)
        self._executor = Qwen3BlendExecutor(model, self.config)
        self._trace.emit(
            "worker_registered",
            rank=rank,
            layers=len(self._kv_caches),
            cpu_cache_bytes=self.config.max_local_cpu_bytes,
        )
        logger.info(
            "CacheBlend worker rank %d registered %d layers with %.1f GiB CPU cache",
            rank,
            len(self._kv_caches),
            self.config.local_cpu_gb,
        )

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, CacheBlendConnectorMetadata):
            raise TypeError("unexpected CacheBlend connector metadata")
        if self._store is None or self._executor is None:
            if any(request.plan is not None for request in metadata.requests):
                raise RuntimeError("CacheBlend worker was not initialized")
            return
        for request in metadata.requests:
            if request.plan is None:
                continue
            trace_enabled = self._trace.enabled
            blend_started_ns = time.monotonic_ns() if trace_enabled else 0
            if trace_enabled:
                self._trace.emit(
                    "blend_started",
                    request_id=request.request_id,
                    apc_prefix_tokens=request.plan.apc_prefix_tokens,
                    allocation_tokens=request.plan.allocation_tokens,
                    hit_tokens=request.plan.hit_tokens,
                    gap_tokens=request.plan.gap_tokens,
                    segments=len(request.plan.segments),
                )
            loader = PagedBlendLoader(
                self._kv_caches,
                self._store,
                request.request_id,
                request.plan,
                request.slot_mapping,
                self.block_size,
                self.config,
                tracer=self._trace,
            )
            saved_moe_index = getattr(forward_context, "moe_layer_index", None)
            has_layer_idx = hasattr(forward_context, "layer_idx")
            saved_layer_idx = getattr(forward_context, "layer_idx", None)
            blend_error: str | None = None
            try:
                self._executor.run(request.request_id, request.token_ids, request.plan, loader)
            except Exception as error:
                blend_error = type(error).__name__
                self._record_load_error(request)
                if not self._recompute_on_failure:
                    raise
                logger.exception(
                    "CacheBlend load failed for %s; reporting its blocks for recompute",
                    request.request_id,
                )
            finally:
                if hasattr(forward_context, "moe_layer_index"):
                    forward_context.moe_layer_index = saved_moe_index
                if has_layer_idx:
                    forward_context.layer_idx = saved_layer_idx
                self._store.release(request.request_id)
                if trace_enabled:
                    self._trace.emit(
                        "blend_finished",
                        request_id=request.request_id,
                        duration_us=(time.monotonic_ns() - blend_started_ns) / 1000,
                        error=blend_error,
                    )

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        if self._store is None:
            return
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, CacheBlendConnectorMetadata):
            return
        layer_id = self._layer_ids.get(layer_name)
        if layer_id is None:
            # Some attention backends use a shortened layer name. Resolve it
            # only when the numeric index is unambiguous.
            key = _layer_sort_key(layer_name)[0]
            layer_id = next(
                (
                    index
                    for name, index in self._layer_ids.items()
                    if _layer_sort_key(name)[0] == key
                ),
                None,
            )
        if layer_id is None:
            return
        for request in metadata.requests:
            if not request.save:
                continue
            usable = min(
                request.save_end,
                len(request.token_ids),
                request.slot_mapping.numel(),
            )
            usable = usable // self.config.chunk_size * self.config.chunk_size
            pending: list[tuple[Any, int, int]] = []
            slot_parts: list[torch.Tensor] = []
            for start, segment_id in self._request_save_segments(request, usable):
                end = start + self.config.chunk_size
                if not self._store.reserve_layer(segment_id, layer_id):
                    continue
                pending.append((segment_id, layer_id, self.config.chunk_size))
                slot_parts.append(request.slot_mapping[start:end])
            if not pending:
                continue
            trace_enabled = self._trace.enabled
            store_started_ns = time.monotonic_ns() if trace_enabled else 0
            try:
                slots = torch.cat(slot_parts)
                if self._store_stream is None or not self.config.async_fingerprint:
                    contiguous = gather_paged_kv(
                        kv_layer,
                        slots,
                        self.block_size,
                    ).contiguous()
                    host = contiguous.detach().to(device="cpu", copy=True)
                    self._publish_host_batch(pending, host)
                else:
                    self._enqueue_store_batch(
                        pending,
                        kv_layer,
                        slots,
                        request_id=request.request_id,
                        store_enqueued_ns=(
                            time.monotonic_ns() if trace_enabled else None
                        ),
                    )
                    if trace_enabled:
                        self._trace.emit(
                            "save_store_enqueued",
                            request_id=request.request_id,
                            layer_id=layer_id,
                            chunks=len(pending),
                            tokens=sum(length for _, _, length in pending),
                            duration_us=(time.monotonic_ns() - store_started_ns) / 1000,
                        )
            except Exception:
                for segment_id, reserved_layer_id, _ in pending:
                    self._store.cancel_layer(segment_id, reserved_layer_id)
                raise

    def _request_save_segments(
        self,
        request: RequestMetadata,
        usable: int,
    ) -> list[tuple[int, SegmentId]]:
        """Build content-addressed chunk identities once per request."""
        assert self._store is not None
        segments = self._save_segments.setdefault(request.request_id, [])
        start = len(segments) * self.config.chunk_size
        while start < usable:
            end = start + self.config.chunk_size
            segment_id = self._store.begin_put(
                self.model_scope,
                start,
                request.token_ids[start:end],
                len(self._kv_caches),
            )
            segments.append((start, segment_id))
            start = end
        return segments[: usable // self.config.chunk_size]

    def _enqueue_store_batch(
        self,
        pending: list[tuple[Any, int, int]],
        kv_layer: Any,
        slots: torch.Tensor,
        request_id: str | None = None,
        store_enqueued_ns: int | None = None,
    ) -> None:
        """Enqueue paged gather and D2H on the single Store stream."""
        assert self._store is not None
        self._save_slots.acquire()
        try:
            if self._store_stream is None:
                raise RuntimeError("CacheBlend NPU Store stream was not initialized")
            current_stream = torch.npu.current_stream()
            with torch.npu.stream(self._store_stream):
                self._store_stream.wait_stream(current_stream)
                contiguous = gather_paged_kv(
                    kv_layer,
                    slots,
                    self.block_size,
                ).contiguous()
                host = torch.empty(
                    contiguous.shape,
                    dtype=contiguous.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                host.copy_(contiguous, non_blocking=True)
                store_event = torch.npu.Event()
                store_event.record(self._store_stream)
            assert self._save_pool is not None
            future = self._save_pool.submit(
                self._publish_after_event,
                store_event,
                pending,
                host,
                contiguous,
                request_id,
                store_enqueued_ns,
            )
        except Exception:
            self._save_slots.release()
            raise
        with self._save_lock:
            self._save_futures.add(future)
            if request_id is not None:
                self._pending_store_request_ids.add(request_id)
        future.add_done_callback(self._save_done)

    def _publish_host_batch(
        self,
        pending: list[tuple[Any, int, int]],
        host: torch.Tensor,
    ) -> bool:
        assert self._store is not None
        offset = 0
        committed = False
        for segment_id, layer_id, length in pending:
            # Store a view of the owned pinned batch.  Keeping the view alive
            # also keeps its parent allocation alive and avoids a second CPU
            # memcpy for every 256-token chunk and layer.
            layer_host = host[:, offset : offset + length]
            committed |= self._store.put_layer_host(segment_id, layer_id, layer_host)
            offset += length
        return committed

    def _publish_after_event(
        self,
        event: Any,
        pending: list[tuple[Any, int, int]],
        host: torch.Tensor,
        device_source: torch.Tensor,
        request_id: str | None = None,
        store_enqueued_ns: int | None = None,
    ) -> bool:
        assert self._store is not None
        try:
            event.synchronize()
            # ``device_source`` is intentionally held by this future until the
            # Store event completes.  Dropping the last reference sooner can
            # let the caching allocator recycle staging while D2H still reads
            # it on the Store stream.
            del device_source
            committed = self._publish_host_batch(pending, host)
            if self._trace.enabled:
                self._trace.emit(
                    "save_store_finished",
                    request_id=request_id,
                    layer_id=pending[0][1] if pending else None,
                    chunks=len(pending),
                    tokens=sum(length for _, _, length in pending),
                    duration_us=(
                        (time.monotonic_ns() - store_enqueued_ns) / 1000
                        if store_enqueued_ns is not None
                        else None
                    ),
                    committed_segment=committed,
                )
            return committed
        except Exception:
            for segment_id, layer_id, _ in pending:
                self._store.cancel_layer(segment_id, layer_id)
            raise

    def _save_done(self, future: Future[bool]) -> None:
        self._save_slots.release()

    def wait_for_save(self) -> None:
        """Wait until the single Store stream and host publication complete."""
        started_ns = time.monotonic_ns() if self._trace.enabled else 0
        if self._store_stream is not None:
            self._store_stream.synchronize()
        with self._save_lock:
            futures = tuple(self._save_futures)
            request_ids = tuple(self._pending_store_request_ids)
            self._pending_store_request_ids.clear()
        for future in futures:
            future.result()
        with self._save_lock:
            self._save_futures.difference_update(futures)
        if self._trace.enabled:
            duration_us = (time.monotonic_ns() - started_ns) / 1000
            for request_id in request_ids:
                self._trace.emit(
                    "save_store_wait_finished",
                    request_id=request_id,
                    stores=len(futures),
                    duration_us=duration_us,
                )

    def _drain_saves(self) -> None:
        while True:
            with self._save_lock:
                futures = tuple(self._save_futures)
            if not futures:
                return
            for future in futures:
                future.result()
            with self._save_lock:
                self._save_futures.difference_update(futures)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        if self._store is not None:
            for request_id in finished_req_ids:
                self._store.release(request_id)
                self._save_segments.pop(request_id, None)
        return None, None

    def get_block_ids_with_load_errors(self) -> set[int]:
        result = self._load_errors
        self._load_errors = set()
        return result

    def _record_load_error(self, request: RequestMetadata) -> None:
        if request.plan is None:
            return
        slots = request.slot_mapping[request.plan.apc_prefix_tokens : request.plan.allocation_end]
        self._load_errors.update(int(value) for value in torch.unique(slots // self.block_size))

    # Scheduler-side API --------------------------------------------------

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if self.role is not KVConnectorRole.SCHEDULER:
            return 0, False
        request_id = request.request_id
        self._cancelled_lookup_ids.discard(request_id)
        self._request_objects[request_id] = request
        if not self._is_cacheable(request):
            return 0, False
        existing = self._plans.get(request_id)
        if existing is not None:
            return max(0, existing.allocation_end - num_computed_tokens), False
        tokens = list(request.prompt_token_ids or [])
        if len(tokens) - num_computed_tokens <= self.config.chunk_size:
            self._plans[request_id] = BlendPlan(num_computed_tokens, num_computed_tokens, 0, ())
            return 0, False
        # Resolve the local-CPU lookup in this scheduler pass. Returning None
        # makes vLLM repeat its full APC block lookup and defer the request to a
        # later engine step; for local UDS, that scheduling round costs much
        # more than the fingerprint scan itself.
        try:
            plan = self._lookup_plan(request_id, tokens, num_computed_tokens)
        except Exception as error:
            self._trace.emit(
                "lookup_failed",
                request_id=request_id,
                error=type(error).__name__,
            )
            logger.exception("CacheBlend lookup failed for %s; recomputing", request_id)
            plan = BlendPlan(num_computed_tokens, num_computed_tokens, 0, ())
        self._plans[request_id] = plan
        return max(0, plan.allocation_end - num_computed_tokens), False

    def _lookup_plan(
        self,
        request_id: str,
        tokens: list[int],
        apc_prefix_tokens: int,
    ) -> BlendPlan:
        assert self._lookup is not None
        trace_enabled = self._trace.enabled
        lookup_started_ns = time.monotonic_ns() if trace_enabled else 0
        if trace_enabled:
            self._trace.emit(
                "lookup_started",
                request_id=request_id,
                prompt_tokens=len(tokens),
                apc_prefix_tokens=apc_prefix_tokens,
            )
        # vLLM requires one prompt token to remain for a regular scheduling
        # step. Searching tokens[:-1] avoids a partial final cache segment.
        segments = self._lookup.lookup_and_prefetch(
            request_id,
            self.model_scope,
            tokens[:-1],
            apc_prefix_tokens,
        )
        if request_id in self._cancelled_lookup_ids:
            self._lookup.cancel(request_id)
            if trace_enabled:
                self._trace.emit(
                    "lookup_finished",
                    request_id=request_id,
                    duration_us=(time.monotonic_ns() - lookup_started_ns) / 1000,
                    outcome="cancelled",
                    apc_prefix_tokens=apc_prefix_tokens,
                    segments=0,
                    hit_tokens=0,
                )
            return BlendPlan(apc_prefix_tokens, apc_prefix_tokens, 0, ())
        plan = BlendPlan.from_segments(apc_prefix_tokens, segments)
        if not plan.passes_gate(
            self.config.min_retrieve_tokens,
            self.config.min_hit_ratio,
            self.config.min_saved_tokens,
            self.config.max_apc_prefix_to_hit_ratio,
        ):
            logger.info(
                "CacheBlend plan rejected request=%s apc=%d span=%d hits=%d segments=%d",
                request_id,
                apc_prefix_tokens,
                plan.allocation_tokens,
                plan.hit_tokens,
                len(plan.segments),
            )
            self._lookup.cancel(request_id)
            if trace_enabled:
                self._trace.emit(
                    "lookup_finished",
                    request_id=request_id,
                    duration_us=(time.monotonic_ns() - lookup_started_ns) / 1000,
                    outcome="rejected",
                    apc_prefix_tokens=apc_prefix_tokens,
                    segments=len(plan.segments),
                    allocation_tokens=plan.allocation_tokens,
                    hit_tokens=plan.hit_tokens,
                    gap_tokens=plan.gap_tokens,
                )
            return BlendPlan(apc_prefix_tokens, apc_prefix_tokens, 0, ())
        logger.info(
            "CacheBlend plan accepted request=%s apc=%d allocation_end=%d "
            "hits=%d gaps=%d segments=%d",
            request_id,
            plan.apc_prefix_tokens,
            plan.allocation_end,
            plan.hit_tokens,
            plan.gap_tokens,
            len(plan.segments),
        )
        if trace_enabled:
            self._trace.emit(
                "lookup_finished",
                request_id=request_id,
                duration_us=(time.monotonic_ns() - lookup_started_ns) / 1000,
                outcome="accepted",
                apc_prefix_tokens=plan.apc_prefix_tokens,
                segments=len(plan.segments),
                allocation_tokens=plan.allocation_tokens,
                hit_tokens=plan.hit_tokens,
                gap_tokens=plan.gap_tokens,
            )
        return plan

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens <= 0:
            return
        plan = self._plans.get(request.request_id)
        if plan is None:
            raise RuntimeError(f"missing CacheBlend plan for {request.request_id}")
        self._load_requests[request.request_id] = request

    @staticmethod
    def _is_cacheable(request: Request) -> bool:
        # Token IDs alone do not identify these model inputs, so sharing them
        # in the base-model namespace would be unsafe.
        return (
            getattr(request, "lora_request", None) is None
            and not getattr(request, "mm_features", None)
            and getattr(request, "prompt_embeds", None) is None
        )

    @staticmethod
    def _all_token_ids(request: Request) -> list[int]:
        values = getattr(request, "all_token_ids", None)
        if values is None:
            values = getattr(request, "_all_token_ids", None)
        if values is None:
            values = request.prompt_token_ids
        return list(values or [])

    def _save_end(
        self,
        request: Request,
        computed_after_step: int,
        token_count: int,
    ) -> int:
        if self.config.save_decode_cache:
            return min(computed_after_step, token_count)
        return min(
            computed_after_step,
            len(getattr(request, "prompt_token_ids", None) or ()),
            token_count,
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> CacheBlendConnectorMetadata:
        metadata = CacheBlendConnectorMetadata()
        emitted_loads: set[str] = set()

        for request_id in scheduler_output.finished_req_ids:
            self._scheduler_requests.pop(request_id, None)
        for request_id in scheduler_output.preempted_req_ids or ():
            state = self._scheduler_requests.get(request_id)
            if state is not None:
                state.block_ids.clear()

        for new_request in scheduler_output.scheduled_new_reqs:
            request = self._request_objects.get(new_request.req_id)
            token_ids = (
                self._all_token_ids(request)
                if request is not None
                else list(new_request.prompt_token_ids or [])
            )
            num_tokens = min(
                len(token_ids),
                len(new_request.block_ids[0]) * self.block_size,
            )
            plan = self._plans.get(new_request.req_id)
            if new_request.req_id in self._load_requests:
                emitted_loads.add(new_request.req_id)
            else:
                plan = None
            cacheable = request is not None and self._is_cacheable(request)
            if request is not None:
                self._scheduler_requests[new_request.req_id] = _SchedulerRequestState(
                    request=request,
                    block_ids=list(new_request.block_ids[0]),
                    cacheable=cacheable,
                )
            computed_after = (
                new_request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[new_request.req_id]
            )
            metadata.requests.append(
                RequestMetadata(
                    request_id=new_request.req_id,
                    token_ids=token_ids,
                    slot_mapping=_slot_mapping(
                        new_request.block_ids[0], self.block_size, num_tokens
                    ),
                    plan=plan,
                    save=cacheable,
                    save_end=(
                        self._save_end(request, computed_after, num_tokens)
                        if request is not None
                        else 0
                    ),
                )
            )

        cached = scheduler_output.scheduled_cached_reqs
        for index, request_id in enumerate(cached.req_ids):
            state = self._scheduler_requests.get(request_id)
            if state is None:
                continue
            block_ids_by_group = cached.new_block_ids[index]
            if block_ids_by_group is not None:
                if request_id in cached.resumed_req_ids:
                    state.block_ids = list(block_ids_by_group[0])
                else:
                    state.block_ids.extend(block_ids_by_group[0])
            if not state.block_ids:
                continue
            request = state.request
            token_ids = self._all_token_ids(request)
            num_tokens = min(len(token_ids), len(state.block_ids) * self.block_size)
            computed_after = (
                cached.num_computed_tokens[index]
                + scheduler_output.num_scheduled_tokens[request_id]
            )
            metadata.requests.append(
                RequestMetadata(
                    request_id=request_id,
                    token_ids=token_ids,
                    slot_mapping=_slot_mapping(state.block_ids, self.block_size, num_tokens),
                    plan=(
                        self._plans.get(request_id) if request_id in self._load_requests else None
                    ),
                    save=state.cacheable,
                    save_end=self._save_end(request, computed_after, num_tokens),
                )
            )
            if request_id in self._load_requests:
                emitted_loads.add(request_id)

        for request_id in emitted_loads:
            self._load_requests.pop(request_id, None)
        return metadata

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        self._cleanup_scheduler_request(request.request_id)
        return False, None

    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        self._cleanup_scheduler_request(request.request_id)
        return False, None

    def _cleanup_scheduler_request(self, request_id: str) -> None:
        self._cancelled_lookup_ids.add(request_id)
        self._plans.pop(request_id, None)
        self._load_requests.pop(request_id, None)
        self._request_objects.pop(request_id, None)
        self._scheduler_requests.pop(request_id, None)
        if self._lookup is not None:
            self._lookup.cancel(request_id)

    def close(self) -> None:
        try:
            self._drain_saves()
        finally:
            if self._save_pool is not None:
                self._save_pool.shutdown(wait=True, cancel_futures=False)
                self._save_pool = None
        if self._server is not None:
            self._server.close()
            self._server = None
        if self._lookup is not None:
            self._lookup.close()
            self._lookup = None
        if self._store is not None:
            self._store.clear()
        self._trace.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - destructors must never escape
            return
