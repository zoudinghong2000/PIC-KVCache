# Architecture

CacheBlend is implemented entirely through vLLM's V1 Connector and general
plugin interfaces. Upstream vLLM and vLLM-Ascend do not need patches.

## Process boundaries

The scheduler-side Connector owns synchronous local lookup state and lightweight
request/block trackers. Each TP worker owns an independent pinned-CPU store and
a local Unix-socket lookup server. Rank 0 proposes token-range matches; every
rank validates and pins the exact intersection before the scheduler allocates
the corresponding vLLM pages.

Lookup clients keep one REQ socket per rank for the engine lifetime. The
scheduler sends only the token suffix not already covered by APC, sends rank
validation/pin messages before receiving any reply, and therefore overlaps TP
worker processing without sharing a ZeroMQ socket across threads.

The stores deliberately do not share KV tensors between TP ranks. A segment ID
is common across ranks, while its stored tensor contains that rank's KV shard.
Entries become visible only after every decoder layer is present and remain
pinned until the side forward completes or the request is cancelled.

## Access to the serving model

The `cacheblend_models` general plugin registers transparent subclasses for
`Qwen3ForCausalLM` and `Qwen3MoeForCausalLM`. Their constructors publish the
loaded model in a process-local weak registry when the configured connector is
`CacheBlendConnectorV1`. They do not override `forward`, weight loading, or
model math. This registry replaces the worker patch used by the embedded
implementation; it is not a forward hook.

## Request lifecycle

1. vLLM finds its normal APC prefix.
2. CacheBlend computes all rolling fingerprints with vectorized `uint64`
   arithmetic, intersects them with the scope's known hashes, verifies exact
   token equality for candidates, and leaves the last prompt token for the
   regular forward.
3. TP workers validate and pin a common set of exact token matches.
4. A cost gate rejects small exact-hit sets and cases where the native APC
   prefix is disproportionately larger than the reuse. Rejected requests stay
   on vLLM's normal APC path.
5. The scheduler reports the complete span through the standard external-token
   count, so vLLM allocates and owns every destination block.
6. `start_load_kv` gathers the APC prefix, loads matched CPU segments, relocates
   cached RoPE keys from source to target positions with a direct delta
   rotation, and zeroes gaps. On Ascend, native paged-cache gather/scatter ops
   replace advanced indexing where their layout constraints are satisfied.
7. The Qwen3 side executor runs the allocated suffix layer by layer. At a check
   layer it compares fresh K with cached K, globally selects high-error hits
   across TP ranks, and always retains gap tokens.
8. Each completed layer is scattered into the vLLM-owned paged cache. The
   regular vLLM forward then computes the remaining prompt token and logits.
9. Complete prompt chunks are gathered once per layer into independent staging
   on a high-priority NPU stream. At the end of the forward, the serving stream
   waits only for the final gather event, which makes vLLM's paged blocks safe
   to reuse. D2H is then enqueued behind an end-of-forward barrier on a separate
   default-priority stream and completes in the background. Chunk identities
   are hashed once per request, and stored tensor views retain their pinned
   batch owner to avoid a second CPU copy. Fingerprints are published atomically
   only after every layer's D2H has completed.

The loader double-buffers its per-layer staging tensors. An event prevents a
buffer from being reused until its previous layer has been scattered into the
vLLM-owned pages, allowing the following layer's CPU-to-NPU transfer to overlap
with model computation. Matched pinned-CPU chunks copy their contiguous K and V
planes directly into one reusable NPU hit buffer; source/target position tensors
are built once per request. This follows LMCache-Ascend's fused-transfer layout
without importing its runtime or rebuilding a fused CPU tensor for every layer.
For writeback, the Ascend paged-cache operator writes K and V directly into the
two planes of a single staging allocation. The asynchronous Future retains that
device allocation until its D2H event completes, so allocator reuse cannot race
the offload stream.

## Correctness invariants

- Fingerprint collisions are verified against the complete token chunk.
- Match segments are sorted, non-overlapping, and never overlap the APC prefix.
- A cache entry is not discoverable until all layers have committed.
- vLLM paged blocks are not reusable until the gather stream has stopped
  reading them; CPU writeback completion is not on the serving-stream path.
- TP uses an exact segment intersection; a partial-rank hit is a miss.
- In-flight entries are pinned and cannot be evicted.
- APC pages are gathered as attention context but never overwritten by the
  side executor.
- Every gap token is recomputed, regardless of the configured recompute ratio.
- LoRA, multimodal, and prompt-embedding requests bypass lookup and storage.
- Only chunks that are known computed after the current scheduler step are
  stored; uncomputed capacity in the final block is never published.

## Failure behavior

Lookup failures degrade to a local miss. A side-forward failure either raises
immediately (`kv_load_failure_policy=fail`) or reports only the externally
allocated block IDs to vLLM for invalidation and recompute
(`kv_load_failure_policy=recompute`). Cancellation releases pins on every TP
rank, including the race where a lookup finishes after its request is removed.

## Version and scope boundary

The Connector API and scheduler metadata are pinned to vLLM/vLLM-Ascend 0.18.
The first release supports Qwen3 dense/MoE, single-host TP1/TP2, and PP1. Adding
another architecture requires a transparent registry adapter and a layerwise
executor that preserves that model's attention, normalization, and MLP rules.
