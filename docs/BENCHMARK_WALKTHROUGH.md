# 用 benchmark 梳理 CacheBlend 数据流水线

这份说明从 benchmark 的一条请求出发，沿着 vLLM scheduler、KV Connector、
TP worker、NPU 和 pinned CPU cache 走完整条链路。建议先跑一次无 trace 的三臂
对比，再单独跑一次带 trace 的 `cacheblend` arm。

## 先读懂四个被测阶段

`benchmark.py` 直接把 token ID 数组发给 OpenAI Completions API，避免聊天模板、
系统提示词或不同 tokenizer 配置改变文档边界。正式阶段前还有一条同长度、但文档
ID 完全隔离的 miss 请求，用来预热长 prefill shape、lazy kernel 和保存线程；它不计入
结果。

| 阶段 | 输入 | 主要观察对象 |
|---|---|---|
| `populate` | 每次一个新文档 | 未命中计算，以及 KV gather/D2H/commit 保存链路 |
| `blend` | 已保存文档重新排序后拼接 | 非连续命中、位置迁移、选择性重算 |
| `apc_repeat` | 完整重复某条 blend prompt | vLLM 原生连续前缀缓存的理想下界 |
| `cold` | 同长度、从未出现的新文档 | 完整 prefill 的对照组 |

因此，`cold/blend` 是 CacheBlend 是否真正节省 TTFT 的核心比值；
`apc_repeat` 不是竞争对手的真实工作负载，而是说明“如果整个 prompt 都能连续
命中，最低大约能到哪里”。

## 一条 blend 请求怎么流动

```text
benchmark client
  -> vLLM HTTP/OpenAI server
  -> native APC prefix lookup
  -> CacheBlend scheduler lookup over Unix sockets
  -> TP-rank match / validate / pin
  -> vLLM allocates pages for the blend span
  -> connector metadata reaches every worker
  -> layerwise CPU->NPU load + RoPE relocation
  -> gap tokens and selected hit tokens are recomputed
  -> completed KV is scattered into vLLM-owned pages
  -> vLLM performs the remaining regular forward/decode
  -> response stream returns the first output token
```

### 1. 客户端生成与发送数据

入口是 `benchmarks/cacheblend/benchmark.py`：

- `build_document()` 生成可复现内容并裁成**精确 token 数**；
- `build_orders()` 固定随机种子并改变文档顺序；
- `build_prompt()` 用固定边界串联文档；
- `stream_completion()` 记录发出请求到首个非空输出片段的 TTFT；
- `run_request()` 在每条请求前后抓取 `/metrics`，得到该请求的指标增量。

`manifest.json` 中的 token digest 可用来确认不同 arm 用的是同一数据，
`records.jsonl` 则保留每个请求的原始测量结果。

### 2. APC 先处理连续前缀

vLLM 先做自己的 block hash lookup，并把 `num_computed_tokens` 传给 Connector。
CacheBlend 只搜索这个位置之后的 token。这样 APC 擅长的连续前缀仍由 vLLM
管理，插件只负责 APC 无法表达的内部、非连续文档命中。

对应入口是 `CacheBlendConnectorV1.get_num_new_matched_tokens()`。第一次调用把
`_lookup_plan()` 提交给单线程 lookup worker，并返回 pending；后续 scheduler
轮询 Future。这样 Unix socket 等待不会阻塞 scheduler，同时单线程保证持久化
ZeroMQ REQ socket 的严格收发顺序。

### 3. scheduler 向所有 TP rank 查找并固定缓存

`TensorParallelLookup.lookup_and_prefetch()` 的顺序是：

1. 只把 APC 后缀 token 作为 NumPy buffer 发给 rank 0；
2. rank 0 用 token fingerprint 找候选 chunk；
3. 其他 TP rank 并行 `VALIDATE`，取各 rank 都存在的交集；
4. 所有 rank 并行 `PREFETCH`，再次验证并按 request ID pin；
5. 任一 rank 缺少 chunk 就取消整次计划，避免 TP rank 使用不同 KV。

本地服务在 `cacheblend_vllm/ipc.py`，token matcher 和 LRU 存储分别在
`matcher.py`、`storage.py`。

### 4. 盈利门控与页分配

`BlendPlan.from_segments()` 把非连续命中转成一个连续 allocation span，并计算：

- `hit_tokens`：真正从 CPU cache 取回的 token；
- `gap_tokens`：命中之间必须重新计算的 token；
- `allocation_tokens`：vLLM 要为整个 span 分配的 token 数。

`passes_gate()` 使用命中量、命中率、最小节省 token 数，以及 APC-prefix/hit
比例拒绝不划算的计划。接受后，`update_state_after_alloc()` 保存请求对象，
`build_connector_meta()` 把 token、slot mapping 和 plan 发送到 worker。

注意 `/metrics` 中的 `external_prefix_cache_hits_total` 在当前 vLLM Connector
接口里反映的是外部 Connector 申请的连续 span，不等于 CacheBlend 精确命中量。
精确的 `hit_tokens` 和 `gap_tokens` 应从 trace 的 `lookup_finished` 读取。

### 5. 每层加载、位置迁移与选择性重算

worker 在 `start_load_kv()` 创建 `PagedBlendLoader`，随后
`Qwen3BlendExecutor.run()` 执行独立的 Qwen3 layerwise side forward：

1. 双 NPU staging buffer 按奇偶层复用；
2. 独立 load stream 预取下一层的 pinned CPU KV；
3. 把多个命中 segment 融合复制到 staging；
4. 对 K 应用 `R(new) * R(old)^-1`，把原始文档位置迁移到新 prompt 位置；
5. gap token 每层都重算；
6. 默认 `kv_deviation` 在 `check_layers` 比较当前 K 与缓存 K，按
   `recompute_ratios` 选出误差较大的命中 token；`sparse_q` 则让 trailing
   question 和其他 gap Query 在一个较后的 boundary layer 对完整 causal Key
   context 打分，再保留 Top-K 命中、全部 gap、overflow block 和 tail fallback；
7. 将完整 suffix scatter 到 vLLM 已分配的 paged KV cache；
8. load stream 与 compute stream 用 event 建依赖，不做全设备同步。

层计算细节在 `cacheblend_vllm/ascend.py`。模型 wrapper 只负责把已加载模型注册
给 Connector，位置在 `cacheblend_vllm/models.py`。

### 6. 未命中请求如何保存 KV

保存使用与 LMCache 一致的单 Store stream 分层流水：

1. `save_kv_layer()` 让 Store stream 等待当前 serving stream 的层计算；
2. Store stream 从 vLLM pages gather 完整 chunk 到连续 NPU staging；
3. 同一条 Store stream 紧接着执行 NPU->pinned CPU，不再切换第二条保存 stream；
4. 后台线程等待该层的 Store event，再把 host tensor view 发布给
   `LocalPinnedCPUStore`；
5. `wait_for_save()` 同步 Store stream，并等待本批 host publication 完成；
6. 一个 chunk 的**所有模型层**都写完后才向 matcher 注册，查询永远看不到半成品。

这条链路解决两个正确性条件：vLLM 可以安全复用原 pages；后续请求只会命中完整
KV。与之前 Gather + Offload 双 stream 不同，保存完成现在是明确的请求边界，后续
请求无需依赖经验性的 sleep 等待 cache ready。

## Trace 事件如何对应代码

| 事件 | 所在进程/代码 | 能回答的问题 |
|---|---|---|
| `lookup_started/finished` | scheduler，`_lookup_plan()` | IPC + fingerprint + TP 交集有多慢，计划是否被 gate 拒绝 |
| `blend_started/finished` | worker，`start_load_kv()` | 整个 side forward 的 host 观察窗口 |
| `loader_initialized` | worker，`PagedBlendLoader.__init__()` | 本请求 hits/gaps/span 是否符合预期 |
| `layer_prefetch_enqueued` | worker，`prefetch()` | CPU->NPU、位置迁移和相关操作的 host enqueue 开销 |
| `layer_prefetch_wait_enqueued` | worker，`get()` | compute stream 是否建立了预取依赖 |
| `layer_compute_enqueued` | worker，executor loop | 每层实际选择了多少 token 进入重算 |
| `selection_finished` | worker，Sparse-Q boundary | Sparse Query、overflow 与最终选择规模是否符合预期 |
| `selection_compared` | worker，`compare` 模式 | Sparse-Q 与 K-deviation 的 overlap/Jaccard 有多大 |
| `layer_committed` | worker，`commit()` | 哪些层已经把 suffix scatter 回 vLLM pages |
| `save_store_enqueued` | worker，`save_kv_layer()` | 单 Store stream 上 gather + D2H 的 host enqueue 开销 |
| `save_store_finished` | worker，保存线程 | 每层 Store event 完成并发布 host view 的时间 |
| `save_store_wait_finished` | worker，`wait_for_save()` | Store stream 与本批 host publication 的阻塞时间 |

`analyze.py` 会把所有进程的 JSONL 按 wall-clock 时间排序，并按 request ID 写出
`pipeline_timeline.jsonl`。分析器会用 `records.jsonl` 中的 response ID 补上 phase、
name、TTFT 和相对客户端起点的时间，并排除 warmup 事件。TP worker 事件每个 rank
各有一份，这是正常现象，也可以借此发现 rank skew。除
`save_store_wait_finished` 包含同步 Store 完成时间以外，其余事件时间主要用于判断
先后、重叠和 host enqueue 延迟；不能把这些 duration 相加当作 kernel 总耗时。

## 按顺序寻找优化点

先只改变一个变量，并至少重复三轮 clean run：

1. **确认命中正确**：blend 的 `lookup_finished.outcome` 应是 `accepted`，
   `hit_tokens` 接近文档 token 总量，错误数必须为 0。
2. **隔离 lookup**：若 lookup 明显占 TTFT，比较 token 序列化、rank 0 matcher、
   TP validation/pin；可尝试把更多固定 metadata 留在持久连接一侧。
3. **看 gap/span 放大**：若 `allocation_tokens / hit_tokens` 高，优化 chunk 边界、
   文档分隔方式或 profitability gate，通常比微调 kernel 更值。
4. **看 H2D 是否隐藏**：检查第 N+1 层 prefetch 是否与第 N 层 compute 重叠。
   若 compute 经常等 load，再考虑更大的融合传输或 native transfer kernel。
5. **调选择性重算**：先用 `compare` 在同一请求上记录 Sparse-Q 与 K-deviation
   的 overlap/Jaccard，再分别运行两种 selector；改变 `check_layers`、
   `recompute_ratios` 时必须同时观察 TTFT 和 retrieval-code accuracy。比例越低
   不一定越好，错误 KV 会传到后续层。
6. **看保存反压**：`save_store_wait_finished` 直接进入未命中请求延迟。若它很大，
   检查单 Store stream 上 gather/D2H 的带宽、staging 数量和 serving 竞争；此时
   不再需要用 sleep 猜测 cache 是否 ready。
7. **检查 APC 交互**：`apc_prefix_tokens` 很大时应让 APC 接管；避免为很少的
   interior hits 再跑整段 side forward。
8. **最后才 profile kernel**：host trace 定位到 load、attention、scatter 或 D2H
   后，再用 NPU profiler 分解算子、带宽和 stream 空洞。

用于性能结论的结果应记录模型、vLLM/vLLM-Ascend/插件 commit、CANN 与驱动版本、
TP/EP、上下文长度、cache 大小、chunk size、重算比例和每个 arm 至少三轮的分布。
