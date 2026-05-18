# Resume-Aware Context 修复改动记录

本文档记录本次围绕 **Resume-Aware Context 修复计划** 做的工程改动、验证结果和后续使用注意事项。计划文件本身未修改。

---

## 1. 改动目标

本次修改的核心目标是把当前 Form-B resume-aware routing 从「能识别 resume 请求」推进到「selected worker 闭环可验证、preferred routing 可软回退、pause/resume 时间语义可观测、context lifecycle 有控制面接口」。

重点解决的问题：

- gateway 之前没有把实际选中的 worker 返回给 ROLL，导致 `_last_backend_id` 闭环不稳定。
- preferred worker 之前是硬亲和，可能在高并发下压热点 worker。
- pause/resume 的时间指标不够细，无法区分外部工具等待、ROLL queue wait 和模型生成耗时。
- 缺少 context/KV 生命周期抽象，后续难以接入 pin/offload/reload/unpin。

---

## 2. Gateway 改动

代码位置：

- `/export/xxl/xxl_sglang/sgl-model-gateway/src/routers/http/router.rs`
- `/export/xxl/xxl_sglang/sgl-model-gateway/src/routers/header_utils.rs`
- `/export/xxl/xxl_sglang/sgl-model-gateway/src/policies/cache_aware.rs`

### 2.1 selected worker 响应 header

在 HTTP router 选中 worker 并拿到 worker response 后，gateway 会向响应追加：

- `X-SMG-Selected-Worker-Url: <worker.url()>`
- `X-SMG-Selected-Worker-Id: <stable worker id>`

这样 ROLL 可以优先根据响应 header 映射 `selected_backend_id`，不再依赖 worker response body 里是否携带 `meta_info.worker_url`。

### 2.2 soft preferred policy

`cache_aware` 的 preferred-worker 分支从硬亲和改为可配置软亲和。新增环境变量：

- `SMG_ENABLE_PREFERRED_OVERRIDE`
  - 默认开启。
  - 设置为 `0` / `false` / `off` / `no` 时关闭 preferred override。
- `SMG_PREFERRED_MAX_LOAD`
  - preferred worker 当前 load 超过该值时跳过 preferred。
- `SMG_PREFERRED_MAX_QUEUE_DEPTH`
  - 作为 `SMG_PREFERRED_MAX_LOAD` 的别名兼容使用。
- `SMG_PREFERRED_LOAD_MARGIN`
  - preferred worker load 相比当前最小 load 超过 margin 时跳过 preferred。
- `SMG_PREFERRED_MAX_PAUSE_AGE_S`
  - resume pause age 超过该值时跳过 preferred，避免过旧上下文继续强亲和。

新增/保留的 policy branch metrics：

- `preferred_hit`
- `preferred_skip_disabled`
- `preferred_skip_overloaded`
- `preferred_skip_old_pause`
- `preferred_miss_unhealthy`
- `preferred_miss_not_found`
- `preferred_miss_empty`
- `preferred_with_history_len`
- `preferred_present`

计算 preferred hit/skip 相关比例时，应使用 `preferred_present` 作为分母，或只在带 preferred header 的 resume 请求上计算。`preferred_miss_empty` 会被 normal 请求放大，不适合作为 preferred hit rate 分母。

ROLL 会通过 header 向 gateway 传入：

- `X-ROLL-Preferred-Worker-Url`
- `X-ROLL-Pause-Age-S`
- `X-ROLL-History-Len-Tokens`

---

## 3. ROLL Router 改动

代码位置：

- `roll/distributed/scheduler/router.py`

### 3.1 selected backend 闭环

`SglangOrderingRouter._router_generate()` 现在会优先读取 gateway response header：

- `x-smg-selected-worker-url`

如果该 URL 在 `self.worker_urls` 中，则写入：

- `out["selected_backend_id"]`
- `out["selected_worker_url"]`

如果 header 不存在或无法映射，则保留原有 `_attach_selected_backend_id()` body meta fallback，继续兼容：

- `meta_info.selected_backend_id`
- `meta_info.worker_url`
- `meta_info.selected_worker_url`
- `meta_info.url`

注意：Form-B 内部 ordering 使用 `_PLACEMENT_SENTINEL_DP_RANK = 0`，ROLL 侧 `resume_affinity_hit_rate` 只反映 ordering 占位队列语义，不代表 gateway 真实 selected worker affinity。为避免误读，新增真实 worker 闭环指标：

- `selected_backend_affinity_hit`
- `selected_backend_migration`
- `scheduler/router/selected_backend_affinity_hit_rate`
- `scheduler/router/selected_backend_migration_rate`

这些指标比较的是 `last_backend_id == selected_backend_id`。

### 3.2 resume soft policy 输入

对 resume 请求，ROLL 会从 `_roll_route_meta` 中提取并同步为 HTTP header：

- `last_backend_id` -> `X-ROLL-Preferred-Worker-Url`
- `pause_age_s` -> `X-ROLL-Pause-Age-S`
- `history_len_tokens` -> `X-ROLL-History-Len-Tokens`

这样 gateway 可以结合 pause age 和 history length 做 soft preferred/fallback 判断。

### 3.3 queue timing 回传

`SglangOrderingRouter.generate_request()` 在 pending 入队和实际 dispatch 时记录：

- `resume_enqueue_ts`
- `resume_dispatch_ts`
- `resume_queue_wait_s`

这些字段会通过 response/meta 传回 `RouterClient._postprocess_generate()`，进入 `DataProto.meta_info`，供 env manager 记录到 metrics。

---

## 4. Env Manager 时间指标改动

代码位置：

- `roll/pipeline/agentic/env_manager/traj_env_manager.py`

新增 tool/pause/resume 相关指标：

- `tool_call_start_ts`
- `tool_return_ts`
- `external_wait_s`
- `resume_enqueue_ts`
- `resume_dispatch_ts`
- `resume_queue_wait_s`
- `resume_infer_start_ts`
- `resume_first_token_ts`
- `resume_infer_end_ts`
- `resume_infer_latency_s`
- `resume_latency_e2e_s`
- `resume_prefill_tokens`

新增 episode 汇总指标：

- `external_wait_mean_s`
- `external_wait_p50_s`
- `external_wait_p95_s`
- `resume_queue_wait_mean_s`
- `resume_queue_wait_p50_s`
- `resume_queue_wait_p95_s`

说明：

- `tool_call_start_ts` 是进入 `env.step()` 前的时间。
- `tool_return_ts` 是 `env.step()` 返回后的时间。
- `external_wait_s = tool_return_ts - tool_call_start_ts`，代表外部工具等待。
- `resume_queue_wait_s = resume_dispatch_ts - resume_enqueue_ts`，代表 ROLL ordering queue wait。
- `resume_first_token_ts` 当前先用同步 generate 返回时间近似，实际等价于本轮 generate end time，不是真实 TTFT；若后续接入 streaming 或后端返回 first-token timing，可替换为真实 first token 时间。
- 当前细粒度指标只接入 `TrajEnvManager` 路径；`AgentNativeStepEnvManager` / `VLTrajEnvManager` 还未同步。如果实验只跑 GEM/toolcall 的 `TrajEnvManager`，这一点不影响当前验证；若要泛化，需要补齐对应 env manager。

---

## 5. Context Lifecycle 控制面

新增文件：

- `roll/pipeline/agentic/context_lifecycle.py`

新增抽象：

- `ContextState`
  - `GPU_HOT`
  - `CPU_WARM`
  - `EVICTED`
- `ContextRecord`
- `ContextLifecycleManager`

提供接口：

- `pin_context(rid, worker_url, ttl_s, estimated_tokens)`
- `retain_context(rid, ttl_s)`
- `offload_context(rid)`
- `reload_context(rid, worker_url, latency_s)`
- `unpin_context(rid)`
- `classify_resume(rid, worker_url)`
- `collect_metrics(prefix="context")`

当前这是控制面状态机和指标接口，不声明真实 KV 已经被底层 SGLang 保留。后续如果 SGLang worker/gateway 提供真实 KV pin/offload/reload API，可以在这些接口后面接入真实实现。

运行时接入点：

- `SglangOrderingRouter` 初始化时创建 `ContextLifecycleManager`。
- 使用 `trajectory_id` 作为稳定 rid。
- 每次 gateway 返回 selected worker 后，按 selected worker 执行控制面 `pin_context()` / `retain_context()`。
- resume 前根据 `pause_age_s` 和配置做最小状态机：
  - `context_offload_after_s`：超过后控制面 `GPU_HOT -> CPU_WARM`，resume 分类为 `cpu_reload`。
  - `context_evict_after_s` / `context_ttl_s`：超过后控制面 evict，resume 分类为 `full_prefill`。
  - `context_token_budget`：按 token 估算做 LRU 控制面淘汰。
- resume 请求会根据当前控制面状态和 selected worker 产出 one-hot 指标：
  - `context_class_gpu_hit`
  - `context_class_cpu_reload`
  - `context_class_full_prefill`
- `RouterClient._postprocess_generate()` 会把上述字段写入 `DataProto.meta_info`。
- `TrajEnvManager` 会把上述字段写入 TensorBoard metrics，聚合模式为 `sum`。

输出指标包括：

- `scheduler/router/context/gpu_hot_count`
- `scheduler/router/context/cpu_warm_count`
- `scheduler/router/context/record_count`
- `scheduler/router/context/estimated_tokens`
- `scheduler/router/context/token_budget`
- `scheduler/router/context/eviction_count`
- `scheduler/router/context/expiration_count`
- `scheduler/router/context/reload_count`
- `scheduler/router/context/reload_latency_mean_s`
- `scheduler/router/context_class/gpu_hit`
- `scheduler/router/context_class/cpu_reload`
- `scheduler/router/context_class/full_prefill`

限制：

- 这些指标目前仍是控制面分类，不代表底层 SGLang KV cache 已真实 pin/offload/reload。
- `estimated_tokens` 是 token 数，不是 KV bytes；如果要做真实内存预算，需要接入模型层 KV bytes 估算或后端实际 usage。
- 在没有真实后端 API 前，系统只能证明 resume-aware scheduling 与控制面 context 分类，不能证明真实 resume-aware context management。

---

## 6. 验证结果

已完成的验证：

- Python 语法检查：
  - `python3 -m py_compile roll/distributed/scheduler/router.py roll/pipeline/agentic/env_manager/traj_env_manager.py roll/pipeline/agentic/context_lifecycle.py`
- Python lints：
  - `ReadLints` 未发现新增 linter error。
- Rust 编译检查：
  - 在 `roll-r3-dev` 容器内执行 `cargo check` 通过。
- Rust 格式检查：
  - `cargo fmt --all --check` 通过。
- Release binary：
  - `cargo build --release --bin sgl-model-gateway` 构建成功。
- 隔离 gateway smoke：
  - 使用两个最小 HTTP worker 和临时 gateway 端口验证。
  - `X-SMG-Selected-Worker-Url` 正常返回。
  - resume 请求携带 preferred URL 后命中同一 worker。
  - Prometheus metric 中 `smg_cache_aware_policy_branch_total{branch="preferred_hit"}` 增长。
- ROLL header mapping smoke：
  - 使用轻量 `postprocess_generate` stub 避免测试环境缺少 `sgl_kernel`。
  - 验证 `_router_generate()` 能把 gateway selected-worker header 映射为 `selected_backend_id`。
- Form-B timing/context smoke：
  - normal 请求返回 selected worker 后会在控制面 `pin_context()`。
  - resume 请求命中同一 selected worker 时回传 `context_class_gpu_hit=1.0`。
  - `pause_age_s` 超过 `context_offload_after_s` 时可触发 `context_class_cpu_reload=1.0`。
  - 未存在/过期/淘汰的 rid 可触发 `context_class_full_prefill=1.0`。
  - `selected_backend_affinity_hit=1.0` 表示 `last_backend_id == selected_backend_id`。
  - `resume_enqueue_ts` / `resume_dispatch_ts` / `resume_queue_wait_s` 可从 route meta 回传到 response。
- Gateway preferred denominator smoke：
  - Prometheus metric 中 `smg_cache_aware_policy_branch_total{branch="preferred_present"}` 增长。
- Context lifecycle smoke：
  - 验证 `pin -> gpu_hit -> offload -> cpu_reload -> reload -> unpin -> full_prefill` 流程和 reload metric。

---

## 7. 使用注意事项

当前机器上的 `30000` gateway 已使用新构建的 binary 重启。若后续再次修改 gateway，需要重新构建并用新 binary 启动：

```bash
/export/xxl/xxl_sglang/sgl-model-gateway/target/release/sgl-model-gateway \
  --policy cache_aware \
  --health-check-endpoint /health_generate_original
```

如果要启用 soft preferred gating，可以在启动 gateway 前设置环境变量，例如：

```bash
export SMG_ENABLE_PREFERRED_OVERRIDE=true
export SMG_PREFERRED_LOAD_MARGIN=8
export SMG_PREFERRED_MAX_PAUSE_AGE_S=30
```

实验时建议同时观察：

- ROLL TensorBoard：
  - `resume_affinity_hit_rate`
  - `selected_backend_affinity_hit`
  - `selected_backend_migration`
  - `resume_migration_rate`
  - `resume_queue_wait_*`
  - `external_wait_*`
  - `resume_infer_latency_*`
  - `context_class_gpu_hit`
  - `context_class_cpu_reload`
  - `context_class_full_prefill`
- ROLL router metrics：
  - `scheduler/router/selected_backend_affinity_hit_rate`
  - `scheduler/router/selected_backend_migration_rate`
  - `scheduler/router/context/*`
  - `scheduler/router/context_class/*`
- Gateway Prometheus：
  - `smg_cache_aware_policy_branch_total{branch="preferred_present"}`
  - `smg_cache_aware_policy_branch_total{branch="preferred_hit"}`
  - `preferred_skip_overloaded`
  - `preferred_skip_old_pause`
  - `preferred_miss_unhealthy`
  - `preferred_miss_not_found`

---

## 8. 后续建议

下一步真实实验建议：

1. 重启真实 SGLang worker，确认 `/workers` 中 worker healthy。
2. 用新构建的 `sgl-model-gateway` binary 重启 gateway。
3. 跑一个小 batch Form-B smoke，确认：
   - `_last_backend_id` 持续更新。
   - resume 请求稳定带 `X-ROLL-Preferred-Worker-Url`。
   - gateway `preferred_present` / `preferred_hit` 增长。
   - TensorBoard 中 `resume_queue_wait_s` / `external_wait_s` / `resume_infer_latency_s` / `context_class_*` 出现。
4. 使用 `selected_backend_affinity_hit_rate` 判断 Form-B 真实 worker 亲和命中率；不要用 sentinel `resume_affinity_hit_rate` 解释 gateway selected worker。
5. 设计真实后端 API：`pin_context` / `unpin_context` / `offload_context` / `reload_context`。没有这一步，结论应限定为“resume-aware scheduling + 控制面 context 分类”，不能写成“真实 KV/context 保留”。
6. 再进行 baseline / resume-aware / Form-B ablation，对比 TPS、queue wait、migration、selected-backend affinity、preferred hit、context class 和 success reward。
