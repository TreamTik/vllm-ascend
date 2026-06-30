# Speculative Token Tree Performance Design

## Goal

在保持 `speculative_token_tree` 当前端到端正确性的前提下，把 Ascend 实现从
`correctness-first reference path` 演进到可用于生产形态的 `fast path`。

本设计的核心不是先做零散微优化，而是优先清理为了验证正确性而引入的结构性慢路径。

## Current State

当前正确性已经收口：

- `Qwen3-32B-W8A8 + eagle3 + speculative_token_tree`
- 长文本 `32B eager + async_scheduling=False`
- baseline/tree `108` token 已全量一致
- 历史晚步数 root-row 漂移点
  `slot 256 / token 9370 / pos 128`
  已被修复

当前性能仍明显落后：

- 正确性对齐后的重型 trace 实测：
  - baseline `13.77s`
  - tree `162.71s`
- 较轻量历史实测：
  - baseline `10.06s`
  - tree `28.71s`

这说明：

- debug trace 会明显放大总耗时
- 但即使不看最重 trace，tree 当前实现仍有结构性慢点

## Non-Goals

- 本轮设计不以优化 `eager + noasync` 验证链路本身为主目标
- 不先围绕 acceptance 做算法调优
- 不为了暂时提速而移除当前 reference correctness path

## Approaches

### Approach 1: 继续在当前 exact path 上做微优化

做法：

- 优化 Python 循环
- 缓存若干 mask / offsets
- 减少 clone / tensor 创建

优点：

- 风险小
- 见效快

缺点：

- 只能缓解，不会消除主瓶颈
- 当前最大开销来自 reference path 的结构性设计，不是小粒度实现细节

结论：

- 不作为主路线

### Approach 2: 分离 reference path 与 production fast path

做法：

- 保留当前 exact/reference path 作为调试与正确性基线
- 新建 production fast path，专门替代 correctness-first 妥协实现
- 后续所有性能工作都优先让 fast path 收益传导到真实服务形态

优点：

- 路线最清晰
- 不会把 debug/reference 逻辑长期背在生产路径上
- 便于做逐步替换和 A/B 正确性验证

缺点：

- 第一轮改动面较大
- 需要明确 path dispatch 和回退策略

结论：

- 推荐

### Approach 3: 先只优化 acceptance

做法：

- 优先调整 tree 结构、verifier、draft 行为
- 提升 accepted / drafted 比例

优点：

- 如果 acceptance 提升明显，理论收益可能很大

缺点：

- 当前主算子路径本身已经过重
- 在底座成本没降下来之前，先做 acceptance 容易治标不治本

结论：

- 放到第二阶段

## Recommended Design

采用 `Approach 2`。

整体结构分成两层：

- `reference path`
  - 保留当前 correctness-first 路径
  - 用于 trace、对账、回归验证
- `production fast path`
  - 面向真实吞吐
  - 逐步移除 dense gather、rowwise kernel launch、row0 二次计算、频繁 metadata 重建

### Why The Current Path Is Slow

当前最明确的 correctness-first 妥协有四类。

#### 1. Per-layer dense KV gather

代码位置：

- `vllm_ascend/attention/tree_attention.py::gather_paged_kv_to_ragged`

问题：

- 每层都把 paged KV 按 request gather 成 dense ragged KV
- prefix 越长，重复搬运越重
- 这属于结构性开销，不是常量级小损耗

#### 2. Python rowwise exact attention

代码位置：

- `vllm_ascend/attention/tree_attention.py::run_exact_tree_attention`

问题：

- 按 request 循环
- vendor path 下再按 speculative row 循环
- 每个 row 都单独：
  - 组装 visible KV
  - 调一次 `npu_fused_infer_attention_score`

这本质上是 reference implementation，不适合作为生产路径。

#### 3. Row0 double compute

代码位置：

- `vllm_ascend/attention/attention_v1.py::forward_tree_exact_attention`

问题：

- 整棵树先算一次 exact attention
- row0 再按 baseline decode path 重算一次覆写输出

这个设计帮助了正确性收口，但对生产吞吐代价明显。

#### 4. Per-step metadata rebuild

代码位置：

- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/spec_decode/tree_first_pass.py`
- `vllm_ascend/spec_decode/tree_execution.py`

问题：

- 每步重建 tree mask
- 每步 `mask.clone()`
- 每步构造 depth offsets tensor
- first-pass 多次 `index_select + cpu().clone()`
- accepted-path slot copy 每次新建 block/offset tensors

这些不是最大头，但频率高，会叠加吞吐损失。

## Implementation Plan

### Phase A: Split Reference And Fast Paths

目标：

- 明确 reference path 和 fast path 的职责边界

改动方向：

- 在 `attention_v1.py` 中把当前 tree 执行路径重命名或拆分为：
  - `forward_tree_reference_attention`
  - `forward_tree_fast_attention`
- 保留 reference path 的现状，作为 correctness fallback
- fast path 单独 dispatch，不再混在 reference 逻辑里演进

预期收益：

- 后续每轮性能优化不会继续污染 reference path

### Phase B: Remove The Biggest Correctness-First Compromises

目标：

- 先打掉最重的结构性慢点

优先顺序：

1. 去掉 row0 二次 attention 计算
2. 去掉 Python rowwise exact execution
3. 尽量避免 per-layer full dense KV gather

设计要求：

- fast path 必须在一次 tree execution 中同时产出 row0 和 speculative rows 输出
- 不允许再用“整棵树算完再单独覆写 row0”的方式维持正确性
- fast path 要尽量直接消费 paged KV 或更轻量的 request-local gathered KV

实现建议：

- 若 Ascend 现有 fused attention 无法直接表达树分支可见性：
  - 新增 tree fast kernel/routine
  - 允许使用新的专用实现，而不是继续拿 rowwise reference 拼出来

### Phase C: Cache High-Frequency Metadata

目标：

- 消除每步重复构造的小热点

改动方向：

- 缓存 runtime 不变量：
  - tree target mask
  - depth offsets
  - execution index mapping
  - row offsets
- first-pass 尽量复用预分配 tensor，减少：
  - `copy.copy`
  - `cpu().clone()`
  - 重复 `index_select`
- accepted-path slot copy 预分配 block/offset buffer，避免反复创建临时 tensor

### Phase D: Evaluate Acceptance As A Separate Lever

目标：

- 在底座开销降低后，再看 speculative 收益是否足够

现状：

- 最近正确性对齐轮次里：
  - `draft_tokens = 455`
  - `accepted_tokens = 16`
  - `acceptance_per_pos[0] ≈ 0.176`

结论：

- acceptance 偏低
- 但必须等 fast path 成型后再判断是不是值得继续做 acceptance 调优

## Files Expected To Change

第一轮大概率会改这些位置：

- `vllm_ascend/attention/attention_v1.py`
- `vllm_ascend/attention/tree_attention.py`
- `vllm_ascend/spec_decode/tree_execution.py`
- `vllm_ascend/spec_decode/tree_first_pass.py`
- `vllm_ascend/worker/model_runner_v1.py`

可能新增：

- `vllm_ascend/attention/tree_attention_fast.py`
  - 如果 fast path 需要与 reference path 物理拆文件

## Validation Strategy

每一轮优化后都固定做三层验证。

### 1. Correctness Gate

- 关闭重型 trace
- 跑当前 32B 长文本 case
- baseline/tree token 必须一致

### 2. Production-Oriented Performance Gate

- 运行更接近真实服务形态的配置
- 观察：
  - 首 token 延迟
  - 生成吞吐
  - accepted / drafted
  - 是否仍然退回 reference path

### 3. Reference Regression Gate

- 如果优化后正确性异常
- 才回退到 reference/debug path 做精细对账

## Success Criteria

- fast path 不再依赖当前 correctness-first 的三大妥协：
  - dense full-KV gather
  - Python rowwise exact execution
  - row0 二次 attention 覆写
- 32B 长文本正确性保持不退化
- tree 相对 baseline 的生成耗时显著收敛
- 生产形态 benchmark 的收益明显优于当前 reference-derived 实现

## Risks

- Ascend 现有 attention 接口未必足够表达树状可见性，可能需要新的专用 fast implementation
- row0 一次性并入 tree fast path 后，之前已修掉的 kernel-path mismatch 可能重新出现
- 只优化算子而不改善 acceptance，最终收益可能仍受限

## Final Recommendation

先不要做零散微优化。

下一步应先把当前实现明确拆成 `reference path` 和 `production fast path`，
然后优先消灭以下三项 correctness-first 妥协：

1. row0 二次计算
2. Python rowwise exact attention
3. per-layer dense KV gather

等这三项收掉后，再处理 metadata 小热点和 acceptance。
