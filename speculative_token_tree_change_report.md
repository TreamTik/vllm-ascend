# speculative_token_tree 更改报告

## 1. 报告目的

这份报告对应当前两个工作树：

- `/root/worktrees/vllm-spec-tree-v023`
- `/root/worktrees/vllm-ascend-clean`

并且对应你刚刚要求的状态：

- 两个仓库都已经 `reset --mixed` 回到 tree 工作开始前的提交
- tree 相关代码改动全部保留在工作区
- 没有删除代码
- 这样可以直接看出“从 0 到现在到底改了哪些东西”

本报告的目标是把这些改动按模块拆开，说明：

1. 上游 `v023` 到底改了什么
2. Ascend 侧到底改了什么
3. 哪些改动是功能主线
4. 哪些改动是验证 / 调试 / 设计文档
5. 当前真实现状是什么
6. 还有哪些没有做完

## 2. 当前仓库基线

### 2.1 `/root/worktrees/vllm-spec-tree-v023`

当前 `HEAD` 已回到：

- `967c5c3bc` `compat-v023`

tree 相关改动都在工作区，主要包括：

- `vllm/config/speculative.py`
- `vllm/v1/spec_decode/llm_base_proposer.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `tests/test_config.py`
- `tests/engine/test_arg_utils.py`
- `tests/v1/spec_decode/test_speculative_token_tree.py`
- `docs/features/speculative_decoding/README.md`
- `docs/superpowers/...` 设计/计划文档

### 2.2 `/root/worktrees/vllm-ascend-clean`

当前 `HEAD` 已回到：

- `5d4964c` `Add tower connector LoRA validation tests`

tree 相关改动都在工作区，主要包括三类：

1. 核心实现
2. 测试与用户文档
3. 调试 / 设计 / 性能验证资产

核心实现文件主要是：

- `vllm_ascend/spec_decode/tree_runtime.py`
- `vllm_ascend/spec_decode/tree_execution.py`
- `vllm_ascend/spec_decode/tree_first_pass.py`
- `vllm_ascend/spec_decode/tree_verifier.py`
- `vllm_ascend/spec_decode/llm_base_proposer.py`
- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/sample/rejection_sampler.py`
- `vllm_ascend/attention/attention_v1.py`
- `vllm_ascend/attention/tree_attention.py`
- `vllm_ascend/attention/utils.py`
- `vllm_ascend/patch/worker/patch_tree_spec_decode.py`

辅助联动改动包括：

- `vllm_ascend/attention/context_parallel/attention_cp.py`
- `vllm_ascend/attention/context_parallel/dsa_cp.py`
- `vllm_ascend/attention/context_parallel/sfa_cp.py`
- `vllm_ascend/attention/dsa_v1.py`
- `vllm_ascend/attention/mla_v1.py`
- `vllm_ascend/ascend_forward_context.py`

调试 / 验证性质改动包括：

- `vllm_ascend/ops/linear_op.py`
- `vllm_ascend/patch/worker/patch_qwen3vl.py`
- `run_qwen32_tree_compare.py`
- `docs/source/developer_guide/Design_Documents/speculative_token_tree_debug_log.md`
- `task_plan.md`
- `findings.md`
- `progress.md`
- `docs/superpowers/...`

## 3. 上游 v023 改了什么

上游这部分的定位非常明确：

- 不是实现 tree 执行
- 而是把 `speculative_token_tree` 变成正式公共配置契约

换句话说：

> 上游做的是“接口、校验、CLI 透传、运行时明确拒绝”，不是“真正执行树状 speculative decoding”。

### 3.1 `vllm/config/speculative.py`

这是上游最核心的配置改动。

新增内容：

- 新增字段 `speculative_token_tree: str | None`
- 新增 `_parse_speculative_token_tree()`
- 新增 `parsed_speculative_token_tree` 属性
- 新增 `use_speculative_token_tree()` 方法

它定义了 tree 配置的公共语义：

- 值仍然是字符串
- 必须能被 `ast.literal_eval` 解析
- 解析结果必须是非字符串 iterable
- 每个 path 都必须非空
- path 必须唯一
- path 必须可排序/可比较
- 非根 path 必须 parent-closed

额外行为：

- 如果只给 `speculative_token_tree`，自动推导 `num_speculative_tokens = len(tree)`
- 如果同时给 `num_speculative_tokens`，两者必须一致

配置级限制：

- 只允许用于 `eagle / eagle3 / mtp / dflash`

这一步的意义是：

- 先把上游公共 API 稳定下来
- 让下游实现不需要再发明一套私有配置格式

### 3.2 `tests/test_config.py`

这一组测试是对配置契约本身的覆盖。

新增测试点：

- 能否从 tree 自动推导 `num_speculative_tokens`
- 缺父节点的树是否报错
- `num_speculative_tokens` 与 tree node 数量不一致是否报错
- 不支持的方法是否报错

这部分的作用是：

- 证明上游对 tree 配置不是口头约定
- 而是已经进入配置校验和测试体系

### 3.3 `tests/engine/test_arg_utils.py`

这一组测试补的是 CLI 入口层。

新增测试点：

- `--speculative-config` 子字段里能接受 `speculative_token_tree`
- 不存在顶层 CLI 参数 `--speculative-token-tree`

这部分非常重要，因为它明确了用户接口：

- tree 只能作为 `speculative_config` 的子字段存在
- 不是单独顶层参数

### 3.4 `vllm/v1/spec_decode/llm_base_proposer.py`

上游在 proposer 基类里增加了两件事：

- 读取 `parsed_speculative_token_tree`
- 在 tree mode 下显式抛 `NotImplementedError`

也就是说，上游在这里明确宣告：

- “我认识这个配置”
- “但我没有实现通用 tree 执行”

这一步非常关键，因为它给下游留下了干净边界：

- 上游 contract 合法
- 上游 built-in runtime 暂时拒绝
- 下游可以接着实现自己的 backend

### 3.5 `vllm/v1/worker/gpu_model_runner.py`

这部分本质上是结构清理，不是功能性 tree 执行。

改动点：

- 把原来 `_set_up_drafter()` 里一大段 method dispatch
  提取成 `_create_speculative_drafter()`

作用：

- 为下游 patch 某些 method 的 proposer 选择提供了更明确的注入点

这不是 tree 功能本身，但它让下游更好 patch。

### 3.6 `tests/v1/spec_decode/test_speculative_token_tree.py`

这组测试非常短，但语义很重要：

- tree mode 下，调用 `_raise_if_speculative_token_tree_unimplemented()` 必须报错
- 非 tree mode 下必须 no-op

也就是说，上游运行时测试覆盖的是：

- “tree 目前不执行” 这个事实

### 3.7 `docs/features/speculative_decoding/README.md`

文档层把 tree 正式写进上游 spec decode 文档：

- 新增 `speculative_token_tree` 字段说明
- 写明仅支持作为 `speculative_config` 子字段
- 写明当前只是 validated contract
- 明确说明 built-in proposer 暂不执行 tree runtime

### 3.8 上游当前没有做的东西

这一点要讲清楚。

`/root/worktrees/vllm-spec-tree-v023` 当前没有实现：

- tree runtime
- tree draft execution
- tree target-side attention mask patching
- tree verifier
- tree accepted-path reconstruction
- tree attention backend

所以如果只看上游：

- `speculative_token_tree` 是“配置和公共接口已存在”
- 但不是“功能已经通用可跑”

## 4. Ascend 侧改了什么

Ascend 侧做的不是 contract，而是真正的执行实现。

一句话总结：

> Ascend 把上游只是“认识 tree 配置”的状态，推进成了“eager + eagle/eagle3 + greedy verifier + 自定义 tree attention 后端”的实际可运行实现。

## 5. Ascend 功能主线改动

### 5.1 入口 patch：`vllm_ascend/patch/worker/patch_tree_spec_decode.py`

这个 patch 的作用是：

- 当 `speculative_token_tree` 开启
- 且 method 是 `eagle/eagle3`
- 则不走上游默认 built-in proposer
- 而是强制切到 `AscendEagleProposer`

意义：

- 这是把上游“tree mode runtime rejection”
  接管成 Ascend 本地实现的入口

同时：

- `vllm_ascend/patch/worker/__init__.py`
  增加了这个 patch 的注册导入

### 5.2 tree 运行时结构：`tree_runtime.py`

新增了 `SpecTreeRuntime` 和 `build_spec_tree_runtime()`。

这部分负责把：

- `((0,), (1,), (0, 0), (0, 1), (1, 0))`

这类 path 表示法，变成运行时结构：

- `node_count`
- `parent_index`
- `depth`
- `children`
- `nodes_by_depth`
- `leaf_indices`

这是 tree 执行的第一层基石。

没有它，后面所有：

- tree mask
- tree execution order
- verifier 深度匹配
- first-pass frontier 重建

都没法稳定表达。

### 5.3 tree 执行元数据：`tree_execution.py`

新增了 `AscendTreeSpecDecodeMetadata` 和若干 helper：

- `build_tree_query_position_offsets()`
- `build_tree_query_slot_offsets()`
- `build_tree_query_depth_offsets()`
- `build_tree_execution_runtime_node_indices()`
- `build_tree_attn_mask()`
- `build_tree_target_attn_mask()`
- `build_tree_step_layout()`
- `build_tree_spec_decode_metadata()`

这部分负责回答几个核心问题：

1. 一个 tree batch 在 verifier target 侧应该怎么铺成 rows
2. runtime node 顺序和 execution row 顺序如何映射
3. 哪些节点在 attention 上可见
4. 同一 request 在 batch 中的 node offset / row offset 是什么

这里的关键点是：

- runtime order 和 execution order 被明确分开了
- 因为树不一定按 BFS/自然 index 顺序消费

### 5.4 verifier：`tree_verifier.py`

新增内容：

- `AcceptedTreePath`
- `select_tree_accept_paths()`
- `pack_tree_output_tokens()`

这部分的职责是：

- 用 target logits 逐层验证 draft tree
- 找出每个 request 当前最深的可接受路径
- 记录：
  - 接受了哪些 runtime nodes
  - 对应 token_ids
  - bonus token
  - terminal row index

然后再把接受结果 pack 成现有 sampler 输出兼容的 tensor 形状。

这是 tree 模式下 rejection/verification 的核心。

### 5.5 first-pass 重建：`tree_first_pass.py`

新增 `build_tree_first_pass_inputs()`。

它解决的不是“普通 speculative decode 的下一步”，而是 tree 模式的下一轮 root 重建：

- 不能把上一轮整个 target tree batch 原样喂回 drafter
- 否则 sibling rows 会被错误延续

当前实现的做法是：

- 只保留 verifier 选中的 terminal rows
- `next_token_ids` 作为新的 root token
- `positions / hidden_states / slot_mapping`
  只从 terminal rows 取
- 重建一个压缩后的 `CommonAttentionMetadata`

它的语义是：

- 下一轮 draft 的根，只从 accepted frontier 出发
- 不是把旧树整棵树都带过去

### 5.6 Ascend proposer 基类：`llm_base_proposer.py`

这是 Ascend tree 实现最关键的接入点之一。

主要改动分为五层。

#### 5.6.1 接管上游 tree rejection

上游基类在 tree mode 会直接报错。

Ascend 这里做的是：

- 覆盖 `_raise_if_speculative_token_tree_unimplemented()`
- 让它在 Ascend 本地不报错

这样本地实现才有机会继续。

#### 5.6.2 初始化 tree runtime

新增 `_get_upstream_parsed_speculative_token_tree()` 和 `_init_speculative_token_tree_runtime()`。

逻辑是：

- 从上游 `SpeculativeConfig` 里读取 tree
- 如果没 tree，就不启用 tree runtime
- 如果有 tree，则施加本地限制

当前本地限制非常明确：

1. 只支持 `eagle/eagle3`
2. 只支持 eager
3. graph/fullgraph 不支持

然后：

- 调用 `build_spec_tree_runtime()`

#### 5.6.3 draft 阶段走 tree 语义

在 `_run_tree_draft()` 和相关调用链中，draft 不再只是线性链式生成。

核心变化是：

- 按 tree 的 depth 分层推进
- 第一轮用 target 侧 patched tree mask
- 按 execution order 产生 draft token
- 再通过 runtime/execution mapping 对齐

这部分就是让 drafter 真正生成“树”，而不是“链”。

#### 5.6.4 first-pass 输入重建

在 `set_inputs_first_pass()` 中增加 tree 分支：

- 一旦 `_tree_runtime is not None`
- 就调用 `build_tree_first_pass_inputs()`

这一步把 verifier 的 terminal rows 收束成下一轮 drafter 的输入。

#### 5.6.5 当前限制被显式写死

当前 Ascend proposer 明确拒绝：

- 非 `eagle/eagle3`
- 非 eager
- `lmhead TP`

所以 tree mode 目前不是全方法、全运行模式通吃。

### 5.7 model runner：`worker/model_runner_v1.py`

这是第二个主战场。

tree mode 在 runner 层主要做了四件事。

#### 5.7.1 新增 tree metadata 构建

新增 `_calc_tree_spec_decode_metadata()`。

作用：

- 基于 `_tree_runtime` 和 batch size
  构造 `AscendTreeSpecDecodeMetadata`
- 计算：
  - `request_node_offsets`
  - `request_row_offsets`
  - `execution_runtime_node_indices`
  - `runtime_node_to_execution_index`
  - `draft_token_ids`
  - `target_logits_indices`
  - `bonus_logits_indices`

这一步是 target verifier 能理解 tree batch 的基础。

#### 5.7.2 target 侧位置和 mask patch

新增：

- `_patch_tree_spec_decode_positions()`
- `_patch_tree_spec_decode_attn_metadata()`

作用：

- 按 tree depth offsets 重写 positions
- 给 attention metadata 覆盖 tree target mask
- 给 fast attention 路径注入 shared plan

这一层非常重要，因为 tree batch 不是普通 decode batch：

- 普通 decode 假设线性 causal extension
- tree batch 需要自定义 ancestor-visible mask

#### 5.7.3 accepted-path KV materialization

新增：

- `_plan_tree_accepted_path_slot_copies()`
- `_copy_tree_accepted_path_kv_slots()`
- `_materialize_tree_accepted_path_kv()`

它解决的问题是：

- verifier 接受了一条 branch 后
- 下一轮 root 不能继续用 branch 上的物理 speculative slot
- 必须把 accepted path 的 KV 写回 canonical prefix slot

这部分是晚期 correctness 修复中的一个关键组件。

#### 5.7.4 runner 状态暴露

新增 runner 字段：

- `tree_terminal_row_indices`
- `tree_accepted_paths`

让：

- rejection sampler
- proposer first-pass
- accepted-path materialization

可以通过 runner 共享 tree 当前轮接受结果。

### 5.8 sampler / verifier 接入：`sample/rejection_sampler.py`

当前 tree 模式没有沿用普通 speculative decode 的 rejection_sample 逻辑。

改动是：

- 在 `AscendRejectionSampler.forward()` 中增加
  `AscendTreeSpecDecodeMetadata` 分支

逻辑是：

1. 先取 target logits
2. 应用 logits processors
3. 当前只允许 greedy
4. `argmax` 得到 target token ids
5. 调 `select_tree_accept_paths()`
6. 记录 `runner.tree_terminal_row_indices`
7. 记录 `runner.tree_accepted_paths`
8. 用 `pack_tree_output_tokens()` 返回 sampler 输出

这一段是 tree verifier 真正接到 sampling 框架里的位置。

当前限制也非常清楚：

- tree verifier 只支持 greedy
- 不支持非 greedy / 通用 rejection sampling 语义

### 5.9 common attention metadata：`attention/utils.py`

这里做了两件关键小改动：

1. 在 `AscendCommonAttentionMetadata` 上增加
   `tree_spec_decode_enabled`
2. 新增 `get_effective_decode_threshold()`

这个改动表面不大，但意义很大：

- tree target verification 是多 query row 的 batch
- 如果仍把它当短 decode 优化路径处理，
  很多 backend 会绕开 tree mask

所以这里通过：

- tree mode 强制有效 decode threshold = 1

把 tree batch 压回 masked/prefill-style 路径，而不是错误走普通 short decode fast path。

### 5.10 多个 attention metadata builder 的联动

这些文件都做了同一类修改：

- `attention/context_parallel/attention_cp.py`
- `attention/context_parallel/dsa_cp.py`
- `attention/context_parallel/sfa_cp.py`
- `attention/dsa_v1.py`
- `attention/mla_v1.py`

共同点：

- `split_decodes_and_prefills()` 的 `decode_threshold`
  改为使用 `get_effective_decode_threshold()`

作用：

- 保证 tree batch 不会被误判成普通 decode-only 快路径

这类改动不是 tree 入口，但属于“必须联动，不然 tree attention 语义会被后端优化破坏”。

### 5.11 attention 主后端：`attention_v1.py`

这是本次改动体量最大的文件之一，也是最核心的文件之一。

它承担了两类任务：

1. 先把 tree attention 正确跑通
2. 再逐轮做性能优化

#### 5.11.1 新增 tree attention 分流

新增了一整套 tree 相关分流：

- `_should_use_tree_exact_attention()`
- `forward_tree_exact_attention()`
- `forward_tree_fast_attention()`

并且通过环境变量控制实现分支：

- `reference`
- `batched_legacy`
- `batched_split`
- `batched`

当前最终语义收敛为：

- `reference`
  - 最保守的逐行 exact/debug 路径
- `batched_legacy`
  - 旧的 `row0 override + non-root batched`
- `batched_split`
  - 试验中的 split path
- `batched`
  - 当前主线 unified all-row exact fast path

#### 5.11.2 row0 路径修正

tree 里 root row 一开始不是复用 baseline 同一路径。

后面这里补了：

- `_build_tree_root_decode_metadata()`
- `tree_row0_paged`
- `tree_row0_fused`

核心思想是：

- tree row0 必须尽量复用 baseline decode 的同一 dispatch 逻辑
- 不能一边 baseline 走 fused，一边 tree row0 走 paged

这部分是后期 correctness 收口的重要修复。

#### 5.11.3 fast path 的逐轮演进

这一文件里同时保留了几代实现：

- 旧的 row0 override 路径
- non-root batched path
- split path
- unified all-row path

也就是说，这里不仅有最终主线，也保留了中间实验实现，方便回退和对比。

### 5.12 新 attention helper：`attention/tree_attention.py`

这是第二个核心大文件。

它把 tree attention 的公共 helper 集中拆出来了。

主要内容包括：

- `TreePagedPlan`
- `TreeSplitPlan`
- `build_tree_paged_plan()`
- `build_tree_split_plan()`
- `run_exact_tree_attention()`
- `run_fast_tree_attention()`
- `run_fast_tree_attention_from_paged_cache()`
- `run_split_current_tree_attention()`
- `merge_split_tree_attention_outputs()`
- `resolve_tree_fast_impl()`

再加上：

- visible-row cache
- profile dump
- debug dump

#### 5.12.1 先后做过哪些性能路线

这个文件里可以看出性能优化经历了几轮：

1. 先有 `reference` exact path
2. 再有 `batched_legacy`
3. 再尝试 `split`
4. 最后收敛到 `batched` unified all-row path

#### 5.12.2 当前主线和实验路径的区别

当前可认为：

- `batched`
  是主线
- `batched_split`
  仍是实验实现

因为 split path 虽然更快，但数值漂移没有完全消掉，不能当正确主线。

## 6. 调试与验证类改动

这一部分要和主线功能分开看。

### 6.1 `ascend_forward_context.py`

新增：

- `positions`

进入 forward context。

作用：

- 给调试 instrumentation 提供更多上下文

### 6.2 `ops/linear_op.py`

这部分基本是 instrumentation：

- 增加 isolated output probe
- 支持 `VLLM_ASCEND_LINEAR_ISO_TRACE_*`

作用：

- 排查“同一行单独算 vs batched 算”是否在 linear 层引入差异

它是调试 correctness 时用的，不是 tree 功能必需主线。

### 6.3 `patch_qwen3vl.py`

这里加了大量 Qwen3/Qwen3-VL block trace / attn trace。

作用：

- 抓 layer/block 级 hidden、attn output、residual 漂移

这是定位 late-step residual issue 时用的调试手段。

它不属于 tree 功能逻辑本体，而属于“排查为什么不对齐”的工具代码。

### 6.4 `run_qwen32_tree_compare.py`

这是一个真实模型 compare 脚本。

用途：

- 拉起 baseline 与 tree
- 在同一 prompt 上对比：
  - token ids
  - 文本
  - 生成时长
  - metrics

它属于验证资产，不是库代码的一部分。

### 6.5 设计/调试文档

这些文件更多是工作记录和设计资产：

- `speculative_decode_upstream_to_ascend.md`
- `docs/source/developer_guide/Design_Documents/speculative_token_tree_debug_log.md`
- `docs/superpowers/plans/...`
- `docs/superpowers/specs/...`
- `task_plan.md`
- `findings.md`
- `progress.md`

它们的价值在于：

- 记录设计思路
- 记录排查路径
- 记录 benchmark 结论

但它们不属于最终产品运行主逻辑。

## 7. 测试和文档改动

### 7.1 上游 v023 测试

新增：

- `tests/v1/spec_decode/test_speculative_token_tree.py`
- `tests/test_config.py` 新 tree 测试
- `tests/engine/test_arg_utils.py` 新 CLI 测试

覆盖的是：

- config
- CLI
- runtime rejection hook

### 7.2 Ascend 单元测试

新增 / 修改：

- `tests/ut/spec_decode/test_tree_runtime.py`
- `tests/ut/spec_decode/test_tree_execution.py`
- `tests/ut/spec_decode/test_tree_first_pass.py`
- `tests/ut/spec_decode/test_tree_verifier.py`
- `tests/ut/spec_decode/a2/test_tree_eagle_proposer.py`
- `tests/ut/worker/a2/test_model_runner_tree_spec_decode.py`
- `tests/ut/patch/worker/test_patch_tree_spec_decode.py`
- `tests/ut/sample/test_rejection_sampler.py`

这些测试分别覆盖：

- tree runtime 构造
- execution/runtime order 映射
- attention mask 语义
- verifier deepest-path 选择
- first-pass terminal-row 收束
- runner metadata 和 slot copy 规划
- patch 是否接管 proposer 选择
- rejection sampler tree 分支输出形状

### 7.3 用户文档

新增 / 修改：

- 上游：
  - `docs/features/speculative_decoding/README.md`
- Ascend：
  - `docs/source/user_guide/feature_guide/speculative_decoding.md`

Ascend 文档里额外写清楚了本地限制：

1. 只支持 `eagle/eagle3`
2. 只支持 eager
3. `num_speculative_tokens` 必须和 tree node 数量一致
4. `lmhead TP` 不支持
5. verifier 只支持 greedy

## 8. 当前真实现状

这一部分不是“理想状态”，而是目前代码实际达到的状态。

### 8.1 上游 v023 现状

上游已经做到：

- `speculative_token_tree` 进入正式公共配置
- 配置和 CLI 都可识别
- 配置语义和限制有测试
- 运行时会明确拒绝 built-in tree 执行

上游还没有做到：

- built-in 通用 tree runtime

### 8.2 Ascend 现状

Ascend 已经做到：

- 真正接管 tree mode proposer
- 建立 tree runtime
- 在 draft 阶段按 tree 执行
- 在 verifier target 阶段使用 tree attention mask
- 实现 tree verifier / accepted-path 选择
- 实现 first-pass frontier 收束
- 实现 accepted-path KV materialization
- 实现 tree attention backend
- 在真实 `32B + eagle3` case 上跑通 end-to-end correctness

### 8.3 correctness 现状

从已有验证记录看，当前已经达到：

- 在 `Qwen3-32B-W8A8 + RedHatAI/Qwen3-32B-speculator.eagle3`
- eager
- `async_scheduling=False`

这一条验证链上：

- baseline / tree 可做到输出一致

后期 attention 主线收敛后的典型结果包括：

- 默认 `batched` unified all-row 路径：
  - baseline `10.25s`
  - tree `10.62s`
  - `same_token_ids = true`
  - `same_text = true`

以及 prompt benchmark：

- 自然长文/说明文：
  - 常常仍慢于 baseline
- 强结构 JSON：
  - 已出现真实加速
- 一个典型长 JSON case：
  - baseline `63.73s`
  - tree `50.37s`
  - `speedup = 1.265x`
  - 输出一致

### 8.4 性能现状

当前结论不是“tree 普遍更快”，而是：

- attention 侧大头已经基本打穿
- 强结构、长输出、schema 稳定的 prompt 上，tree 已能转化成真实加速
- 自然长文、自由生成 prompt 上，tree 仍常被 proposer/verify 额外成本抵消

换句话说，当前主要矛盾已经从：

- “tree attention 逻辑是否正确”

逐渐转向：

- “acceptance 是否足够高”
- “生产形态吞吐是否真的划算”

## 9. 还没有做完的东西

这里要区分“明确没做”和“做了实验但没有转正”。

### 9.1 上游没有做完

上游当前没做：

- built-in tree runtime
- built-in tree proposer execution
- built-in tree verifier
- built-in tree attention backend

所以：

- 上游只是 contract 完成
- 不是 feature runtime 完成

### 9.2 Ascend 当前明确没做的功能

#### 9.2.1 非 eager / graph / fullgraph

当前 tree mode 明确只支持 eager。

没做完：

- graph
- fullgraph
- 生产图构建后的稳定 tree runtime 方案

#### 9.2.2 非 greedy verifier

当前 `sample/rejection_sampler.py` tree 分支显式要求：

- `all_greedy = True`

没做完：

- 非 greedy tree verify
- 与普通 speculative decode rejection sampling 完整统一

#### 9.2.3 方法覆盖面

上游 contract 允许：

- `eagle`
- `eagle3`
- `mtp`
- `dflash`

但 Ascend 当前真正执行只支持：

- `eagle`
- `eagle3`

没做完：

- `mtp` tree execution
- `dflash` tree execution
- 其他方法的 tree runtime

#### 9.2.4 LM-head TP

当前代码里明确拒绝：

- `lmhead TP` 下的 tree mode

还没做。

#### 9.2.5 V2 路径

这轮实际 tree 执行主线在 V1。

当前没有对应的 V2 tree 实现主线改动落地到：

- `worker/v2/...`

因此不能说 V2 已经等价支持。

### 9.3 做了实验但没有转正的东西

#### 9.3.1 `batched_split`

这个路径已经做了实验实现，也验证过更快。

但问题是：

- 数值仍有 `1e-3 ~ 1e-2` 级漂移
- 足以让 token 序列提前分叉

所以当前状态是：

- 保留实验实现
- 不能当主线

#### 9.3.2 生产形态吞吐

当前验证主力还是：

- eager
- `async_scheduling=False`

还没有把 tree runtime 在真实服务形态下完全证明：

- graph
- async scheduling
- 长驻 server 吞吐

#### 9.3.3 acceptance 自适应

目前树形还是固定的：

- `[(0,), (1,), (0, 0), (0, 1), (1, 0)]`

还没有做：

- 自适应树深度/宽度
- 按 prompt / acceptance 动态切树

这意味着：

- 在自然语言自由生成里，很多 draft row 可能白算了

### 9.4 调试代码清理没有完成

当前工作区里除了功能代码，还有不少调试和工作记录资产：

- `patch_qwen3vl.py` 大量 block trace
- `linear_op.py` isolated trace
- `task_plan.md / findings.md / progress.md`
- `docs/superpowers/...`

这些并不一定都应该最终保留进主线提交。

所以从“最终提交质量”角度看，还没做完：

- 调试代码和主线功能代码的进一步清理/分离
- 文档与工作笔记的裁剪

## 10. 哪些改动最关键

如果只从“tree 功能真正由无到有”的角度看，最关键的文件是这些：

### 上游契约层

- `vllm/config/speculative.py`
- `vllm/v1/spec_decode/llm_base_proposer.py`

### Ascend 执行主线

- `vllm_ascend/patch/worker/patch_tree_spec_decode.py`
- `vllm_ascend/spec_decode/tree_runtime.py`
- `vllm_ascend/spec_decode/tree_execution.py`
- `vllm_ascend/spec_decode/tree_first_pass.py`
- `vllm_ascend/spec_decode/tree_verifier.py`
- `vllm_ascend/spec_decode/llm_base_proposer.py`
- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/sample/rejection_sampler.py`
- `vllm_ascend/attention/attention_v1.py`
- `vllm_ascend/attention/tree_attention.py`

如果只读这些文件，已经能理解 80% 以上的 tree 实现。

## 11. 哪些文件主要是验证或辅助

这些不是 tree runtime 必需主线，但帮助很大：

- `tests/ut/...` 全套 tree 单测
- `run_qwen32_tree_compare.py`
- `docs/source/user_guide/feature_guide/speculative_decoding.md`
- `speculative_decode_upstream_to_ascend.md`
- `patch_qwen3vl.py`
- `linear_op.py`
- `task_plan.md`
- `findings.md`
- `progress.md`

## 12. 最终结论

这次改动的整体结果可以概括成三句话。

第一句：

> 上游 `v023` 完成了 `speculative_token_tree` 的公共配置契约，但没有完成 built-in tree 执行。

第二句：

> Ascend 侧已经把 tree mode 从“配置能识别”推进成了“eager + eagle/eagle3 + greedy verifier + 自定义 tree attention 后端”的实际可运行实现。

第三句：

> 当前 correctness 基本打通，attention 主瓶颈也已经大幅优化，但生产形态、非 greedy verifier、更多 method 覆盖、以及调试代码清理都还没有完成。

