# Speculative Token Tree Debug Log

## 目的

本文档用于持续记录 `speculative_token_tree` 在 Ascend 侧端到端调试过程中的“可疑点 / 假设 / 证据 / 结论 / 下一步验证”。

重点不是写成最终设计文档，而是避免调试过程中遗漏线索、重复猜测、或者把已经被证伪的怀疑点继续当作事实使用。

## 使用原则

1. 只要出现新的可疑点，先登记，再验证。
2. 每次验证都必须记录：
   - 验证动作
   - 观测结果
   - 对原假设的影响
3. 如果某个可疑点后续被证伪，不能只在脑子里推翻，必须回到本文档修改状态和结论。
4. 未验证的猜测必须明确标注为“待验证”，不能写成既成事实。
5. 如果某次代码修改只是试探性实验，也要在文档中登记，避免后续忘记当前代码已偏离哪个基线。
6. 调试优先级遵循 root-cause-first 原则：
   - 先查数据语义是否一致
   - 再查执行路径是否错位
   - 最后才考虑性能优化或 Triton 内核

## 当前目标

目标不是“先让它跑起来”，而是让 Ascend 的 tree speculative decoding 在真实 32B 模型上输出质量合理，且实现路径具备可维护性。

当前已知现象：

- baseline 输出稳定、基本合理。
- tree 模式可以跑通，但输出质量明显下降。
- 退化表现包括：
  - 复述
  - 奇怪措辞
  - 零散 markdown 符号（如 `**`）
- 首次 token 分歧不是第 0 个生成 token，而是大约第 7 个生成 token。

这个现象很重要，因为它说明：

- tree 路径不是从一开始就完全错；
- 至少最早几步 root proposal / verify / advance 是部分成立的；
- 问题更可能出现在“若干轮之后的语义累积偏移”，而不是单纯首 token 即错。

## 当前运行基线

### 工作目录

- 主调试仓：`/root/worktrees/vllm-ascend-clean`
- 上游参考仓：`/root/worktrees/vllm-spec-tree-v023`

### 模型与脚本

- base model:
  - `/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-32B-W8A8`
- draft model:
  - `/root/.cache/modelscope/hub/models/RedHatAI/Qwen3-32B-speculator.eagle3`
- 主验证脚本:
  - `/root/tmp_tree_eager_quick_check.py`
- 对比脚本:
  - `/root/worktrees/vllm-ascend-clean/run_qwen32_tree_compare.py`

### tree 配置

当前重点验证树：

```python
[(0,), (1,), (0, 0), (0, 1), (1, 0)]
```

### 当前验证模式

- eager mode
- greedy 路径
- 使用 `LLM.chat(...)`
- 已修正 prompt/chat-template 使用方式

## 2026-06-29 最新覆盖性更新

### 最新待验证假设 A：row0 fused rerun 当前卡在 metadata owner 选错，不是卡在 fused kernel 本身

为验证“baseline `decode_fused` 与 tree row0 override 的 kernel-path mismatch”
这一主怀疑点，tree exact path 已改成：

- row0 若满足 paged 条件，则走 `tree_row0_paged`
- 否则走 `tree_row0_fused`

第一次重跑的直接结果不是数值对账，而是运行时报错：

- `When attnMask is not provided, sparseMode must be 0`

这条报错本身有价值，因为它说明：

- row0 override 已经不再被硬编码锁死到 paged path
- fused path 确实被真正执行到了

随后补 `row0_metadata.attn_mask` 后，
又出现新的运行时错误：

- `AttributeError: 'AscendAttentionBackendImpl' object has no attribute 'attn_mask_builder'`

代码归属复核显示：

- `attn_mask_builder`
- `model_config`

当前都定义在 `AscendAttentionMetadataBuilder`，
而不是 `AscendAttentionBackendImpl`

因此当前新的工作假设是：

- row0 decode metadata 在 fused override 下
  不应在 backend impl 内重新访问 builder 私有状态
- 更合理的做法应当是：
  - 方案 A：复用已经构建好的 `attn_metadata.attn_mask`
  - 方案 B：在 backend impl 内显式调用
    `AttentionMaskBuilder(device)`，
    并使用 `self.vllm_config.model_config`
    现取一份标准 causal mask

当前更偏向方案 B，
因为 row0 override 的目标是尽量贴近 baseline 的
`decode_fused` 路径，
而原 `attn_metadata.attn_mask`
已经被 tree target 逻辑 patch 过

状态：

- 待验证

下一步：

- 先修 row0 metadata owner 错位
- 再重跑最小串行 trace
- 看 `tree_row0_fused` 是否能稳定产生日志并消除
  `slot 256 / token 9370 / pos 128`
  处的 `current_output = 6.103515625e-05` 漂移

最新实现状态补记：

- `_build_tree_root_decode_metadata()` 已改成：
  - 不再访问 `self.attn_mask_builder / self.model_config`
  - 改为在 backend impl 内使用
    `AttentionMaskBuilder(device)` +
    `self.vllm_config.model_config`
    构建标准 causal mask
  - 同时补齐 decode-only 所需的
    `query_start_loc / max_query_len / num_decodes` 等字段

这一步目前只完成了语法校验，
还没有做串行功能验证。

### 最新结论 B：owner 修复 + `tree_row0_fused` 已把长文本 case 拉回端到端一致

首轮串行功能验证已经完成：

- 脚本：
  - `/root/tmp_tree_eager_quick_check_noasync.py`
- 配置：
  - eager
  - `async_scheduling=False`
  - tree:
    `[(0,), (1,), (0, 0), (0, 1), (1, 0)]`

结果：

- baseline `108` tokens
- tree `108` tokens
- `/tmp/qwen32_tree_eager_quick_check_noasync.json`
  中两侧 `token_ids` 已完全一致

同轮 trace 还证明了：

- baseline root 路径：`decode_fused`
- tree root override 路径：`tree_row0_fused`

而不是：

- `tree_row0_paged`

这条结论非常关键，因为它强烈支持：

- 先前晚步数 residual drift 的主根因
  确实是 row0 override 的 kernel-path mismatch
- 当 tree root 行被强制走 paged path 时，
  即便可见输入面（`current_q/k/v`, `prefix_tail_k/v`）对齐，
  仍会在 `layer0 current_output`
  处出现 `6.103515625e-05` 级漂移
- 改成与 baseline dispatch 一致的 fused path 后，
  至少在当前长文本 case 上，
  该漂移已经足以从端到端结果里消失

当前未完事项：

- 本轮 `prefix-cache trace` 默认只保留 `32` 步
- 还没有直接覆盖旧定位点：
  `slot 256 / token 9370 / position 128`
- 因而还要再补一轮
  `VLLM_ASCEND_PREFIX_CACHE_TRACE_STEPS=128`
  的晚步数局部验证，
  把旧微漂移结论正式核销

### 最新结论 C：旧的 `slot 256` layer0 微漂移已被 `tree_row0_fused` 直接消除

补完 `128-step` 晚步数 trace 后：

- 文件：
  - `/tmp/layer0_prefix_compare_v5.jsonl`
- trace source 统计：
  - `decode_fused = 107`
  - `tree_row0_fused = 91`

重新对齐旧定位点：

- baseline:
  - `trace_source = decode_fused`
  - `step = 68`
  - `token = 9370`
  - `position = 128`
  - `slot = 256`
  - `seq_len = 129`
- tree:
  - `trace_source = tree_row0_fused`
  - `step = 61`
  - `token = 9370`
  - `position = 128`
  - `slot = 256`
  - `seq_len = 129`

逐项对账结果：

- `prefix_last_k_max_diff = 0.0`
- `prefix_last_v_max_diff = 0.0`
- `current_q_max_diff = 0.0`
- `current_k_max_diff = 0.0`
- `current_v_max_diff = 0.0`
- `current_output_max_diff = 0.0`

含义：

- 旧结论里那条
  `layer0 current_output = 6.103515625e-05`
  的 root-row 微漂移
  现在已经被直接清掉
- 这进一步坐实：
  - 根因不是“可见输入面不同”
  - 而是 row0 override 之前走错了 kernel path
  - 当 tree root 行按 baseline dispatch 进入 fused path 后，
    晚步数 residual mismatch 与端到端 token mismatch
    都一起消失了

下面这组结论会覆盖本文档里一部分更早期的 attention 怀疑点。

### 最新结论 -1：sample trace 已把主分歧前移到 raw hidden/logits

新增 `VLLM_ASCEND_SAMPLE_TRACE_PATH` 后，已经可以把：

- 输出前缀 tail
- sampled token
- tree terminal row
- row0 / 各 target row 的 hidden head、hidden norm
- top logits / top token ids

放到同一份日志里对比。

关键观测：

- baseline `step 11` 与 tree `step 10`
  - 逻辑前缀都结束于 `... 97145, 29490`
- 但 top logits 已经不同：
  - baseline row0:
    - `115995: 33.25`
    - `100210: 33.0`
  - tree row0:
    - `100210: 33.5`
    - `115995: 33.25`
- 同时 hidden state 的头部值与范数也已不同

含义：

- 当前主分歧发生在 sampler / verifier 之前
- 不是“采样规则不同才让 token 变了”
- 而是 target 前向在看似相同前缀下，已经算出了不同的 raw hidden/logits

### 最新结论 -0：last-prefix-token 的 cache continuity 已被进一步直接证伪

扩展 `VLLM_ASCEND_TREE_EXACT_DEBUG_PATH` 后，除了已有的：

- current K/V direct vs gather 一致
- row0 together vs isolated 几乎一致

之外，又补了：

- `current_slots`
- `prefix_tail_slots`
- `prefix_last_k_head / prefix_last_v_head`
- `current_k_heads / current_v_heads`

关键观测：

- 当逻辑 prefix 从 `69 -> 70 -> 71` 增长时
- 下一轮 `prefix_last_k_head`
- 与上一轮 row0 `current_k_heads[0]`
- 在已对账的 64 层上完全一致

含义：

- “最后一个已接受 token 在 cache 中被 sibling 覆盖 / 读成旧值”这个怀疑点当前没有证据支持
- 后续若继续怀疑 cache，更应该去查整段 prefix tail 的逻辑视图，而不是继续盯最后一个 slot

### 最新结论 0：first accept 之后的 first-pass replay 链已被日志直接证实

新增 `VLLM_ASCEND_TREE_FIRST_PASS_TRACE_PATH` 后，长文本 case 上已经可以把：

- verifier 的 `accepted_paths`
- proposer 的 `terminal_row_indices`
- next root 的 `rebuilt_token_ids / rebuilt_positions / rebuilt_seq_lens`

串成一条连续证据链。

关键观测：

- step 7 首次非空 accept：
  - `node_indices = [0]`
  - `token_ids = [3837]`
  - `bonus_token_id = 97145`
  - `terminal_row_index = 1`
- step 8 first-pass：
  - `rebuilt_token_ids = [97145]`
  - `rebuilt_positions = [68]`
- step 8 verify：
  - bonus token 仍然与 baseline 对齐，为 `29490`

含义：

- “第一次 accept 后，next root token 立刻喂错”这个怀疑点当前已被明显削弱。
- 剩余问题不是 first-pass immediate replay 断链，而是 accept 之后再往后一轮的 target 语义偏移。

### 最新结论 1：Qwen3 实跑确实走到了 Ascend tree exact attention

新增运行时 trace 后确认：

- Qwen3 模型实际 backend 为：
  - `Attention`
  - `AscendAttentionBackendImpl`
  - backend name `CUSTOM`
- tree target 的 attention trace 共记录 2048 条，全部为：
  - `path=exact`

含义：

- “Qwen3 没走到 `attention_v1.py` 的 tree exact path” 这个怀疑点已被证伪。
- 后续不应再重复花时间确认是否真的进了 exact path。

### 最新结论 2：cache 写回 / gather 当前不是主根因

新增 `VLLM_ASCEND_TREE_EXACT_DEBUG_PATH` 后，对 exact path 做了两类对账：

1. 对比“本轮 direct current K/V”与“写入 cache 后再 gather 回来的 current K/V”
2. 对比 row0 在：
   - 整棵树一起算
   - 只拿 `prefix + row0` 单独算
   这两种情况下的 attention 输出

实跑结果：

- `current_k_max_diff = 0`
- `current_v_max_diff = 0`
- 共 16384 条 exact debug 记录中：
  - `row0_output_max_diff_vs_isolated` 最大值为 `0.015625`
  - 大多数层为 `0` 或极小 bf16 数值噪声

含义：

- 当前 exact path 中，cache 写入和 gather 回读没有发生语义级错位。
- row0 也没有在 exact attention 内被 sibling rows 明显污染。
- 因此当前剩余分歧不再优先怀疑 attention batching / gather / cache。

### 最新结论 3：attention 之后的 logits / sampler / verifier 链路成为新的主战场

长文本“介绍中国”用例仍然复现：

- `baseline` 前 3 个生成 token：
  - `58695, 103987, 104320`
- `tree-eagle3` 前 3 个生成 token：
  - `58695, 103987, 116037`
- `first_diff_index = 2`

而且 verify trace 继续显示：

- step 0 bonus token = `103987`，与 baseline 第二个 token 对齐
- step 1 bonus token = `116037`，从第三个 token 开始偏离

结合最新 exact debug，可以把当前重点收缩到：

- tree 分支是否在 attention 之后取错了 logits / hidden-state 行
- tree 分支是否绕开了标准 speculative sampler 的某些必要约束
- verifier 当前消费的 token 是否与 baseline / 标准 spec decode 真正同源

### 最新结论 3.5：`seq_len=71` 的 `layer0 attention output` 也已经被直接排除

最新一轮 `v5` trace：

- baseline:
  - `/tmp/baseline_prefix_trace_v5.jsonl`
- tree:
  - `/tmp/tree_prefix_trace_v5.jsonl`

对齐方式仍然按 `seq_len`，不是按 step。

关键对齐点：

- baseline `step 10`
- tree `step 9`
- 共用 `seq_len = 71`
- 当前 token 都是：
  - `current_input_ids = [29490]`
  - `current_input_positions = [70]`
  - `current_slots = [198]`

这一轮对账的关键新增点是：

- 不再只看 `q/k/v`
- 还直接看 `current_output_heads`

实测结果：

- `prefix_last_k_head` 一致
- `prefix_last_v_head` 一致
- `current_q_heads` 一致
- `current_k_heads` 一致
- `current_v_heads` 一致
- `current_output_heads` 也一致

### 最新结论 3.6：accepted-path canonical slot copy 已执行，但单靠它并没有修复 late-step 分叉

为了验证“accepted token 没有被 materialize 回 canonical prefix slot”这个怀疑点，
本轮新增了 accepted-path KV copy，并挂了三个日志：

- `/tmp/tree_kv_materialize_v1.jsonl`
- `/tmp/tree_first_pass_after_materialize_v1.jsonl`
- `/tmp/tree_sample_after_materialize_v1.jsonl`

关键观测：

- `step 15` 时，四个 TP rank 都实际执行了：
  - `token_id = 20412`
  - `source_slot = 207`
  - `destination_slot = 206`
- 同一轮后续行为仍然没有变化：
  - `step 16 sampled_token_ids = [[20412, 99489]]`
  - `step 17 sampled_token_ids = [[31235]]`

含义：

- 不能再把问题简单归结为“accepted token 根本没有 copy 回 canonical slot”
- copy 动作本身已经存在
- 当前更窄的问题变成：
  - copied `slot 206` 是否真的被下一轮 attention 当作 prefix 读取
  - 是否存在另一份 cache / hidden-state / runtime state 仍保持旧值
  - 或 copied slot 在后续某个阶段又被覆盖掉了

因此下一轮验证不应继续重复“有没有 copy”，
而应直接观测 materialize 之后 next-step 的 prefix/cache 可见状态。

### 最新结论 3.7：materialize 之后，真正没对上的只剩 canonical `slot 206`

在上一条基础上，又补了一轮更窄的 trace：

- `/tmp/tree_prefix_after_materialize_v1.jsonl`
- `/tmp/tree_sample_after_materialize_v2.jsonl`

关键要看的是：

- tree `step 16`
- 此时：
  - `current_input_ids = [99489]`
  - `current_slots = [207]`
  - `seq_len = 80`
- sample 仍然是：
  - `sampled_token_ids = [[20412, 99489]]`

把它与 baseline `step 19` 对账后，得到一个非常强的结果：

- tree `current_k/v` 已经和 baseline 当前 token `99489` 对齐
- tree `prefix slots 203/204/205` 也已经和 baseline 对齐
- 只有 `prefix slot 206` 仍然不是 baseline 那个 accepted token `20412` 的状态

含义非常直接：

- 问题已经不是“整个 prefix tail 乱了”
- 也不是“current row 算错了”
- 而是：
  - canonical accepted slot `206`
  - 在 attention 真正 gather 到的 cache 视图里
  - 仍然保留着错误值

因此下一步不应再继续纠结 verifier / sampler / current row，
而应该直接去查：

1. `model_runner` 里被 copy 的 `self.kv_caches`
   是否就是 attention backend 读取的那份物理 cache；
2. 如果是同一份 cache，
   那么 `slot 206` 是在哪个后续阶段被覆盖回错误值的。

### 最新结论 3.8：materialize v1 的物理索引方式本身就是错的

在继续追 storage identity 之前，直接复核了代码：

- `tree_attention.py::gather_paged_kv_to_ragged()`

它清楚说明 paged KV cache 的 layout 是：

- 第 0 维：`block`
- 第 1 维：`block_size` 内的 token offset

而当前 `_copy_tree_accepted_path_kv_slots()` 却做了：

- 把 flat `slot_id = block * block_size + offset`
- 直接作为 `cache_tensor.index_select(0, slot_ids)` 的索引

也就是说：

- 它把 flat slot id 错当成了 block id
- 这不是“可能有问题”，而是物理地址层面就错了

这与现有 trace 完全吻合：

- materialize trace 里确实看到 `207 -> 206`
- 但 `step16` 时 attention 真正看到的 canonical `slot 206`
  仍然没有变成 baseline 的 accepted token 状态

因此下一步不需要再优先怀疑 verifier / sampler / hidden-state reuse，
也不需要先补更多 Triton/kernel，
而是应该先把 materialize copy 改成：

- `src_block = src_slot // block_size`
- `src_offset = src_slot % block_size`
- `dst_block = dst_slot // block_size`
- `dst_offset = dst_slot % block_size`

再按 `[block, offset]` 做真正的 gather/scatter。

### 最新结论 3.9：修正 block/offset 物理索引后，late-step canonical continuity 已恢复

按上一条直接修复 `_copy_tree_accepted_path_kv_slots()` 后，
立刻重跑了同一组晚步数定向 trace：

- `/tmp/tree_prefix_after_materialize_v2.jsonl`
- `/tmp/tree_sample_after_materialize_v3.jsonl`
- `/tmp/tree_kv_materialize_v2.jsonl`

关键观测如下：

- `step 15`
  - trace 里仍明确看到：
    - `source_slot = 207`
    - `destination_slot = 206`
- `step 16`
  - `current_input_ids = [99489]`
  - 这时 canonical `prefix slot 206` 的 `K/V`
    已与 baseline `step19 prefix_last_k/v` 完全一致
- `step 17`
  - sample 不再是旧的错误值 `31235`
  - 而是修正成 baseline 对齐的 `97145`

这组结果的含义非常明确：

- 之前不是“materialize 思路不对”
- 也不是“copy 后一定又被后续链路覆盖”
- 而是 v1 copy 从一开始就在错误的 paged cache 物理位置上操作

补充一个侧面信号：

- 同一轮窄 trace 中，accepted token 数也从 `3`
  提升到了 `5`

因此当前阶段性结论可以更新为：

- late-step `slot 206` canonical prefix continuity 缺口已经修通
- 这条 root cause 已从“待验证假设”升级为“已修复且已通过定向 trace 复验”
- 下一步不再继续围绕 `207 -> 206` 这个点做局部猜测，
  而是回到完整 `32B eager` compare，
  看是否还存在新的、更晚的 residual divergence

### 最新结论 3.10：完整 compare 的首个分歧点已从 `20` 后移到 `74`

在 3.9 的修复基础上，重跑了完整的长文本 `32B eager` compare：

- 结果文件：
  - `/tmp/qwen32_tree_eager_quick_check_noasync.json`

关键结果：

- baseline
  - `token_count = 108`
  - `generate_seconds = 10.06`
- tree
  - `token_count = 115`
  - `generate_seconds = 28.71`
  - metrics 最终打印到：
    - `Accepted = 6 tokens`
    - `Drafted = 180 tokens`
- compare
  - `first_diff_index = 74`
  - `baseline_token_id_at_diff = 104963`
  - `tree_token_id_at_diff = 105355`

这组结果的含义很重要：

- 旧版本里最早出现在 `first_diff_index = 20` 的那条端到端分歧，
  现在已经被显著推后
- 因而 3.9 修掉的 `slot 206` continuity 问题
  不只是“局部 trace 变好”，而是已经真实传导到完整生成结果
- 但 tree 仍然没有与 baseline 完全一致：
  - baseline 总 token 数是 `108`
  - tree 总 token 数是 `115`
  - 当前还存在一个更晚位置的新 residual divergence

因此后续调试纪律需要更新为：

- 不再把新的第 `74` 个 token 分歧
  继续归因到旧的 `207 -> 206` materialization bug
- 应把它当成已经暴露出来的下一层问题，
  重新围绕：
  - 第 `74` 个 token 附近的 prefix/cache/state
  - 以及对应 sample / first-pass / runtime state
  单独挂 trace

### 最新待验证判断 3.11：第 `74` 个 token 更像 residual drift，而不是旧式硬错误

先只记录当前判断，不把它写成结论。

当前可直接观察到：

- 分歧发生在这段共同前缀之后：
  - `...孕育了灿烂的中华文化。从古代的四大发明到现代的`
- baseline 继续生成：
  - `高铁、航天、信息技术等领域的飞速发展...`
- tree 继续生成：
  - `科技创新，中国在历史长河中不断演进...`

这与早前 `slot 206` 问题的表现不同：

- 早前是明显的错误 continuation，并且很快就能在局部 trace 上看到 prefix continuity 断裂
- 现在两边文本都仍然通顺，且分歧点已经被推迟到第 `74` 个 token

因此当前更合理的待验证判断是：

- 这可能是长程累积下的轻微 hidden/logits drift
- 在一个语义自由度较高、top logits 可能接近的位置，
  把 greedy top-1 推到了另一条仍然合理的续写分支

下一轮验证顺序固定为：

1. 挂 sample trace，抓第 `74` 个 token 前后的 top logits / margin
2. 如果看到 top-1 / top-2 极接近，再继续向前追 drift 的最早放大点
3. 如果 margin 很大，再回头怀疑新的 runtime state / cache 语义错位

补充一个 instrumentation 约束，避免重复踩坑：

- `vllm_ascend/worker/model_runner_v1.py` 里
  `VLLM_ASCEND_SAMPLE_TRACE_STEPS` 默认只有 `32`
- 所以第一次 `/tmp/qwen32_tree_eager_sample_late.jsonl`
  只覆盖到：
  - baseline `32` step
  - tree `32` step
- 这份日志本身不可能看到第 `74` 个 token
- 后续若继续用 sample trace 追这个问题，
  必须显式把 `VLLM_ASCEND_SAMPLE_TRACE_STEPS` 提高到足够大的值

### 最新结论 3.12：新的首个分歧已经前移到 tree `step 66` 内部的 `row [0]`

把 `VLLM_ASCEND_SAMPLE_TRACE_STEPS` 提高到 `128` 后，
终于拿到了覆盖第 `74` 个 token 的 sample trace。

关键对齐关系如下：

- baseline
  - `step 73 -> sampled 9370`
  - `step 74 -> sampled 104963`
- tree
  - `step 66 -> sampled [9370, 105355]`
  - `tree_terminal_row_indices = [1]`

更关键的是同一 step 内的 row 级 logits：

- tree `bonus` 行仍大体对齐 baseline `step 73`
- 但 tree `row [0]` 已经给出：
  - `105355: 31.5`
  - `99602: 31.125`
  - `104963: 30.875`
- 而 baseline `step 74` 是：
  - `104963: 30.0`
  - `105355: 29.625`
  - `99602: 29.5`

这条证据的意义非常直接：

- 当前 residual divergence
  不是在 tree `step 66` 结束后、进入 next-root replay 才发生
- 而是在 tree verify batch 的内部，
  accepted token `9370` 对应的后继 speculative row `[0]`
  就已经把排序翻掉了

因此下一步优先级必须调整为：

- 不先追 replay
- 先验证 baseline `step 74`
  与 tree `step 66 row [0]`
  是否真的消费了同一 logical prefix / slot / cache 视图
- 若输入面一致，
  问题就可以继续收缩到 non-bonus speculative row 的
  attention / block / logits 路径

### 最新结论 3.6：`layer1` 的问题不是“当前轮突然算歪”，而是历史 token 的高层 K/V 已经先轻微漂移

串行重跑后的 `v6` trace：

- baseline:
  - `/tmp/baseline_prefix_trace_v6.jsonl`
- tree:
  - `/tmp/tree_prefix_trace_v6.jsonl`

先看 `seq_len=71`：

- 对齐点：
  - baseline `step 10`
  - tree `step 9`
  - `layer = model.layers.1.self_attn.attn`
- 结果：
  - `current_input_ids / positions / slots` 一致
  - `current_q/k/v` 一致
  - `prefix_last_k/v` 一致
  - 但 `current_output_heads` 已有极小差异：
    - `max_diff = 0.00018310546875`

这说明 `layer1 attention output` 已经不是 exact match。

但更关键的是再往前一轮 `seq_len=70`：

- 对齐点：
  - baseline `step 9`
  - tree `step 8`
  - 同层 `model.layers.1.self_attn.attn`
- 结果：
  - `current_input_ids = [97145]`
  - `current_input_positions = [69]`
  - `current_slots = [197]`
  - 当前轮 `current_q/k/v` 仍一致
  - 但 `prefix_tail_slots = [193, 194, 195, 196]` 中，
    只有 `slot 196` 的历史 K/V 已经开始漂移：
    - `prefix_tail_k_heads.max_diff = 0.125`
    - `prefix_tail_v_heads.max_diff = 0.0018515586853027344`

这条证据非常重要，因为它把问题重新定性为：

- 不是 `seq_len=71` 当前轮的 `layer1 attention` 无缘无故临时算歪
- 而是更早一轮生成出的历史 token，
  在 `layer1` 就已经留下了轻微分叉
- 后面各轮只是稳定地继续消费这份已分叉的历史高层 K/V

进一步把 slot 和 token 对齐后，又能把问题再缩一圈：

- 这个发生漂移的 `slot 196`
- 对应的正是先前通过 accepted-path 进入前缀的 token `3837`

这意味着：

- 之前的 `row0 isolated` 精确对账虽然很重要，
  但它只证明了 row0 本身没被 sibling 污染
- 它没有回答另一个更关键的问题：
  - 被 accept 的非 row0 token
  - 在写入高层 cache 时

### 最新结论 3.9：block-level 对账把“首个可见放大点”收缩到 non-row0 的 `layer0 self_attn` 返回值

这轮新加的 block trace 记录位置不是 `attention_v1.py` 内部，而是直接挂在：

- `Qwen3DecoderLayer.forward`
- `Qwen3MoeDecoderLayer.forward`

记录字段包括：

- `layer_input_hidden`
- `pre_attn_hidden`
- `attn_output`
- `post_attn_hidden`
- `post_attn_residual`
- `mlp_output`

对齐方式改成：

- baseline `seq68` 对 tree `seq73 row0(slot195)`
- baseline `seq69` 对 tree `seq73 row1(slot196, token 3837)`

关键结果如下。

#### 3.9.1 `slot195` 仍然完全对齐

`baseline seq68` 对 `tree seq73 row0(slot195)` 时：

- `model.layers.0`
- `model.layers.1`

上的：

- `layer_input_hidden`
- `pre_attn_hidden`
- `attn_output`
- `post_attn_hidden`
- `post_attn_residual`
- `mlp_output`

全部 `max_diff = 0.0`

这与前面已有结论一致：

- row0 在 tree exact path 里会被单独用 decode path 覆盖
- 因此 row0 不能代表 non-row0 speculative row 的真实正确性

#### 3.9.2 `slot196` 在 `layer0 self_attn` 返回值就开始偏

`baseline seq69` 对 `tree seq73 row1(slot196)` 时：

- `layer0 layer_input_hidden = 0.0`
- `layer0 pre_attn_hidden = 0.0`

说明在进入 `self_attn` 之前还完全一致。

但：

- `layer0 attn_output max_diff = 0.001953125`

随后开始连续放大：

- `layer0 post_attn_hidden max_diff = 0.000732421875`
- `layer0 post_attn_residual max_diff = 0.001953125`
- `layer0 mlp_output max_diff = 0.00390625`
- `layer1 layer_input_hidden max_diff = 0.00390625`
- `layer1 pre_attn_hidden max_diff = 0.000732421875`
- `layer1 attn_output max_diff = 0.005859375`
- `layer1 mlp_output max_diff = 0.015625`

#### 3.9.3 这与 `attention_v1` trace 组合后的真正含义

要注意两个 trace 的语义不同：

- `attention_v1.py` 里的 `current_output_heads`
  - 是 attention impl 的输出
  - 还没有经过 `o_proj`
- block trace 里的 `attn_output`
  - 是 `Qwen3Attention.forward()` 返回给 decoder block 的值
  - 已经经过 `o_proj`

因此最新证据链应写成：

1. `slot196` 的 `layer0 current_q/k/v` 一致
2. `slot196` 的 attention impl 输出已经出现极小漂移
3. 该漂移经 `o_proj` 后被进一步放大
4. 再经 residual / MLP / layer1 继续放大

#### 3.9.4 当前最窄问题面

这意味着当前不应再把主要精力放在：

- `layer0` 之后的 async batch state
- block glue
- residual materialization

而应优先怀疑：

- non-row0 speculative row 的 exact tree attention 数学路径
- 或该路径消费到的“更老 prefix K/V”仍有未观测到的轻微漂移

下一步验证应该围绕这两者继续收缩，而不是再泛化地怀疑“整个 block 后半段”。

### 待验证 3.10：先用“全 prefix head dump”区分 older prefix 漂移 vs exact attention 数学路径

当前虽然已经证明：

- `slot196` 在 `layer0 pre_attn_hidden` 之前一致
- recent generated prefix `60..67` 一致
- `slot195` 当前 row0 的 `current_q/k/v/output` 一致

但这还不足以推出“`slot196` 可见的全部 prefix K/V 都一致”，因为：

- tree `row1(slot196)` 仍会消费更老的 prompt / prefix cache
- 之前并没有把这整段历史全量导出过

所以当前最小下一步不是继续补更多 block hook，而是：

1. 复用现有 trace 逻辑
2. 把 `prefix_tail` 临时拉满到全 prefix
3. 只对 `layer0`、`slot196` 周围的对齐点重跑 baseline / tree

如果全 prefix head 仍完全一致，而 `current_output` 还不同，
就可以把主怀疑点进一步压到：

- exact attention 数学路径本身
- 或 GQA / head expand / softmax 数值路径

否则就说明：

- 更老 prefix K/V 里确实还有未观测到的历史漂移

### 最新结论 3.10：在 `slot196/layer0` 上，full-prefix head 已证实输入面对齐，remaining 差异就是 non-row0 attention 路径本身

上面的待验证项已经实跑完成，使用的是：

- baseline:
  - `/tmp/baseline_prefix_trace_fullpref_v1.jsonl`
- tree:
  - `/tmp/tree_exact_fullpref_v1.jsonl`

对齐点仍然是：

- baseline `seq69`
- tree `seq73 row1(slot196)`
- `layer0`

为了把可见输入面完全摊开，对比方式是：

1. baseline `seq69` 的 prefix `slots 128..195`
2. tree `row1(slot196)` 的可见输入：
   - prefix `slots 128..194`
   - current row0 `slot195`
   - current row1 `slot196`

程序化结果：

- older prefix `slots 128..194`
  - `K` 全 `0.0`
  - `V` 全 `0.0`
- `slot195`
  - baseline `prefix_last_k/v`
  - 对 tree `current row0 k/v`
  - 全 `0.0`
- `slot196`
  - baseline `current_k/v`
  - 对 tree `current row1 k/v`
  - 也全 `0.0`
- 但：
  - baseline `current_output_heads`
  - 对 tree `current_output_heads`
  - 仍有 `max_diff = 6.103515625e-05`

这条证据非常关键，因为它说明：

- 至少在当前观测 head 上
- `slot196` 的所有可见输入已经对齐
- 仍然发生的 `current_output` 差异
- 已不能再靠“older prefix/cache 先漂了”来解释

当前最强结论应更新为：

- non-row0 speculative row 的 attention 执行路径
- 与 baseline decode/fused kernel 的数值路径并不严格同值

也就是说：

- 当前 Python dense softmax `run_exact_tree_attention()`
- 虽然语义上看似正确
- 但在 greedy 长文本场景里仍会积累出可见分叉

因此后续修复方向不应再主要围绕日志扩展，而应转成：

- 让 non-row0 speculative row 尽量复用更接近 baseline 的 Ascend attention 数值路径
- 或设计一个 correctness-first 的 sequential fallback

### 待验证 3.11：把 non-row0 改成“row-wise compact visible KV + Ascend fused attention”路径

当前最直接的修复思路是：

1. 保留 row0 的现有 decode override
2. 对 non-row0 speculative rows，不再走 Python `matmul + softmax`
3. 而是按每一行实际可见的：
   - prefix
   - ancestors
   - self
   压成一条紧凑的 dense KV 序列
4. 然后直接调用 Ascend `npu_fused_infer_attention_score`

这样做的目标不是先追求性能，而是先把数值路径尽量拉回到更接近 baseline kernel 的实现。

如果这条路有效，再考虑：

- 是否默认启用
- 如何在保持正确性的前提下做后续性能优化

### 最新结论 3.11：vendor rowwise fallback 已显著后移首个分叉点，但还没完全消除 later-stage 漂移

上面的修复思路已经落到代码：

- 文件：
  - `vllm_ascend/attention/tree_attention.py`
- 行为：
  - row0 继续保持 decode override
  - non-row0 不再走 Python `matmul + softmax`
  - 改成逐 row 构造 compact visible KV
  - 然后调用 `npu_fused_infer_attention_score`
  - 如果 vendor 调用异常，再退回旧的 Python dense path

长文本主复现的最新结果：

- baseline token count: `108`
- tree token count: `112`
- `first_diff_index = 20`

与旧版本相比：

- 旧版 `first_diff_index = 11`
- 现在前 20 个 token 已经与 baseline 完全一致

这说明：

- non-row0 attention 数值路径确实是当前主问题之一
- vendor rowwise fallback 方向是正确的
- 当前改动不是只修了日志，而是已经真实改善了端到端输出对齐

但同样要明确：

- 它还没有完全消除后续分叉
- 后面的剩余偏差说明：
  - 要么 later speculative rows 还有别的 attention 数值问题
  - 要么在更深/更晚的阶段，问题又转移到了其他链路

因此下一轮不应该把这次修复当作“已经彻底完成”，
而应围绕新的 `first_diff_index = 20` 继续定点调试。
  - 是否就已经和 baseline 的同 token 表示不同

所以接下来的最优先验证目标，应直接围绕这个 accepted token `3837` 展开，
而不是抽象地继续往更深层扫：

- baseline 正常 decode 下的 `3837`
- tree accepted row 中的 `3837`
- 两者从 `layer0` 到 `layer1` 的哪一处开始不再相同

### 最新结论 3.7：accepted token `3837` 的最早分叉点已经被钉到 `layer0 -> layer1` 之间

这一步没有新增重跑，而是直接复用了现有三份日志：

- baseline `layer0`：
  - `/tmp/baseline_prefix_trace_v5.jsonl`
- baseline `layer1`：
  - `/tmp/baseline_prefix_trace_v6.jsonl`
- tree accepted-row exact debug：
  - `/tmp/tree_only_exact_debug_step11.jsonl`

做法：

1. 先用 baseline `seq_len=68` 的单 token bonus 行，
   去匹配 tree exact debug 中 `seq_len=73, current_slots=[195,196,197,198,199,200]`
   的 4 个 TP shard
2. 结果 `pid=725850` 在 `layer0 row0` 上与 baseline 完全一致，
   因而可以把它当作同一 head shard 来继续比 accepted row
3. 再比较该 shard 上的 accepted row：
   - `slot 196`
   - `token 3837`

结果：

- `layer0`
  - tree accepted row vs baseline `seq_len=69`
  - `current_k/v` 完全一致
- `layer1`
  - 同一 token 的 `current_k/v` 开始出现微小差异
  - `current_k_heads.max_diff = 0.01953125`
  - `current_v_heads.max_diff = 0.0015869140625`

含义：

- 这不是“accepted row 一生成出来就和 baseline 完全不同”
- 它在 `layer0` 仍是正确的
- 但到了下一层之前，
  某一段 block 内状态变换已经开始把它带偏

于是当前最小且最有价值的下一步已经明确：

- 不再泛化扫更深层
- 直接给这个 accepted row 补：
  - `layer0 current_output_heads`
  - 或 block-level output hook
- 目标是把问题压缩到：
  - `layer0 attention output`
  - `o_proj`
  - residual add
  - MLP / block output
  其中之一

### 最新实验计划 3.8：把 tree exact attention 改成逐 row 独立计算，验证 batched multi-row 数值路径是否就是根因

目前证据链已经足够具体：

- row0 单 row exact 与 baseline 完全一致
- accepted row `3837` 在 `layer0 current_q/k/v` 上也与 baseline 一致
- 但它在 batched multi-row exact 下，
  `layer0 current_output` 已出现 `6.103515625e-05` 级微小漂移
- 这点微漂移到 `layer1 current_q` 已被放大到 `0.04296875`

因此一个非常直接的修正实验是：

- 保持 tree 语义不变
- 但不再对同一 request 的 `q_len` 行一起做 batched GEMM
- 改成每个 speculative row 单独取自己的 visible K/V，
  独立做一次 `q_len = 1` 的 exact attention

如果这样一改，accepted row 的 `layer0 output` 与 baseline 恢复 exact match，
那就说明：

- 当前主问题并不是 tree mask 语义错
- 而是 batched multi-row exact path 的数值实现对这个模型不够稳定

### 最新结论 3.9：逐 row 独立 exact attention 也没有改变 accepted row 的漂移，说明问题不在 batched-q 实现形式

我已经实际把 `run_exact_tree_attention()` 改成“每个 speculative row 单独做一次 q_len=1 的 exact attention”，
然后重新跑了 tree-only 32B，并用同样的方法对账 accepted row `3837`。

结果完全没有改善：

- `layer0 row1 current_output.max_diff`
  仍然是 `6.103515625e-05`
- `layer1 row1 current_q.max_diff`
  仍然是 `0.04296875`
- `layer1 row1 current_k.max_diff`
  仍然是 `0.125`

含义：

- 之前怀疑的“batched multi-row GEMM 数值路径导致 accepted row 漂移”
  目前已被证伪
- 即便改成逐 row 独立 attention，
  accepted row 的最早漂移位置也完全不变

所以接下来的优先级要切换为：

- 撤回这个无收益的功能改动
- 保留本轮新增的 instrumentation
- 继续查 accepted row 所消费的更早 prefix 历史，
  尤其是 positions `60..67` 的 generated prefix，
  是否已经在 `layer0/1` 上埋下了微小偏差

因此下一步最值得查的不是继续泛化地往深层扫，
而是直接围绕这个历史 token 去定位：

- 它在 `layer0` 是否仍完全一致
- 如果 `layer0` 一致，
  那么最早分叉点就位于：
  - `layer0 attention output`
  - `layer0 o_proj / residual / mlp / block output`
  - 到 `layer1 attention input`
  之间

并且：

- `current_output_heads` 对账维度已经是整头 `128` 维
- `max_diff = 0.0`

含义：

- `layer0 attention` 现在可以直接排除
- 根因不在：
  - 当前 token 的 prefix gather
  - 当前 token 的 q/k/v 构造
  - `layer0` 的 decode attention kernel 本身
- 根因更可能在：
  - `layer0 attention` 之后的同层路径，例如残差 / `o_proj` / MLP / block output
  - 更深层 transformer block
  - 或 block 之后到 sampler 之前的 hidden/logits 链路

基于这个结论，下一轮的策略也要随之调整：

- 不急着一上来就对每个 block 大面积加 hook
- 先继续复用现有的 `prefix-cache trace`
- 但把 `VLLM_ASCEND_PREFIX_CACHE_TRACE_LAYERS` 切到更深层 attention，例如：
  - `model.layers.1.self_attn.attn`

原因：

- 如果 `layer1` 的 `current_q_heads` 已经分叉，
  那么问题就被压到了：
  - `layer0 attention output` 之后
  - `layer1 attention` 之前
- 如果 `layer1` 仍完全一致，
  就继续看更深层 attention 边界
- 这样能先用现有 instrumentation 做“分层二分”，
  再决定是否真的需要单独对：
  - `o_proj`
  - `post_attention_layernorm`
  - `mlp`
  - block output
  补新的 hook

### 最新结论 3.6：`layer1` 边界探针首轮没有进入模型对账，先撞上了启动环境问题

为了继续沿 attention 边界往更深层缩圈，我把：

- `/root/tmp_baseline_prefix_trace.py`
- `/root/tmp_tree_only_step11.py`

都补成了自带 CANN/ATB bootstrap 的版本，目标是后续可以直接切：

- `VLLM_ASCEND_PREFIX_CACHE_TRACE_LAYERS=model.layers.1.self_attn.attn`

但是首轮真正执行时，baseline/tree 都没有进入模型对账，而是在 worker 初始化阶段报：

- `pool INTERNAL ASSERT FAILED`
- `ParallelOpenMP.cpp:64`
- `Invalid thread pool`

含义：

- 这不是新的模型语义证据
- 只是说明新脚本 bootstrap 还没有完全对齐之前稳定脚本的 worker 启动环境
- 当前最值得优先修的不是模型代码，而是：
  - multiprocess worker 启动相关环境项
  - 尤其是与旧稳定脚本相比遗漏的环境变量

在这个问题修好前，不能把“`layer1` 还没分叉”或者“`layer1` 已分叉”当作任何结论。

后续又得到一个更具体的补充结论：

- 把旧稳定脚本中的 worker 启动环境项补齐后，
  `Invalid thread pool` 已经不再出现
- 这说明上一轮报错并不是模型或 trace 逻辑问题，
  而是脚本 bootstrap 缺项

但是第二轮 deeper-layer trace 仍然没有进入有效语义对账，原因变成了：

- 我把 baseline 和 tree 两个 4 卡任务并行跑了
- 两边的 HCCL 初始化发生端口冲突
- 关键报错是：
  - `Failed to enable listening for the NPU network adapter socket`
  - `The IP address 192.46.2.199 and port 16666 have already been bound`

这条新事实的直接含义很简单：

- 后续 deeper-layer trace 不能再并行跑 baseline/tree
- 必须改成串行
- 或单独管理 `HCCL_NPU_SOCKET_PORT_RANGE`

在这一点修好之前，同样不能从这轮实验推出任何关于 `layer1` 是否分叉的语义结论。

### 最新结论 3.7：`layer1 attention` 开始出现第一批可测的微小输出漂移

修正脚本 bootstrap，并把 baseline/tree 改成串行执行后，
`layer1` 边界探针终于拿到了有效结果。

结果文件：

- baseline:
  - `/tmp/baseline_prefix_trace_v6.jsonl`
- tree:
  - `/tmp/tree_prefix_trace_v6.jsonl`

对齐点：

- baseline `step 10`
- tree `step 9`
- `seq_len = 71`
- `layer = model.layers.1.self_attn.attn`

对账结果：

- `current_input_ids = [29490]`
- `current_input_positions = [70]`
- `current_slots = [198]`
- baseline/tree 完全一致

并且：

- `prefix_last_k_head` 一致
- `prefix_last_v_head` 一致
- `current_q_heads` 一致
- `current_k_heads` 一致
- `current_v_heads` 一致

但是：

- `current_output_heads` 不再像 `layer0` 那样逐元素全等
- 已出现极小非零差异：
  - `max_diff = 0.00018310546875`
  - `mean_diff = 5.607865750789642e-05`

含义：

- 漂移不是在 `layer1` 的 `q/k/v` 构造阶段才出现
- 更像是更早链路带来的微小扰动，
  到 `layer1 attention output` 才第一次被当前 trace 直接观测到
- 这也解释了为什么：
  - `layer0` 看起来完全一致
  - 但更后面的 hidden/logits 已经会逐步漂移

当前最合理的下一步：

- 继续沿 attention 边界往更深层看
- 判断这种 tiny drift 是从 `layer1` 起逐层放大，
  还是在某一更深层突然加剧

注意：

- v5 首轮人工复盘时曾误读成“字段全是 null”
- 这不是 trace 本身失败，而是因为这些观测字段位于 `requests[0]` 下
- 已更正，不应再把“v5 没有落值”当作事实

### 最新结论 3.8：accepted row 会消费到的 recent generated prefix `60..67` 仍然完全对齐

`v5` 这轮 exact debug 终于把 prefix tail 扩到了 8 个 token，
足够直接回答一个之前一直悬而未决的问题：

- accepted row `slot 196 / token 3837`
- 在后续 `seq_len = 73` 的 target forward 中
- 它会消费到的 recent generated prefix
  到底有没有在 `layer0/1` 上提前漂移

这次使用的对齐方式是：

- tree exact debug:
  - `/tmp/tree_only_exact_debug_step11_v5.jsonl`
- baseline `layer0`:
  - `/tmp/baseline_prefix_trace_v5.jsonl`
- baseline `layer1`:
  - `/tmp/baseline_prefix_trace_v6.jsonl`
- 对齐 shard：
  - `tp_rank = 0`
- 对齐点：
  - tree `seq_len = 73`
  - `current_slots = [195, 196, 197, 198, 199, 200]`

在这个视角下：

- `prefix_tail_slots = [187, 188, 189, 190, 191, 192, 193, 194]`
- 其中真正覆盖到 recent generated prefix `positions 60..66` 的是：
  - `slots 188..194`
- accepted row 的直接前驱 `position 67`
  则是当前 batch 里的：
  - `slot 195`

程序化对账结果是：

- `layer0`
  - `prefix slots 188..194` 的 `K/V max_diff` 全部为 `0.0`
  - `current slot 195` 的 `q/k/v/output max_diff` 全部为 `0.0`
  - `current slot 196` 才第一次出现：
    - `output.max_diff = 6.103515625e-05`
- `layer1`
  - `prefix slots 188..194` 的 `K/V max_diff` 仍全部为 `0.0`
  - `current slot 195` 的 `q/k/v/output max_diff` 仍全部为 `0.0`
  - `current slot 196` 才开始明显放大为：
    - `q.max_diff = 0.04296875`
    - `k.max_diff = 0.125`
    - `v.max_diff = 0.0018515586853027344`
    - `output.max_diff = 0.00048828125`

这条证据的含义非常直接：

- recent generated prefix `positions 60..67` 现在已经基本可以排除
- 问题不在“accepted row 读到了错误的最近历史 token”
- 也不在它的直接前驱 `slot 195`
- 当前最窄的问题面就是：
  - accepted token `3837` 自身在 `layer0 -> layer1` 之间开始偏离 baseline

因此下一步策略要再收紧一层：

- 不再继续重复检查 recent generated prefix
- 优先补 block-level hook
- 专门查：
  - `layer0 attention output`
  - `layer0 o_proj / residual / mlp / block output`
  - 到 `layer1 attention input`
  之间到底哪一段先分叉
- 如果后续还要继续怀疑“更早历史已漂移”，
  也应该直接去查更老的 prompt prefix，
  而不是继续回到最近这 8 个 generated prefix token

### 最新结论 4：root isolated exact attention 不是正确修正方向

做过一轮更激进的功能实验：

- 把 tree exact path 的 root/bonus 行从 `paged_attention` 切成 isolated exact attention

结果：

- 32B eager 长文本 case 从
  - `first_diff_index = 11`
  退化为
  - `first_diff_index = 2`

并且输出重新出现更早期的异常措辞与 `**`。

处理：

- 该实验改动已撤回
- 工作树已重新复跑确认恢复到 `first_diff_index = 11`

含义：

- “root paged decode kernel 本身就是主根因”目前没有证据支持
- 后续不能继续沿这个方向做功能性替换
- 当前更像是 branching tree 的后续 target 轮次仍存在 prefix/cache/logits 级别错位

### 最新结论 5：accepted draft continuity 已被直接证明正确

在 tree-only 复跑里，又补了一轮更细的 continuity 对账。

关键观测：

- `step 7` 首次非空 accept：
  - `accepted_paths = [{"node_indices":[0], "token_ids":[3837], "bonus_token_id":97145, "terminal_row_index":1}]`

### 待验证分流 6：用 row0 isolated full-forward 直接判断 batch-dependent drift 是否真实存在

前面虽然已经用 sample trace 看到：

- baseline `step 10` vs tree `step 9` 在 sampled token 仍同为 `29490` 时，
  hidden/logits 就开始漂移；
- baseline `step 11` vs tree `step 10` 才出现可见 token 分叉。

但这仍然留下一个关键分岔：

1. 是不是 tree target row0 本身只要和 sibling rows 一起跑，
   在 non-attention 路径上就会产生 batch-dependent drift？
2. 还是说 row0 自身其实没问题，
   只是后续消费的行、状态或上下文有错位？

为此，新增一个更直接的定位动作，状态目前是“待验证”：

- 在 `sample_tokens` 阶段，拿同一轮 tree target 的：
  - `input_ids[0:1]`
  - `positions[0:1]`
  - row0 对应的 decode-only style metadata
- 在同一进程里再做一次 isolated full forward，
  直接输出：
  - `batched row0 hidden/logits`
  - `isolated row0 hidden/logits`
  - top-k token/logit

这条 trace 的解释规则如下：

- 如果 batched vs isolated 仍显著不同：
  - 则主怀疑点继续向 non-attention batch drift 收缩；
  - 后续就该优先查量化线性、通信、MoE、forward context 泄漏。
- 如果 batched vs isolated 基本一致：
  - 则 row0 自身前向未必是主矛盾；
  - 需要回到 row selection / state reconstruction / runtime 消费语义继续排查。

注意：

- 这只是新的验证分流，不是新结论。
- 在看到首轮真实日志前，不能把它写成“已经证明 batch drift 存在/不存在”。

### 待验证分流 6 的首轮结果：先暴露出 instrumentation 本身的路径错误

首轮真实运行已经执行过，但结果不是有效对账，而是 helper 自身先崩了。

运行条件：

- 脚本：`/root/tmp_tree_only_step11.py`
- 模式：32B eager tree-only
- 额外挂载：
  - `VLLM_ASCEND_TREE_ROW0_ISO_TRACE_PATH=/tmp/tree_row0_iso.jsonl`

首轮暴露出的两个事实：

1. 脚本入口本身没有做 CANN bootstrap
   - 直接运行会在 `import acl` 失败；
   - 需要补齐 `PYTHONPATH/LD_LIBRARY_PATH/PATH` 后才能进入真实模型阶段。
2. 进入 sample 阶段后，新的 isolated helper 自身构造出了非法 attention metadata
   - 崩溃点：
     - `_maybe_dump_tree_row0_isolated_trace()`
     - `self._model_forward(...)`
     - `attention_v1.py -> forward_fused_infer_attention()`
   - NPU 报错核心信息：
     - `When attnMask is not provided, sparseMode must be 0`

这条报错的含义非常具体：

- 当前 helper 并没有把 isolated row0 送进我们预期的 correctness-first isolated path；
- 它实际上落到了 fused infer attention；
- 但与此同时，metadata 又没有满足 fused infer attention 对 `attnMask/sparseMode` 的约束。

因此本轮的结论只能是：

- row0 isolated rerun 这条 instrumentation 方向仍然值得做；
- 但当前实现还不能产出任何关于“batch drift 是否存在”的证据；
- 下一步必须先修 helper 的 metadata/path 语义，再做第二轮验证。

### 待验证分流 6 的第二轮结果：单 rank rerun 会挂在 TP collective

修完 metadata/path 之后，又做了第二轮真实运行。

这次没有再次报 `attnMask/sparseMode` 错，但依然没拿到有效 trace。

新现象：

- 主日志长时间停住；
- `tree_row0_iso.jsonl` 没有生成；
- `tree_only_result_step11.json` 也没有生成；
- EngineCore 日志出现：
  - `No available shared memory broadcast block found in 60 seconds`

结合本轮改动可以直接得到一个实现层面的新结论：

- nested isolated `_model_forward` 里仍然会走 TP collectives / TP logits 路径；
- 如果只让 `TP rank 0` 进入这段 rerun，而其他 TP ranks 不参与，
  就不会得到“单卡局部调试”，而是会直接导致多卡同步死等。

所以这条 helper 的正确执行约束已经明确了：

- 所有 TP ranks 都必须共同执行 isolated rerun；
- 只能把“写 trace”限定在 rank0；
- 绝不能把“执行 rerun”限定在 rank0。

### 待验证分流 6 的第三轮结果：row0 isolated full-forward 已正式跑通，且与 batched row0 完全一致

在修正了：

- metadata/path 语义；
- TP collective 参与方式；

之后，第三轮真实运行终于拿到了有效证据。

产物：

- helper trace：
  - `/tmp/tree_row0_iso.jsonl`
- tree-only 主结果：
  - `/tmp/tree_only_result_step11.json`

关键观测：

- `step 0..15` 全部为：
  - `hidden_max_abs_diff = 0.0`
  - `logits_max_abs_diff = 0.0`
- top-k token/logits 也逐步完全一致

最关键的是首个可见偏离前后：

- `step 8`
  - `row0_input_ids = [97145]`
  - `root_seq_len = 70`
  - batched/isolated top-1 都是 `29490`
- `step 9`
  - `row0_input_ids = [29490]`
  - `root_seq_len = 71`
  - batched/isolated top-1 都是 `100210`
  - batched/isolated top-2 都是 `115995`

这条证据的含义非常强：

- tree run 里当前 cache/state 条件下的 row0，
  本身就稳定地产生 `100210`；
- sibling rows 没有通过 batched non-attention 路径污染 row0；
- 因而“量化线性 / 通信 / MoE 的 batch-dependent drift”当前已经不是主矛盾。

新的收缩结论：

- 问题不在“同一 tree row0 因为是否与 sibling 同跑而变”；
- 问题更可能在：
  - tree 与 baseline 在进入这一轮 row0 之前，历史 cache/prefix 状态已经不同；
  - 或者 runtime 对历史 token/position/slot 的 materialization 仍有更早期错位。

### 下一条证据应该长什么样

既然 tree row0 本身已经能被 isolated rerun 完全复现，
后续最有价值的不是继续盯 row0 输出，
而是直接比较 baseline vs tree 在同一逻辑前缀下的历史 cache 视图。

具体要拿到的字段：

- 每层：
  - `prefix_tail_slots`
  - `prefix_tail_k_heads`
  - `prefix_tail_v_heads`
  - `prefix_last_k_head`
  - `prefix_last_v_head`
  - `current_k_heads`
  - `current_v_heads`

对比规则：

- 如果 baseline 与 tree 在 layer0 的 prefix tail 就已经不同，
  那么问题明确在更早期的历史状态构造，而不是当前 row0 计算本身；
- 如果 layer0 prefix tail 一致，但更深层开始分叉，
  再继续看 attention 输出 / residual / MLP 链路。
- 为了避免后续继续靠多份异构日志人工拼字段，
  当前决定先补一套统一 schema 的 `prefix-cache trace`：
  - baseline decode-only 与 tree row0 都输出同结构 JSON；
  - 首轮只过滤 `layer0`，控制日志规模；
  - 等确认 layer0 是否已经分叉后，再决定是否扩到更深层。

### 最新结论 8：baseline eager 当前不走 paged decode，首轮 prefix-trace 挂点打偏了

统一 schema 的 `prefix-cache trace` 第一版已经接进 `attention_v1.py`，并做了 baseline 首轮实跑。

首轮现象：

- `/tmp/baseline_prefix_result.json` 正常生成；
- baseline token 序列与历史记录一致；
- 但 `/tmp/baseline_prefix_trace.jsonl` 没有任何内容。

这不是“prefix 对不上”或“trace 条件没命中 step”，而是路径判断问题：

- 当前 baseline 配置：
  - 无 speculative config
  - eager mode
  - cudagraph mode 不是 `FULL_DECODE_ONLY`
- `using_paged_attention(...)` 在这组条件下返回 `False`
- 所以 baseline decode 会走：
  - `forward_fused_infer_attention`
  - 而不是 `forward_paged_attention`

含义：

- baseline 侧的统一 trace 不能只挂在 paged decode 上；
- 必须再补到 fused decode 路径；
- 这条修正属于 instrumentation 自身修正，不是功能逻辑新结论。

### 最新结论 9：baseline 第二轮失败是 `layer` 传参链没接全

在把统一 trace 补进 `forward_fused_infer_attention` 之后，baseline 第二轮实跑又暴露出一个新的 instrumentation 错误：

- `forward_impl()` 内调用：
  - `self.forward_fused_infer_attention(layer, ...)`
- 但当时 `forward_impl()` 自己的签名里还没有 `layer`
- 因此 worker 直接报：
  - `NameError: name 'layer' is not defined`

含义：

- 这不是模型输出或 cache 语义的新证据；
- 只是 trace 接线时把 `layer` 沿调用链往下传漏了一层；
- 已修正为：
  - `forward_impl(layer, ...)`
  - 并同步更新 `forward()` 调用点；
- 下一轮 baseline 重跑若仍失败，不能再重复怀疑同一个问题。

### 最新结论 10：按 `seq_len` 对齐后，首个真实分叉点在 `seq_len=72` 的 current K/V

baseline 与 tree 的统一 prefix trace 都跑通之后，不能再按 sample step 直接对齐，
因为 tree 在首个非空 accept 之后会天然“快一拍”。

改按 `seq_len` 对齐之后，结果很清楚：

- `seq_len=61..71`
  - `prefix_last_k/v` 完全一致
  - `current_k/v` 也完全一致
- `seq_len=72`
  - `prefix_last_k/v` 仍完全一致
  - 但 `current_k/v` 首次分叉
- `seq_len=73` 开始
  - 由于上一轮 `current_k/v` 已分叉，
  - 它们又会成为新的 prefix last，
  - 所以 prefix tail 才开始跟着分叉

这条证据的重要性非常高：

- 它直接推翻了“prefix/cache 历史更早就已经错了”这个当前主怀疑点；
- 至少在 `layer0` 上，第一次问题出现在：
  - 当前 token 的本轮前向
  - 而不是 prefix history 读取

因此下一步优先级立刻变化为：

1. 核对当前 token 的真实 `input_id`
2. 核对当前 token 的真实 `position`
3. 核对 row0 当前 slot / row 选择
4. 必要时再看 RoPE / mRoPE / qkv 输入是否已在 layer0 前就不同

### 最新结论 11：`current_input_ids/current_input_positions = null` 是代理白名单问题

为了继续验证 `seq_len=72` 当轮输入是否已不同，这一轮已经：

- 在 `set_ascend_forward_context()` 中写入：
  - `input_ids`
  - `positions`
- 在统一 prefix trace 中尝试读取：
  - `_EXTRA_CTX.input_ids`
  - `_EXTRA_CTX.positions`

首轮结果仍然是 `null`。代码核对后确认原因更具体：

- `_EXTRA_CTX` 不是直接访问原始 forward context，
  而是一个带 `extra_attrs` 白名单的代理；
- 当前白名单中并没有：
  - `input_ids`
  - `positions`
- 因而 `getattr(_EXTRA_CTX, "input_ids", None)` 实际上拿到的是默认值 `None`

含义：

- 这不是模型行为的新结论；
- 只是刚补的 instrumentation 还差最后一层代理接线；
- 下一步只需补白名单并重跑，
  不应该把这次 `null` 结果误判成“当前 token 输入不存在差异”。

### 最新结论 12：`seq_len=71` 的当前 token 输入一致，`seq_len=72` 的 token 分叉只是结果不是原因

补完 `_EXTRA_CTX` 代理白名单后，baseline/tree 的当前 token 输入已经可以直接对账。

关键结果：

- `seq_len=70`
  - baseline/tree:
    - `current_input_ids = [97145]`
    - `current_input_positions = [69]`
    - `current_slots = [197]`
- `seq_len=71`
  - baseline/tree:
    - `current_input_ids = [29490]`
    - `current_input_positions = [70]`
    - `current_slots = [198]`
- `seq_len=72`
  - baseline:
    - `current_input_ids = [115995]`
  - tree:
    - `current_input_ids = [100210]`
  - 两边的 `position / slot` 仍同为 `[71] / [199]`

含义：

- `seq_len=72` 的 current K/V 分叉并不神秘，它只是“当前 token 已经不同”后的自然结果；
- 根因必须回到前一轮 `seq_len=71`；
- 而在 `seq_len=71`，我们已经确认：
  - 当前 token 相同
  - 当前位置相同
  - 当前 slot 相同
  - prefix K/V 相同
  - current K/V 相同

补完 `current_q_heads` 后，结果比预期更强，但这里必须限定观测口径：

- `seq_len=71`：
  - baseline/tree 的 `current_q_heads` 在当前 trace 的“第一头前若干维切片”上完全一致

因此当前最合理的最小验证已经进一步收缩为：

1. 直接补 `layer0 attention output`
2. 同时把 `head_dim` 拉大到整头大小，避免只看前 8 维切片
3. 若同一组输入下 baseline fused decode 与 tree row0 decode 的 output 已不同，
   根因就落在 decode attention kernel / 路径差异
4. 若 output 仍一致，再往后看后续层或 post-attention residual/MLP
- target metadata 显示：
  - accepted draft token `3837` 对应 row1 的 `slot 196`
- exact debug 显示：
  - round 7 当前 row1 `slot 196`
  - round 8 prefix `slot 196`
  - round 9 prefix `slot 196`
  - 64 层 K/V `max diff = 0.0`

含义：

- “accepted draft token 写错 cache / 下一轮 prefix 读错 slot”这个怀疑点当前已经没有证据支持。
- 这比仅仅证明 last-prefix-token continuity 更强，因为它直接覆盖了首次非空 accept 之后最关键的 accepted draft token。

### 最新结论 6：hidden/logits 漂移早于可见 token 分歧

对齐 baseline/tree 的 sample trace 后，关键比较变成：

- baseline `step 10` vs tree `step 9`
  - sampled token 仍同为 `29490`
  - 但 hidden head 已有明显差异，`max diff = 0.5`
- baseline `step 11` vs tree `step 10`
  - baseline top-1 `115995`
  - tree top-1 `100210`
  - 可见 token 这时才第一次分叉

含义：

- 问题不是“这一轮 sampler 选错才导致分叉”。
- 更准确地说，target raw hidden/logits 在前一轮就已经开始漂移，只是还没立即翻转 top-1。

### 最新结论 7：`tree_state_trace` 仍有价值，但已降级为旁证

当前 `tree_state_trace` 依旧能看到：

- `prev_sampled_token_ids` 保持单 token 形态
- `spec_token_ids` 长期为全 `-1`
- `token_ids_cpu` tail 仍大量保留 placeholder

但在 accepted draft continuity、bonus continuity、row0 isolated attention、sample hidden/logits 时序都补齐之后，
这组现象已经不足以单独支撑“主根因就是 accepted path 未 materialize”这个判断。

因此当前优先级调整为：

1. 先查 non-attention 路径是否存在 batch-dependent drift
2. 再查 drafter / target 之间是否有 `_EXTRA_CTX` 等 forward context 状态泄漏
3. 最后才回到 async `input_batch` state materialization

### 最新怀疑点：tree sampler 分支与标准 spec decode 分支存在真实行为差异

代码对账发现：

- 标准 spec decode 分支会：
  - 先走 `bonus_sampler_output = self.sampler(..., predict_bonus_token=True, ...)`
  - 再走 `apply_sampling_constraints(...)`
- 当前 tree 分支则是：
  - 直接对 `target_logits.argmax(dim=-1)` 做 verifier
  - 没有走 `apply_sampling_constraints(...)`
  - 也没有走标准 bonus sampler 路径

当前状态：

- 这还不是最终结论
- 但它已经是比“attention 又出问题了”更强、更值得继续验证的新怀疑点

### 最新结论 5：async tree batch state 仍在沿用线性 placeholder 语义

为了避免继续只围绕 hidden/logits 做猜测，这轮又新增了：

- `VLLM_ASCEND_TREE_STATE_TRACE_PATH`

并在 `model_runner_v1.py` 的几个关键状态点落盘：

- `_bookkeeping_sync()` async 分支
- `_copy_valid_sampled_token_count()`
- `_prepare_inputs()` 中：
  - `update_num_computed_tokens_for_batch_change` 之后
  - `_correct_optimistic_seq_lens_cpu` 之后

对 `pid=705018` 的首轮 trace，已经看到一组比 sampler 差异更强的信号：

- `prev_sampled_token_ids` 持续表现为单 token 语义，例如：
  - `[[58695]]`
  - `[[103987]]`
  - `[[104320]]`
  - `[[106565]]`
  - `[[3837]]`
- `spec_token_ids` 长期仍是：
  - `[-1, -1, -1, -1, -1]`
- `request_output_tail`
- `token_ids_cpu_computed_tail`
- `token_ids_cpu_no_spec_tail`

这些尾部字段里长期保留大量：

- `-1`
- `0`
- placeholder 行为

更关键的是，到较后面的 step 已经出现：

- `prev_num_draft_tokens = [5]`
- `num_computed_tokens = [64]`
- `input_batch_num_computed_tokens_cpu = [69]`
- `optimistic_seq_lens_cpu = [70]`

但与此同时：

- `prev_sampled_token_ids = [[3837]]`
- `spec_token_ids` 仍全 `-1`
- `token_ids_cpu` 视图依然没有出现清晰的 accepted-path materialization

含义：

- 当前 async tree runtime 很可能没有把“真实 accepted path + bonus”写回成一个 self-consistent 的 batch state
- 反而更像是在：
  - 只保存一个 `prev_sampled_token_ids` 单 token
  - 继续沿用线性 speculative decode 的 `spec_token_ids == -1 placeholder` 语义

这会直接解释为什么：

- 日志里看到的输出前缀看似一致
- 但 target 前向内部真正看到的逻辑前缀状态已经开始漂移

### 最新结论 6：state trace 首轮报错只是 instrumentation 缺保护

首轮加入 `tree_state_trace` 时，trace 本身因为：

- `tree_terminal_row_indices` 在最早几个状态点尚未创建

导致过一次异常。

已经修复为：

- `getattr(self, "tree_terminal_row_indices", None)`

因此：

- 这不是 speculative tree 的功能性根因
- 只是 trace 补点时缺了防御式读取
- 当前 trace 可以稳定复跑

## 已掌握的关键证据

### 证据 1：root proposer 在最初几步基本正常

来自 draft trace：

- step 0:
  - regular root token: `334`
  - tree root top-k 首项也是 `334`
- step 1/2/3 也有类似现象

含义：

- tree drafter 的 root 选 token 逻辑至少在早期没有明显跑偏；
- “root top-1 立刻错” 不是当前最强怀疑点。

### 证据 2：前几步并非完全 reject，但也不是稳定长路径 accept

来自 verify trace，已观察到的模式包括：

- step 0: accept none, bonus `103987`
- step 1: accept none, bonus `104320`
- step 2: accept none, bonus `106565`
- step 3: accept node `[0]`, token `[3837]`, bonus `109068`
- step 4: accept none, bonus `60686`
- step 5: accept none, bonus `56137`
- step 6: accept none, bonus `99467`
- step 7: accept none, bonus `50366`

含义：

- 状态机在推进，不是完全卡死；
- 但 tree accept path 的语义是否被正确解释，仍然高度可疑。

### 证据 3：请求状态在前进，不像是简单复用了旧 token

tree draft root inputs 观察值：

- step 0 root input: `58695`
- step 1 root input: `103987`
- step 2 root input: `104320`
- step 3 root input: `106565`
- step 4 root input: `109068`
- step 5 root input: `60686`
- step 6 root input: `56137`
- step 7 root input: `99467`

含义：

- accepted / bonus token 确实在持续影响下一轮输入；
- 所以问题更像是“推进逻辑错位”而不是“没有推进”。

### 证据 4：首次分歧发生在大约第 7 个生成 token

含义：

- 最早几轮 all-reject 本身不一定就是根因；
- 更强怀疑是：
  - verify 输出解释错
  - accepted path 回灌下一轮 drafter 的语义错
  - tree row / tree mask / position 的语义在几轮后累积偏移

## 当前最值得盯的轮次

在当前稳定代码下，首次分歧已不是早期 step 1/2，而是：

- 输出 token 前缀：
  - baseline:
    - `[58695, 103987, 104320, 106565, 3837, 109068, 60686, 100490, 3837, 97145, 29490, 115995, ...]`
  - tree:
    - `[58695, 103987, 104320, 106565, 3837, 109068, 60686, 100490, 3837, 97145, 29490, 100210, ...]`
- `first_diff_index = 11`

这意味着：

- step 8 bonus token `29490` 仍然对齐
- 真正的首次 target 偏离发生在 step 9

因此当前最优调试策略是围绕 step 9 查：

- target root 所见的 committed prefix 是否仍与 baseline 等价
- branching tree 是否让 target 侧 cache / logical prefix / hidden-logits 组装出现后移一轮的偏差

### 证据 5：canonical 顺序、runtime 顺序、execution 顺序是三层不同概念

已从代码和 trace 中确认：

- 上游配置解析会先对 `speculative_token_tree` 做 `sorted(normalized)`；
- 因此用户输入顺序不会直接保留；
- 当前例子里：

```python
用户输入:
[(0,), (1,), (0, 0), (0, 1), (1, 0)]

上游 canonical / runtime.paths:
[(0,), (0, 0), (0, 1), (1,), (1, 0)]

runtime.nodes_by_depth:
[[0, 3], [1, 2, 4]]

execution_runtime_node_indices:
[0, 3, 1, 2, 4]
```

含义：

- `runtime.paths` 是 canonical path order；
- `execution_runtime_node_indices` 是 verifier / target row / drafter return path 使用的 BFS execution order；
- 两者不一样本身不是 bug，但任何一处若把这两套顺序混用，就会产生隐蔽语义错位。

### 证据 6：verifier 当前消费的是 runtime order 的 `draft_token_ids`

从新 trace 可见：

- drafter 最终同时存在两种视角：
  - `runtime_order_tokens`
  - `execution_order_tokens`
- verify trace 中的 `draft_token_ids` 与 `runtime_order_tokens` 一致；
- verifier 在内部通过 `runtime_node_to_execution_index` 把 runtime node 映射回 target row / execution row。

例如第一轮：

```text
runtime_order_tokens   = [334, 334, 5616, 374, 55786]
execution_order_tokens = [334, 374, 334, 5616, 55786]
verify.draft_token_ids = [334, 334, 5616, 374, 55786]
```

含义：

- 当前 verifier 不是“直接把 execution order 当 runtime order 硬吃掉”；
- 至少这一层存在显式映射逻辑；
- 因此问题更像是“某层仍然在错误时机/错误位置混用了 execution 顺序的残留数据”。

### 证据 7：下一轮 drafter 输入会带着上一轮 execution rows 的残留 sibling tokens

新 trace 中观察到的轮次关系：

```text
round 0
prev execution_order = [334, 374, 334, 5616, 55786]
next root model_input_ids = [103987, 374, 334, 5616, 55786, 104811]

round 1
prev execution_order = [334, 5616, 5616, 22282, 11]
next root model_input_ids = [104320, 5616, 5616, 22282, 11, 104811]

round 2
prev execution_order = [15946, 9370, 3837, 9370, 3837]
next root model_input_ids = [106565, 9370, 3837, 9370, 3837, 104811]
```

这里可以看到：

- 下一轮 root 输入的第一个 token 往往是上一轮 `bonus_token_id`；
- 但后续 token 并不是“仅保留 accepted frontier”后的干净 tree 状态；
- 它会带着上一轮 execution rows 中的一部分 sibling / descendant 残留；
- 末尾还持续带有一个固定尾 token（当前 trace 中是 `104811`）。

含义：

- 这不是最终根因结论，但已经是目前最强的异常现象；
- 它强烈支持“drafter 下一轮准备输入时仍夹带了不该保留的 tree row 残留语义”这一方向；
- 也解释了为什么输出不是立刻崩，而是几轮后逐渐串味。

### 证据 8：`set_inputs_first_pass()` 当前仍按线性 speculative chain 语义重建下一轮输入

已从代码路径确认：

- `model_runner_v1.py::propose_draft_token_ids(...)`
  - 调用 `prepare_next_token_ids_padded()`
  - 调用 `prepare_inputs_padded()`
  - 然后将 `target_token_ids`、`next_token_ids`、`token_indices_to_sample`
    传给 `llm_base_proposer.py::_propose()`
- `_propose()` 内部会调用 `set_inputs_first_pass(...)`

`set_inputs_first_pass(...)` 默认路径中的关键逻辑是：

```python
self.input_ids[: num_tokens - 1] = target_token_ids[1:]
self.input_ids[token_indices_to_sample] = next_token_ids
```

这是一套标准的线性 speculative chain first-pass 语义：

- 先整体左移 `target_token_ids[1:]`
- 再只在 sample 位置覆盖 `next_token_ids`

对 tree 模式的直接影响：

- `next_token_ids` 虽然当前看起来是正确的 bonus / next root token；
- 但 `target_token_ids[1:]` 里的 execution rows 也会整体被保留下来；
- 因而上一轮未被接受的 sibling / descendant rows 会一起进入下一轮 drafter 输入。

含义：

- 这不是“trace 看起来怪”的间接推断，而是直接的代码级证据；
- 当前 tree 模式并没有专门的 tree-aware first-pass 输入重建逻辑；
- 只要还走这段默认路径，下一轮输入就天然带有线性链残留语义。

### 证据 9：tree 模式当前没有走 `needs_extra_input_slots` 的扩展输入路径

已确认：

### 证据 10：`async_scheduling=False` 不改变分歧点

验证动作：

- 重跑 `/root/tmp_tree_eager_quick_check_noasync.py`

观测结果：

- `first_diff_index = 7`
- baseline diff token: `100490`
- tree diff token: `56137`
- 结果文件：
  - `/tmp/qwen32_tree_eager_quick_check_noasync.json`

结论：

- 当前问题不是 async scheduling 导致的时序错位。
- 后续不再继续围绕 async / no-async 做重复验证。

### 证据 11：tree target 确实已经强制走成 masked prefill-style attention

已做改动：

- 在 `vllm_ascend/attention/utils.py` 中新增：
  - `tree_spec_decode_enabled`
  - `get_effective_decode_threshold(...)`
- 在 `vllm_ascend/worker/model_runner_v1.py` 中：
  - `_build_attention_metadata(..., force_tree_prefill=False)`
  - tree target 调用时传 `force_tree_prefill=True`
- 在多个 attention metadata builder 中统一改为使用 `get_effective_decode_threshold(...)`

target trace 观测结果：

- `common_tree_spec_decode_enabled = true`
- `metadata_attn_state = ChunkedPrefill`
- `metadata_num_decodes = 0`
- `metadata_num_prefills = 1`
- `metadata_num_decode_tokens = 0`
- `metadata_num_actual_tokens = 6`

trace 文件：

- `/tmp/tree_target_trace_treeonly.jsonl`

结论：

- “tree target 错误地走了 decode fast path，导致 tree mask 没生效”这个假设已被证实不成立。
- 后续不再把 decode/prefill 分类当作主排查方向。

### 证据 12：即使强制走 masked prefill-style attention，branching tree 输出仍不变

验证动作：

- 重跑 tree-only 用例 `/root/tmp_tree_treeonly_check.py`

观测结果：

- tree 输出前缀仍为：

```text
[58695, 103987, 104320, 106565, 3837, 109068, 60686, 56137, 99467, 50366, 108402, 101920, ...]
```

- 结果文件：
  - `/tmp/qwen32_tree_treeonly_result.json`

结论：

- 单纯把 target batch 调到 masked prefill-style 路径，并不能修复真分叉树偏差。
- 问题更深层，已经超出 metadata 分类问题。

### 证据 13：链式 tree 与 baseline 完全一致，真分叉树才会出错

验证动作：

- 重跑 chain-tree 用例 `/root/tmp_tree_chain_check.py`

观测结果：

- `baseline.token_ids == tree_chain.token_ids`
- 结果文件：
  - `/tmp/qwen32_tree_chain_check.json`

结论：

- multi-token target verification 本身并非普遍错误。
- 当前剩余问题是 branching-specific。
- 因此最强假设变为：
  - 现有 fused attention 路径并不能正确表达任意分叉树的可见性语义；
  - “positions + tree mask 直接套到现有 fused path” 这条路对于真分叉树不够。

## 当前结论

截至当前，以下判断已经足够稳固：

1. first-pass 污染问题已修复，但不是最后根因。
2. decode/prefill 分类问题已被系统性证伪。
3. 链式 tree 正确，真分叉 tree 错误。
4. 因此下一阶段不再围绕 scheduler / metadata 分类微调，而是直接在 `vllm_ascend/attention/` 中实现 tree-only 精确 attention 执行路径。

## 新进展：tree-only exact attention 已落地

已完成实现：

- 新增：
  - `vllm_ascend/attention/tree_attention.py`
- 接入：
  - `vllm_ascend/attention/attention_v1.py`
  - 标准 attention 路径
  - C8 attention 路径

核心思路：

1. tree 模式下不再依赖现有 fused path 的分叉可见性语义。
2. 显式从 paged KV cache gather 出 request-major 的 ragged dense KV。
3. 对每个 request 的 speculative suffix rows：
   - prefix token 全可见
   - suffix token 仅按 tree mask 可见
4. 用精确 attention 先保 correctness。

### 证据 14：新 attention fallback 对某些 prompt 已经完全修正

验证动作：

- 运行：
  - `/root/worktrees/vllm-ascend-clean/run_qwen32_tree_compare.py`

观测结果：

- 数学分数题 prompt 下：
  - `baseline.token_ids == tree.token_ids`
  - `first_diff_index = null`
  - 文本完全一致，均输出 `7/18`

结果文件：

- `/tmp/qwen32_tree_compare_result.json`

结论：

- tree-only exact attention 不是无效改动。
- 当前 tree 分支语义问题至少已有一部分被真正修复。

### 证据 15：旧的长文本复现仍未完全修复，但分歧形态已经改变

验证动作：

- 重跑：
  - `/root/tmp_tree_eager_quick_check_noasync.py`

观测结果：

- baseline 与 tree 仍不一致
- 但分歧点已从旧的：
  - `first_diff_index = 7`
- 变成新的：
  - `first_diff_index = 2`
- 对应 token：
  - baseline: `104320`
  - tree: `116037`

结果文件：

- `/tmp/qwen32_tree_eager_quick_check_noasync.json`

结论：

- attention 语义问题确实被触动并部分修复了，否则分歧模式不会发生这种结构性变化。
- 但当前剩余问题已经不能再简单归结为“tree attention 没实现好”。
- 更强假设变为：
  - verifier 消费 logits 的行选择仍可能错位
  - accepted / bonus token 回灌下一轮 drafter 的语义仍可能对某些 prompt 不稳定
  - prompt 复杂度提升后，root / branch 提前分化，attention 只是其中一部分根因

## 下一步执行策略

1. 在 `attention` 层新增 tree-only helper，而不是继续把 tree 逻辑堆在 `spec_decode/` 下。
2. 从 paged KV 中显式 gather dense KV。
3. 对每个 request 按“prefix 全可见 + suffix 按 tree mask 可见”执行精确 attention。
4. 先在 eager / 单卡 / 32B 场景把 correctness 跑通，再决定是否追加 Triton / fused kernel 路径。
5. 当前下一步应转向 verifier / logits 行对齐与 accepted-path 语义排查，而不是继续只在 attention 层打补丁。

- tree 模式当前仍落在：

```python
if not self.needs_extra_input_slots:
    ...
```

这条默认分支中；

- 它没有走下面那条更复杂的扩展输入构造路径：

```python
torch.ops._C_ascend.npu_copy_and_expand_eagle_inputs(...)
```

含义：

- 当前 tree 模式不是“已经有专用输入重建，只是某处小错”；
- 它本质上仍在复用线性 EAGLE first-pass 框架；
- 因而后续修复大概率需要：
  - 为 tree 模式单独加 first-pass 输入重建；
  - 或让 tree 模式走一条新的 tree-aware expand path；
  - 而不是只在现有线性左移逻辑上打补丁。

### 证据 10：tree first-pass 已改成 terminal-row 压缩，旧树残留 rows 已不再进入下一轮 drafter root

本轮新增结构化修复：

- 新增：
  - `vllm_ascend/spec_decode/tree_first_pass.py`
- 在：
  - `vllm_ascend/spec_decode/llm_base_proposer.py::set_inputs_first_pass()`
  - tree 模式下不再复用默认线性左移逻辑；
  - 改为：
    - 只保留 verifier 选中的 `terminal_row_indices`
    - 用 `next_token_ids` 作为下一轮 root token
    - 将下一轮 first-pass 压缩为“每请求 1 行”

新 trace 直接显示：

```text
step 0:
token_indices_to_sample = [0]
target_positions = [59]
model_input_ids = [58695]

step 1:
token_indices_to_sample = [0]
target_positions = [60]
model_input_ids = [103987]

step 2:
token_indices_to_sample = [0]
target_positions = [61]
model_input_ids = [104320]
```

这和之前：

```text
model_input_ids = [103987, 374, 334, 5616, 55786, 104811]
```

这种“整棵旧树 execution rows 残留一起带入下一轮”的现象已经不同。

含义：

- H3 中“next-round drafter 输入夹带旧 sibling / descendant rows 残留”这一部分，当前已有修复证据；
- proposer first-pass 至少在 token / position / hidden-state 批形状这一层，已经不再是旧的线性链残留模型。

### 证据 11：修完 first-pass 后，输出质量显著改善，但 greedy 等价性仍未恢复

使用同一 32B eager 对比脚本再次验证：

- baseline:
  - `generate_seconds ≈ 9.67`
  - `token_count = 108`
- tree:
  - `generate_seconds ≈ 8.12`
  - `token_count = 94`

tree 输出从之前的明显异常：

- 零散 `**`
- 明显复述
- 更重的措辞漂移

改善为当前更接近正常中文段落：

- 仍有轻微重复（例如 `金融、金融`）
- 仍有措辞不完全自然
- 但整体可读性明显好于修复前

同时，关键事实是：

- `comparison.first_diff_index` 仍然是 `7`
- 说明“proposer first-pass 旧树残留”不是唯一根因
- 也说明修复虽然有效，但还没有恢复到 greedy baseline 等价

### 证据 12：在 first-pass 修复后，前若干轮 verifier 仍几乎全 reject，剩余问题更像 target 侧执行语义

新的 verify trace 前几轮：

```text
step 0: accept none, bonus 103987
step 1: accept none, bonus 104320
step 2: accept none, bonus 106565
step 3: accept none, bonus 3837
step 4: accept none, bonus 109068
step 5: accept none, bonus 60686
step 6: accept none, bonus 56137
...
step 11: accept [0], bonus 97145
```

现在有两个重要变化：

1. root drafter 输入已经压成单行，说明 proposer 残留链路已被切断。
2. 但在大量 all-reject 轮次下，tree 的 bonus token 序列仍会在较早位置偏离 baseline。

含义：

- 当 `terminal_row_index = 0` 且 all-reject 时，tree 结果本应最接近 baseline；
- 但现在即使在这种情况下仍出现早期偏离；
- 因此剩余问题已经更集中地指向：
  - target 侧 tree execution 的 metadata / attn / slot / KV 语义
  - 而不是 proposer next-round root 输入本身。

## 调试记录规则

下表里的每个“可疑点”必须维护以下状态之一：

- `待验证`
- `验证中`
- `已证实`
- `已证伪`
- `暂缓`

若状态变化，必须同步更新“证据”和“结论”列。

## 当前可疑点台账

### H1: verifier 的 tree row 映射有误

- 状态：`待验证`
- 位置：
  - `vllm_ascend/spec_decode/tree_verifier.py`
- 重点函数：
  - `select_tree_accept_paths()`
- 怀疑内容：
  - `target_row_index` / `terminal_row_index` 的定义或使用顺序不对；
  - verifier 看到的“第 i 行”并不一定真对应 runtime 里“第 i 个 tree node”。
- 支持证据：
  - 输出不是一开始就崩，但在几轮后偏移，符合“row 语义错位后逐渐积累”的特征。
- 反证：
  - 暂无。
- 下一步验证：
  - 记录并核对每轮：
    - tree node canonical order
    - target row index
    - verifier 读取的 token/logit
    - 最终 accepted node path

### H2: target 侧 metadata 构建仍带有 chain speculative decode 假设

- 状态：`验证中`
- 位置：
  - `vllm_ascend/worker/model_runner_v1.py`
- 重点函数：
  - `_calc_tree_spec_decode_metadata()`
- 怀疑内容：
  - target row 的物理布局虽然是 tree 输入，但 metadata 仍按线性 speculative chain 去解释；
  - 例如 `draft_token_ids`、`target_logits_indices`、bonus row 等语义还不够 tree-native。
- 支持证据：
  - 当前实现来自 chain speculative decode 路径的复用很多；
  - 首次分歧不在首 token，说明“前几轮勉强能解释，后续累计失真”很像 metadata 语义不完全一致。
- 新增支持证据：
  - 在 H3 的 proposer first-pass 残留被切掉之后，`first_diff_index` 仍为 `7`；
  - 所以剩余偏差不能再主要归因于 next-round drafter root 输入污染；
  - 问题重心更可能已经转移到 target 侧 tree execution 的 metadata / slot / attn 语义。
- 已验证事实：
  - `verify.draft_token_ids` 当前对应的是 `runtime_order_tokens`，不是 `execution_order_tokens`；
  - verifier 依赖 `runtime_node_to_execution_index` 做 runtime -> execution 的映射；
  - 因而 metadata 至少已经显式区分了这两套顺序，不是完全扁平链式处理。
- 反证：
  - “metadata 完全没有 tree 语义”这一种最粗粒度说法已被削弱。
- 暂未排除部分：
  - 某些后续步骤仍可能把 execution 顺序残留当成 next-round draft 输入使用；
  - bonus row / terminal row 的定义也仍可能存在细粒度错误。
- 下一步验证：
  - 将单轮 target 输入的物理 row 排布打印出来；
  - 对照 tree runtime 的 canonical node order；
  - 明确每个 row 对应的 path / depth / parent。

### H3: accept 结果回灌到下一轮 drafter 时，仍按线性链语义处理

- 状态：`已证实`
- 位置：
  - `vllm_ascend/spec_decode/llm_base_proposer.py`
  - `vllm_ascend/spec_decode/tree_runtime.py`
  - `vllm_ascend/spec_decode/tree_execution.py`
- 怀疑内容：
  - verifier 得到的是 tree accepted path；
  - 但下一轮 drafter 输入准备，可能仍沿用 linear chain speculative decode 的状态推进方式。
- 支持证据：
  - 这是当前最强怀疑点；
  - 前几步正常，后面开始退化，非常符合“状态推进语义渐渐错位”的特征。
- 已验证事实：
  - 现有 trace 显示，至少在 token 级别上：
    - `step N` 的 `bonus_token_id`
    - 与 `step N+1` 的 `root_input_ids[0]`
    - 在已检查的前 8 步内完全一致。
  - 这说明“上一轮 bonus token 没有正确传到下一轮 root token”这一种最粗粒度的回灌错误，当前没有证据支持。
- 反证：
  - token 级前推是自洽的，弱化了“完全按错 token 前推”的怀疑。
- 暂未排除部分：
  - hidden state 回灌
  - slot / position 回灌
  - drafter 内部 query packing 语义
  - next-round root input 中是否错误保留了 sibling / descendant rows
- 新增代码级证据：
  - `set_inputs_first_pass()` 当前在 tree 模式下仍走默认线性路径：

```python
self.input_ids[: num_tokens - 1] = target_token_ids[1:]
self.input_ids[token_indices_to_sample] = next_token_ids
```

  - 这会把 `target_token_ids[1:]` 中的 execution rows 整体左移保留；
  - 因而 tree 模式下 next-round 输入天然会夹带上一轮 sibling / descendant 残留。
- 当前结论：
  - “tree 模式的 next-round drafter 输入仍按线性链 first-pass 语义准备” 这一点已有直接代码证据和 trace 证据，判定为已证实。
- 后续修复记录：
  - 已新增 tree-aware first-pass 重建路径；
  - 新 trace 已确认 next-round root 输入压缩为单行 terminal-row frontier；
  - 因而 H3 所指向的问题已经至少被部分修复，不再是当前唯一主矛盾。
- 下一步验证：
  - 继续确认 remaining divergence 是否完全来自 target 侧 tree execution。

### H6: target 侧 tree execution 的 slot / KV 语义仍不等价于 baseline decode

- 状态：`待验证`
- 位置：
  - `vllm_ascend/worker/model_runner_v1.py`
  - `vllm_ascend/attention/*`
- 怀疑内容：
  - target 侧虽然已补 tree positions 和 tree attention mask；
  - 但 slot / KV / seq_lens / query layout 的组合语义仍可能不是“greedy baseline 等价”的。
- 支持证据：
  - proposer first-pass 修完后，tree root 输入已不再残留旧树 rows；
  - 但 early all-reject 轮次下 bonus token 仍会在较早位置偏离 baseline；
  - 说明剩余问题更像 target 侧执行本身，而不是 drafter 下一轮输入准备。
- 反证：
  - 暂无。
- 下一步验证：
  - 补 target-side trace：
    - target input ids
    - positions
    - slot_mapping
    - row0 / terminal row bonus token
  - 重点核对 all-reject 轮次下，row0 target execution 是否严格等价于 baseline 单步 decode。

### H4: tree query position / slot offset 计算错误

- 状态：`待验证`
- 位置：
  - `vllm_ascend/spec_decode/tree_execution.py`
  - `vllm_ascend/worker/model_runner_v1.py`
- 怀疑内容：
  - sibling 节点应共享 depth，不应被当作线性追加 token；
  - 如果 query position 仍按 chain 递增，则 target attention 语义会错。
- 支持证据：
  - tree attention 最常见的错误之一就是 position 仍按线性偏移；
  - 这种错误通常不会让首 token 必然错，但会在更深层分叉后逐渐失真。
- 反证：
  - 暂无。
- 下一步验证：
  - 直接打印每个 node 的：
    - path
    - depth
    - query position
    - slot offset
  - 与预期 tree depth 对齐。

### H5: tree target attention mask 语义或极性错误

- 状态：`待验证`
- 位置：
  - `vllm_ascend/spec_decode/tree_execution.py`
  - `vllm_ascend/attention/attention_mask.py`
  - `vllm_ascend/attention/attention_v1.py`
- 怀疑内容：
  - mask 本身构造有误；
  - 或者 mask 构造正确，但传入 Ascend attention backend 后被重新解释；
  - 或极性与 backend 期望相反。
- 支持证据：
  - tree 模式本质上依赖 mask 正确表达 parent-closed visibility；
  - 若 mask 错，输出常表现为部分合理、部分串味。
- 反证：
  - 前几步 root proposal 合理，说明至少不一定是最表层 mask 全坏。
- 下一步验证：
  - 构造极小树的 mask 样例；
  - 明确 backend 最终看到的 mask 数值和语义；
  - 与手工期望可视化比对。

### H6: 第一轮 proposer 路径里 tree target mask 可能被错误复用

- 状态：`验证中`
- 位置：
  - `vllm_ascend/spec_decode/llm_base_proposer.py`
- 怀疑内容：
  - 第一轮 drafter 路径可能不该套用 tree target mask；
  - 过早覆盖 mask 可能污染 root hidden states。
- 当前实验状态：
  - 已有一个“未验证的试探性代码修改”：
    - 从 `llm_base_proposer.py` 的 first-pass proposer path 中移除了一个 tree target mask patch。
- 支持证据：
  - 从代码结构看，这里复用了较多 linear/padded EAGLE 路径，存在误套 mask 的风险。
- 反证：
  - 还没做实验回归，暂时不能下结论。
- 下一步验证：
  - 在当前 patch 保持不变的情况下重新跑对比；
  - 若无改善，再回文档将其标记为 `已证伪` 或恢复代码后继续验证。

### H7: tree runtime canonical order 与实际执行 row order 不一致

- 状态：`验证中`
- 位置：
  - `vllm_ascend/spec_decode/tree_runtime.py`
  - `vllm_ascend/spec_decode/tree_execution.py`
  - `vllm_ascend/worker/model_runner_v1.py`
- 怀疑内容：
  - tree runtime 定义的节点顺序，与 target/draft 实际输入 tensor 的 row 顺序并非同一顺序；
  - BFS / path order / tensor packing order 之间可能有一个地方不一致。
- 支持证据：
  - 任何 tree runtime 最容易出现的 bug 之一就是“逻辑顺序”和“物理顺序”脱钩。
- 已验证事实：
  - 上游 `speculative_token_tree` 在解析阶段会先做 `sorted(normalized)`；
  - 因此用户输入的路径顺序并不会直接保留到 Ascend runtime；
  - 当前例子里，用户输入：

```python
[(0,), (1,), (0, 0), (0, 1), (1, 0)]
```

  - 上游 canonical 后变成：

```python
[(0,), (0, 0), (0, 1), (1,), (1, 0)]
```

  - 当前 verify trace 中的 `runtime_paths` 与该 canonical 顺序一致。
- 反证：
  - “顺序和用户输入不一样”本身不是 bug，因为这是上游明确的 canonical 行为。
- 暂未排除部分：
  - `runtime_paths` 的 canonical 顺序
  - `execution_runtime_node_indices` 的 BFS 顺序
  - `target row` 的物理顺序
  - `drafter` 内部 query packing 顺序
  - 这四者之间仍可能存在错位。
- 新增证据：
  - 新 trace 已经确认：
    - `runtime_order_tokens`
    - `execution_order_tokens`
    - `verify.draft_token_ids`
  - 三者之间当前并非“无意混成一团”，而是存在显式映射关系；
  - 但 next-round root 输入仍表现出 execution rows 残留，因此可疑点已从“有没有映射”收敛为“映射之后哪些数据应该被裁掉却没被裁掉”。
- 下一步验证：
  - 输出：
    - canonical node list
    - packed row index -> node path 映射
    - verifier row index -> node path 映射
  - 三者逐项对齐。

### H9: drafter 内部 query packing 顺序可能与 runtime / verifier 顺序不一致

- 状态：`验证中`
- 位置：
  - `vllm_ascend/spec_decode/llm_base_proposer.py`
- 怀疑内容：
  - root phase 的 trace 中，后续步的 `model_input_ids` 排布看起来并不直观；
  - 例如 step 1 已观察到：
    - `root_input_ids = [103987]`
    - `tree_root_token_ids = [[334, 5616]]`
    - 但 `model_input_ids` 中出现的顺序是 `[103987, 374, 334, 5616, 55786, ...]`
  - 这提示：
    - drafter 实际消费的 query token 排布
    - 未必与 runtime canonical 顺序或 execution 顺序一一对应；
    - 或者至少当前 trace 还不足以证明它们一致。
- 支持证据：
  - 这是当前最具体的新异常信号；
  - 如果 drafter 输入顺序和 verifier 假设顺序不一致，就会造成“前几步看似能跑，后面逐步串味”的症状。
- 已验证事实：
  - 新 trace 已显示每一轮的：
    - `root.tree_root_token_ids`
    - `level.cumulative_input_ids`
    - `children.next_level_tokens`
    - `result.runtime_order_tokens`
    - `result.execution_order_tokens`
  - 这些内部结构彼此是可以对齐解释的；
  - 但是下一轮 `root.model_input_ids` 会继承上一轮 execution-order 的残留 tokens，而不只是 accepted frontier。
- 反证：
  - “drafter 内部顺序完全随机/完全不可解释”目前没有证据支持。
- 当前更具体的怀疑：
  - 问题不一定在 drafter 当前轮的 tree 展开；
  - 更可能在“上一轮验证结束后，下一轮 drafter 准备输入时，没有正确裁剪只应保留的 frontier token / hidden state / row”。
- 新增结论：
  - 当前已确认“残留 token 的来源”不是随机 packing，而是 `set_inputs_first_pass()` 的默认线性左移逻辑。
- 下一步验证：
  - 对照：
    - `prepare_next_token_ids_padded()`
    - `prepare_inputs_padded()`
    - `root.model_input_ids`
  - 确认 next-round drafter 输入里每个残留 token 的来源。

### H8: attention backend 虽接收了 tree mask，但内部并未按预期生效

- 状态：`待验证`
- 位置：
  - `vllm_ascend/attention/attention_v1.py`
  - `vllm_ascend/attention/attention_mask.py`
- 怀疑内容：
  - Python 层构造正确，但 backend 内部可能：
    - 截断
    - 重排
    - 广播错误
    - 或按非预期格式读取
- 支持证据：
  - Ascend attention 路径和上游 GPU backend 不同，mask 兼容层本身就是独立风险点。
- 反证：
  - 暂无。
- 下一步验证：
  - 在最小输入上抓 backend 前最后一跳的 attention metadata；
  - 对照 backend 真实使用的 mask tensor shape / dtype / values。

## 已做的试探性修改登记

### E1: 移除 first-pass proposer path 中的一个 tree target mask patch

- 状态：`已修改，未验证`
- 文件：
  - `vllm_ascend/spec_decode/llm_base_proposer.py`
- 动机：
  - 怀疑第一轮 drafter 过早套用了 target tree mask，污染 root hidden states。
- 当前结论：
  - 尚无。
- 处理原则：
  - 在下一轮实验完成前，不能把这次修改当作“有效修复”。
  - 如果实验无收益，需要在本文档中明确标为无效或恢复基线。

## 验证顺序

按当前信息，建议优先级如下：

1. H2: target metadata / first-pass input 裁剪阶段是否仍残留 chain 假设
2. H9: drafter 内部 query packing 顺序是否与 runtime / verifier 一致
3. H7: canonical order 与物理 row order 是否一致
4. H4: tree query position / slot offset 是否正确
5. H1: verifier row mapping 是否正确
6. H5: tree target mask 本身的语义与极性
7. H8: backend 对 mask 的实际消费是否符合预期
8. H6: first-pass mask patch 移除实验是否真的改善

排序理由：

- 当前现象更像“状态推进或语义解释错位”，而不是“attention 完全不可用”；
- 因此优先查 runtime / metadata / verifier / feedback loop；
- mask/backend 放在后面，但保留为高风险备选项。

## 下一步具体动作

下一轮调试先做以下几件事：

1. 继续检查 `set_inputs_first_pass()` 对 hidden_states / positions 的处理，确认是否与 token 残留同源。
2. 评估 tree 模式 first-pass 需要保留的最小状态：
   - accepted frontier token
   - bonus token
   - 对应 hidden state
   - 对应 position / slot
3. 再决定是扩展现有函数，还是为 tree 模式单独实现 first-pass 输入重建。
4. 实验结束后立刻回填本文档：
   - 哪个假设被增强
   - 哪个假设被削弱
   - 哪个需要改状态

## 当前阶段结论

截至目前，最强结论只有两条：

1. 上游 `vllm-spec-tree-v023` 只提供了 `speculative_token_tree` 的配置/契约入口，并没有可直接复用的完整 runtime tree 实现。
2. 当前输出质量问题更大概率在 Ascend 本地 tree runtime / verifier / metadata / state feedback 语义，而不是普通 baseline 推理路径。

## 2026-06-29 接续会话增量记录

### 新原则

- 本轮起，任何新的怀疑点都先登记到本文档，再去跑验证。
- 当前 attention 相关老怀疑点如果没有新反证，一律保持降权，不再重复扩 trace。

### 最新确认：vendor rowwise fallback 已经修掉一部分主问题

- `vllm_ascend/attention/tree_attention.py```
  中的 non-row0 vendor fallback 已经接入并通过长文本主复现验证。
- 结果不是“完全修好”，但已把首个可见分叉从旧版 `first_diff_index = 11` 推后到 `20`。
- 这说明：
  - non-row0 attention 数值路径之前确实是主问题之一
  - 当前这部分修正不应回退

### 当前最高优先级怀疑点：多 token accept 后的 next-root materialization

- 最新 sample 对账文件：
  - baseline:
    `/tmp/baseline_sample_after_vendor_v1.jsonl`
  - tree:
    `/tmp/tree_sample_after_vendor_v1.jsonl`
- 关键对齐：
  - tree `step16` 一次产出 `[20412, 99489]`
  - 其中：
    - tree `step16 bonus` 对齐 baseline `step18 sample`
    - tree `step16 row_label=[1]` 对齐 baseline `step19 sample`
  - 但紧接着：
    - tree `step17 bonus/row0 = 31235`
    - baseline `step20 sample = 97145`

这组现象的含义是：

- 当前剩余问题不再像“non-row0 attention 先漂，再慢慢传染 row0”
- 更像是：
  - 一次接受多个 token 之后
  - next-root 的 first-pass replay
  - 或 async `input_batch` 状态 materialization
  - 在进入下一轮 root 时已经错位

### 当前待验证项

状态：`待验证`

下一轮只做这一件事，不再同时扩别的怀疑点：

1. 重跑长文本 32B eager tree-only case。
2. 同时打开：
   - `VLLM_ASCEND_TREE_FIRST_PASS_TRACE_PATH`
   - `VLLM_ASCEND_TREE_STATE_TRACE_PATH`
   - `VLLM_ASCEND_SAMPLE_TRACE_PATH`
3. 只盯：
   - `tree step16 -> step17`
4. 重点核对字段：
   - `accepted_paths`
   - `terminal_row_indices`
   - `rebuilt_token_ids`
   - `rebuilt_positions`
   - `rebuilt_seq_lens`
   - `prev_sampled_token_ids`
   - `spec_token_ids`
   - `token_ids_cpu_computed_tail`
   - `token_ids_cpu_no_spec_tail`
   - `num_computed_tokens`
   - `optimistic_seq_lens_cpu`

### 当前判定规则

- 如果 `step16 -> step17` 之间的 first-pass / state 字段已经完全 self-consistent，
  再回头怀疑更深层 hidden/logits 或 sampler。
- 如果这些字段存在：
  - 少 replay
  - 多 replay
  - token 顺序错
  - seq_len 没推进
  - CPU 视图仍残留 placeholder
  那么当前应优先修 runtime state 链，而不是继续补 attention/kernels。

### 最新反证：next-root 的表面输入其实已经对了

状态：`已验证`

在当前代码上又补做了一轮定向 `tree target trace`，只看问题点附近的 target 输入。

关键结果：

- `tree step16` 对齐 baseline `step18/19` 仍成立：
  - `bonus` 行完全等于 baseline `step18`
  - `row_label=[1]` 完全等于 baseline `step19`
- 更关键的是下一轮 target batch：
  - row0 `input_id = 99489`
  - row0 `position = 79`
  - row0 `slot_mapping = 207`

含义：

- “next-root 把 token / position / slot 直接喂错了”这条说法不再准确。
- 当前代码至少已经把问题批次的表面输入重建对了。

### 新主怀疑点：promoted terminal row 的 canonical KV / slot 收束缺口

状态：`待验证`

虽然表面输入已对，但 tree 仍然在下一轮立刻偏成：

- baseline `step20 sample = 97145`
- tree `step17 bonus = 31235`

这更像是：

- accepted terminal row 被提升为 root 时
- `input_id / position` 是对的
- 但它对应的 canonical prefix KV / slot 没有真正收束到连续历史中

这条怀疑点尤其集中在：

- `terminal_row_indices = [2]`
- 也就是接受的不是第一个 depth-1 row，而是分叉树里的另一个 sibling

因为当前已知现象是：

- 早期 `terminal_row_indices = [1]` 的场景可以继续对齐更久
- 但这次 `terminal_row_indices = [2]` 之后，下一轮 row0 bonus 立刻分叉

### 新增代码观察：`tree_first_pass` 当前没有显式 canonical re-materialize

状态：`已验证`

直接复核：

- `vllm_ascend/spec_decode/tree_first_pass.py`
- `build_tree_first_pass_inputs()`

当前实现对 promoted row 的处理是：

- `positions = index_select(terminal_row_indices)`
- `hidden_states = index_select(terminal_row_indices)`
- `slot_mapping = index_select(terminal_row_indices)`

也就是说：

- promoted row 当前是把上一轮 target batch 中选中的那一行“原样提成 root”
- 代码里没有显式执行 accepted-path 的 canonical re-materialize / repack

注意：

- 这不等于现实现一定错误
- 因为选中行本身的物理 slot 也可能正好就是后续连续 decode 要继续使用的 slot

但它足以把当前怀疑点进一步收缩成：

- 若这里出错，
  更可能是“被提成 root 的 full-prefix 视图并没有真正等价于 baseline 连续前缀”
- 而不是 token / position / slot 单字段本身填错

### 最新验证结果：continuity 的真实断点是 `prefix slot 206 / position 78`

状态：`已验证`

为了验证上面的代码观察，我又补了一轮更晚步数的统一 schema trace：

- baseline:
  - `/tmp/baseline_prefix_trace_late_v1.jsonl`
- tree:
  - `/tmp/tree_prefix_trace_late_v1.jsonl`

对齐点选成：

- baseline `step19`
- tree `step16`
- `current_input_id = 99489`
- `seq_len = 80`
- `current_slot = 207`

也就是正好落在：

- 一次多 token accept 之后
- next root 继续向前推进
- 但尚未再次显式分叉成更长文本之前

程序化结果非常关键：

- 一致的部分：
  - `current_input_id`
  - `current_input_position`
  - `current_slot`
  - `current_q`
  - `current_k`
  - `current_v`
- 首次分叉的部分：
  - full-prefix 中只有最后一个 prefix entry 出现明显差异
  - 对应：
    - `prefix slot = 206`
    - `position = 78`
  - 数值：
    - `prefix_tail_k_heads.max_diff = 26.0`
    - `prefix_tail_v_heads.max_diff = 0.322265625`
  - 进一步后果：
    - 同一轮 `current_output_heads.max_diff = 0.023468017578125`

这条证据说明：

- continuity 的断点并不在 current root `slot 207`
- 而是在它前一个 canonical prefix slot `206`
- 这个位置正好对应：
  - 一次多 token accept 中
  - 第一个 accepted draft token 应该被落到的连续前缀位置

因此当前最强结论应更新为：

- 不是“promoted root 本身喂错”
- 而是“accepted draft token 没有从 speculative physical slot materialize 到 canonical prefix slot”

换句话说：

- `build_tree_first_pass_inputs()` 里直接复用 terminal row 的 `slot_mapping`
  只是症状的一部分
- 真正缺失的是 accepted-path KV materialization
- 特别是：
  - accepted draft token 要左移压入连续前缀
  - bonus token 才能继续作为新的 current token 占用下一个 canonical slot

### 下一轮验证目标

1. 当前“是否断 continuity”已经不是待验证项，而是已证实。
2. 下一步直接改：
   - accepted-path KV materialization
2.5 本轮实现约束已经固定：
   - `accepted_paths` 不能只在 sampler 内部消费后丢掉，
     需要挂到 runner，供下一轮 drafter 前的 KV 修正使用
   - accepted draft token 的 source 不是 `terminal_row_indices` 本身，
     而是每个 accepted runtime node 对应的 execution row
   - destination slot 必须由 token 语义 position + block table 计算，
     不能继续沿用 speculative row 的物理 slot
   - 拷贝顺序必须先 gather 后 scatter，
     否则 accepted path 从右侧 speculative row 压回左侧连续前缀时会自覆盖
   - bonus token 不参与 prefix materialization；
     它只应该在 next-root first pass 中成为新的 current token
3. 最小修复目标：
   - 把 accepted draft tokens 从 speculative physical slots
     拷到 canonical contiguous prefix slots
   - bonus token 继续作为新的 current token 使用下一个 canonical slot
4. 修后立刻复验：
   - baseline / tree 对齐 `current_input_id = 99489`
   - 检查 `prefix slot 206 / position 78` 的 K/V 是否恢复一致
   - 再看 `current_output_heads` 和最终 token 是否一起恢复

### materialization v1 复验结果：端到端输出暂时没有变化

accepted-path KV materialization 的第一版代码已经接入，并重跑了真实 32B eager 长文本对比。

结果文件：

- `/tmp/qwen32_tree_eager_quick_check_noasync.json`

观测结果：

- baseline 仍正常
- tree 仍然：
  - `first_diff_index = 20`
  - diff token 仍是：
    - baseline: `97145`
    - tree: `31235`

这条结果的含义是：

- 当前 root-cause 方向没有被推翻
- 但 materialization v1 还没有真正传导到最终有效状态

所以当前下一步不是回退 hypothesis，
而是优先验证 helper 自身：

1. 它是否在问题步真的执行了。
2. 它是否真的规划出了 `source_slot=207 -> destination_slot=206` 这样的 copy。
3. 它做完 copy 之后，是否又被后续链路覆盖。

除此之外，其余都仍属于待验证假设，不能提前下结论。

### 新增待验证假设：late-step `9370` 对齐到 tree speculative row `[0]`

状态：`已验证`

基于已经拿到的两类日志：

- sample trace：
  - baseline `step 73 -> sampled 9370`
  - tree `step 66 -> sampled [9370, 105355]`
  - 且同一 step 的 speculative `row [0]` 已经把
    `104963` / `105355` 的排序翻转
- block trace：
  - baseline 中已经有 `current_input_ids = [9370]` 的记录
  - tree `index = 65` 的 query rows
    对应 `positions = [132, 133, 133, 134, 134, 134]`
    与 `slots = [260, 261, 262, 263, 264, 265]`

当前新增的工作假设是：

- baseline block trace 里 `current_input_ids = [9370]` 的那一轮
- 很可能对应 tree block trace `index = 65` 的第二个 query row
  也就是 sample trace 里的 speculative `row [0]`

验证结果如下：

1. 对齐成立：
   - baseline block trace `index = 73`
     - `current_input_ids = [9370]`
     - `position = 133`
     - `slot = 261`
   - tree block trace `index = 65` 的第二个 query row
     - `current_input_ids = 9370`
     - `position = 133`
     - `slot = 261`
2. `layer0` 上没有分叉：
   - `layer_input_hidden = 0.0`
   - `pre_attn_hidden = 0.0`
   - `attn_output = 0.0`
   - `post_attn_hidden = 0.0`
   - `mlp_output = 0.0`
   - 对应 norm 也都是 `0.0`

这条验证很重要，因为它把一个自然但错误的直觉排除了：

- 虽然 sample trace 里 `step 66` 的 speculative `row [0]`
  已经把 `104963` / `105355` 的排序翻转
- 但 block trace 对齐后说明，
  这个翻转不是在“输入 token = 9370 的这一行 `layer0`”
  就已经产生

因此下一条候选对齐改成：

- baseline block trace `index = 74`
  - `current_input_ids = [104963]`
  - `position = 134`
  - `slot = 262`
- tree block trace `index = 66` 的 bonus row
  - `current_input_ids = 105355`
  - `position = 134`
  - `slot = 262`

下一步只验证一件事：

- 在共享 `position = 134 / slot = 262` 的条件下，
  `layer0` 的首个非零差异究竟出现在
  `layer_input_hidden`、`pre_attn_hidden`，
  还是 `attn_output` 之后。

### 后续修正：`104963` vs `105355` 这对比较不能用于定位首个根因

状态：`已验证并否决其定位价值`

我随后确实做了这组比较，结果是：

- baseline `index = 74`
  - `token = 104963`
  - `position = 134`
  - `slot = 262`
- tree `index = 66` bonus row
  - `token = 105355`
  - `position = 134`
  - `slot = 262`

观测结果：

- `layer_input_hidden` 就已经明显不同
- 后续 `pre_attn_hidden / attn_output / post_attn / mlp_output`
  差异继续放大

这并不奇怪，因为这已经是：

- 上一轮 sample 分叉之后
- 两边拿着不同 token 进入下一轮 root

所以这组比较只能说明：

- token 级分叉已经真实传导到了下一轮

不能说明：

- 首个根因发生在 `layer0`

因此当前正确的下一步重新收敛为：

- 保持真正有效的对齐点：
  - baseline `index = 73`
  - tree `index = 65 row [0]`
- 继续补 `layer1+` block trace
- 找到这对“同输入、同 position、同 slot”的调用
  在更深层首次出现非零差异的位置

### 新进展：这对真正对齐的调用在 `layer1 self_attn` 输出处首次出现非零差异

状态：`已验证`

为了避免继续凭想象讨论“也许在更深层”，
我补了串行的 `layer1` block trace：

- baseline:
  - `/tmp/baseline_block_trace_l1_v1.jsonl`
- tree:
  - `/tmp/tree_block_trace_l1_v1.jsonl`

继续使用已经验证过的对齐点：

- baseline `index = 73`
  - `token = 9370`
  - `position = 133`
  - `slot = 261`
- tree `index = 65 row [0]`
  - `token = 9370`
  - `position = 133`
  - `slot = 261`

对账结果：

- `layer_input_hidden = 0.0`
- `layer_input_residual = 0.0`
- `pre_attn_hidden = 0.0`
- `attn_output.max_diff = 0.0068359375`
- `post_attn_hidden.max_diff = 0.0078125`
- `post_attn_residual.max_diff = 0.0068359375`
- `mlp_output.max_diff = 0.008026123046875`

另外我还回查了 hook 定义：

- `patch_qwen3vl.py` 里 block trace 记录的 `attn_output`
  来自 decoder block 中的
  `attn_output = self.self_attn(...)`
- 也就是说，
  这不是 layernorm 之后的量，
  而是 `layer1 self_attn` 的直接输出

这条证据把当前 root-cause 范围再次收缩为：

- `layer0` 不是首个分叉点
- 同输入的 `layer1` 调用里，
  首个可观测非零差异已经出现在 `self_attn` 输出

因此下一步优先级应该改成：

1. 先区分：
   - `layer1 attention impl` 本身算出了不同输出
   - 还是它读到的更老 prefix K/V 已经不同
2. 只有这一步查不清，再继续往 `layer2+` 扩散

### 下一条待验证分流：给 `layer1` 补 attention prefix trace

状态：`待验证`

当前 block trace 已经说明：

- 同输入的这对调用在 `layer1 self_attn` 输出处首次出现非零差异

但 block trace 还不能回答：

- 是 `layer1 qkv`/prefix 输入面先分叉
- 还是 `layer1 self_attn` 在同输入下算出了不同输出

因此下一步固定为：

1. 对同一 prompt 串行补 `layer1` attention prefix trace。
2. 仍然只看这对已经验证过的对齐点：
   - baseline `index = 73`
   - tree `index = 65 row [0]`
3. 直接比较：
   - `current_q/k/v`
   - `prefix_tail_k/v`
   - `current_output`

判别规则：

- 如果 `current_q/k/v` 已不同：
  - 问题前移到 `layer1 qkv` 投影或更早状态
- 如果 `current_q/k/v` 与可见 prefix 都一致，
  但 `current_output` 仍不同：
  - 问题更接近 `layer1 attention impl`
  - 或 prefix 中更老、未被 tail 覆盖的历史 K/V

### 首轮 `layer1` prefix trace 失败原因：source filter 配置写错

状态：`已验证`

我随后按上面的思路去重跑 `layer1` 的 attention prefix trace，
但出现了一个纯 instrumentation 问题：

- baseline 正常生成并退出
- 但 `/tmp/baseline_prefix_trace_l1_v1.jsonl` 并没有产生
- tree 侧还没进入分析前，
  就已经可以确认 baseline 没有任何 trace 落盘

回查 `attention_v1.py` 后确认：

- decode 路径调用 `_maybe_dump_prefix_cache_trace(...)` 时
  传入的 `trace_source` 是：
  - `decode_fused`

而我这轮环境变量写的是：

- `VLLM_ASCEND_PREFIX_CACHE_TRACE_SOURCES=fused_infer,decode_only,common`

这意味着：

- 过滤条件大概率把真实的 decode 记录全挡掉了

所以这一步当前不能拿来解释模型，
只能得出一个配置层面的结论：

- 下一轮要先把 source filter 修正为包含 `decode_fused`
- 再重跑 baseline/tree 的 `layer1` prefix trace

### 新认识：`prefix trace` 不适合直接观察 speculative `row [0]`

状态：`已验证`

修正 `decode_fused` 之后：

- baseline `layer1` prefix trace 已经成功落盘：
  - `/tmp/baseline_prefix_trace_l1_v2.jsonl`
- tree `layer1` prefix trace 也成功落盘：
  - `/tmp/tree_prefix_trace_l1_v2.jsonl`

但新问题也随之明确了：

- tree 文件里的记录源是：
  - `trace_source = tree_row0_decode`
- 也就是说，它看到的是 root row 的 decode override
- 它仍然看不到当前真正要对齐的：
  - speculative `row [0]`
  - 也就是晚步数 `9370 / position 133 / slot 261` 这一行

因此当前最合理的下一步不是继续挤 `prefix trace`，
而是直接换工具：

1. 给 `tree exact debug` 增加按层过滤能力。
2. 只抓：
   - `model.layers.1.self_attn.attn`
3. 用它直接导出 multi-row speculative rows 的：
   - `current_q/k/v`
   - `current_output`
   - `prefix_tail_k/v`
4. 再与 baseline `layer1` prefix trace
   做同一步、同行的对账。

### instrumentation 更新：`tree exact debug` 已支持按层过滤

状态：`已完成`

为避免继续拿大文件人工筛选 `layer0..63`，
我给：

- `vllm_ascend/attention/tree_attention.py`
  - `maybe_dump_tree_exact_debug()`

补了两个过滤开关：

- `VLLM_ASCEND_TREE_EXACT_DEBUG_LAYERS`
- `VLLM_ASCEND_TREE_EXACT_DEBUG_ALL_TP`

作用：

- 可以只抓：
  - `model.layers.1.self_attn.attn`
- 默认也可以只保留 `tp_rank = 0`

这样下一轮就能在不改功能逻辑的前提下，
直接拿到 `layer1` 的 speculative rows，
再与 baseline `layer1` prefix trace 做 row-level 对账。

### 接续恢复记录：先用现有 `layer1` 日志回答输入面 vs 输出面分流

状态：`待验证`

恢复本轮上下文后，先确认上一轮按层过滤的 exact debug 任务已经完整结束：

- baseline:
  - `/tmp/baseline_prefix_trace_l1_v2.jsonl`
- tree:
  - `/tmp/tree_exact_layer1_late_v1.jsonl`

当前恢复后的单一待验证问题固定为：

- 对齐样本：
  - baseline `index = 73`
  - tree `index = 65 row [0]`
  - 共享 `token / position / slot = 9370 / 133 / 261`
- 现有 block trace 已知：
  - `layer0` 全链路 `0.0`
  - `layer1 self_attn` 返回值开始出现非零差异
- 但还没最终回答：
  - `layer1` 的 `current_q/k/v` 是否已经先分叉
  - 还是 `current_q/k/v` 仍一致，而 `current_output` 才首次分叉

本轮执行纪律：

1. 先解析现有日志，不直接重跑 4 卡任务。
2. 只有当这两份日志无法回答上述分流时，才继续补 trace。
3. 解析结论必须回填本文档，不能停留在临时脚本或脑内判断。

判定规则：

- 若 `current_q/k/v` 已分叉：
  - 下一步前移到 `layer1` 之前的 block 链路
- 若 `current_q/k/v` 仍一致、但 `current_output` 已分叉：
  - 下一步优先下钻 `layer1 attention impl`
    或更老 prefix K/V

### 现有 `layer1` 日志的直接结论：`q/k/v` 没漂，最近几格 prefix K/V 先漂

状态：`已验证`

按上面的纪律，先只解析现有日志：

- baseline:
  - `/tmp/baseline_prefix_trace_l1_v2.jsonl`
- tree:
  - `/tmp/tree_exact_layer1_late_v1.jsonl`

对齐结果如下：

- baseline 命中样本：
  - `step = 73`
  - `current_input_ids = [9370]`
  - `current_input_positions = [133]`
  - `current_slots = [261]`
- tree 命中样本：
  - `index = 65`
  - `current_slots = [260, 261, 262, 263, 264, 265]`
  - 其中 `query_pos = 1`
    对应同一逻辑行 `slot = 261`

先看当前 token 自身：

- `current_q.max_diff = 0.0`
- `current_k.max_diff = 0.0`
- `current_v.max_diff = 0.0`
- `current_output.max_diff = 0.00013399124145507812`

这一步已经说明：

- `layer1` 的当前 query/value 构造还没有分叉
- `current_output` 的 tiny diff 不是由当前 token 的 `q/k/v` 自身差异直接引起

再看这条 query 真正可见的最近 prefix：

- baseline 可见 prefix slots：
  - `253..260`
- tree 对齐后的可见 prefix slots：
  - `253..259` 来自 `prefix_tail`
  - `260` 来自当前 batch 里的前一行 `row0`

逐 slot 对账结果：

- `slot 253..255`
  - `K/V` 全部 `0.0`
- `slot 256`
  - `K max diff = 0.125`
  - `V max diff = 0.00225830078125`
- `slot 257`
  - `K max diff = 0.046875`
  - `V max diff = 0.0029296875`
- `slot 258`
  - `K max diff = 0.03125`
  - `V max diff = 0.00286865234375`
- `slot 259`
  - `K max diff = 0.0625`
  - `V max diff = 0.002838134765625`
- `slot 260`
  - `K max diff = 0.125`
  - `V max diff = 0.00341796875`

这组证据的含义需要明确写死：

- 当前 residual divergence 不是
  “`layer1 attention impl` 在完全同输入面下凭空算出不同输出”
- 真正先漂的是：
  - 当前 token `9370` 所消费的最近几格 `layer1 prefix K/V`
- `current_output` 的 tiny diff
  是这段已漂 prefix 历史的直接后果

因此下一步优先级应从：

- 下钻 `layer1 attention impl`

切换为：

- 先把 `slot 256..260` 映射回具体历史 token / step
- 再追这些历史 token 是在哪一轮开始把 `layer1 K/V` 写偏
- 然后再回到它们各自的
  `layer0 -> layer1`
  block 边界定位首个真实根因

### `slot 256..260` 已映射回具体历史 token

状态：`已验证`

继续只用现有 baseline `layer1` trace 反查：

- `slot 256 -> step 68 -> token 9370 -> position 128`
- `slot 257 -> step 69 -> token 104978 -> position 129`
- `slot 258 -> step 70 -> token 102832 -> position 130`
- `slot 259 -> step 71 -> token 26939 -> position 131`
- `slot 260 -> step 72 -> token 100390 -> position 132`

这一步的意义是：

- 后续不再只说“最近几格 prefix 有 drift”
- 而是可以直接验证：
  - 上述 5 个 token 中，
    到底是谁在自己作为 current row 的那一轮，
    最早把 `layer1 current K/V/output` 写偏

因此下一步最小验证已固定为：

1. 继续复用 `/tmp/tree_exact_layer1_late_v1.jsonl`
   中 `index 61..65` 的 current rows
2. 分别对齐 baseline `step 68..72`
3. 逐个比较：
   - `current_k`
   - `current_v`
   - `current_output`
4. 找出首个非零 token 后，
   再回到那个 token 的 `layer0 -> layer1` 边界定位

### 逐 token current-row 对齐的新结果：最早已观测非零点前移到 `slot 256`

状态：`待进一步前验`

继续只用现有日志，把：

- baseline `step 68..72`
- tree `index 61..65`

按 `current slot` 逐个对齐后，得到：

- `slot 256 -> token 9370 -> position 128`
  - `q_max = 0.04296875`
  - `k_max = 0.125`
  - `v_max = 0.00225830078125`
  - `out_max = 0.000732421875`
- `slot 257 -> token 104978`
  - `q_max = 0.0625`
  - `k_max = 0.046875`
  - `v_max = 0.0029296875`
  - `out_max = 0.00048828125`
- `slot 258 -> token 102832`
  - `q_max = 0.046875`
  - `k_max = 0.03125`
  - `v_max = 0.00286865234375`
  - `out_max = 0.0006103515625`
- `slot 259 -> token 26939`
  - `q_max = 0.04296875`
  - `k_max = 0.0625`
  - `v_max = 0.002838134765625`
  - `out_max = 0.0009765625`
- `slot 260 -> token 100390`
  - `q_max = 0.068359375`
  - `k_max = 0.125`
  - `v_max = 0.00341796875`
  - `out_max = 0.0001220703125`

这说明：

- 最近几格 prefix 的漂移并不是
  “token 在成为 prefix 之后才被污染”
- 至少从当前已观测范围看，
  `slot 256`
  在它自己作为 current row 的那一轮，
  `layer1 current_q/k/v/output`
  就已经开始偏离 baseline

但这还不能直接写成最终结论，因为还缺最后一步：

- 需要继续向前验证
  `slot 253..255`
  在它们自己作为 current row 的那一轮是否仍然 `0.0`

只有这一步也成立，才能把：

- `slot 256 -> token 9370`

升级为当前最早的已证实漂移点。

### 前验完成：`slot 253..255` 仍全等，因此最早漂移点正式收敛到 `slot 256`

状态：`已验证`

继续只用现有日志，把更早三格也做了 current-row 对齐：

- `slot 253 -> token 1773 -> position 125`
  - `q/k/v/output` 全部 `0.0`
- `slot 254 -> token 45181 -> position 126`
  - `q/k/v/output` 全部 `0.0`
- `slot 255 -> token 102640 -> position 127`
  - `q/k/v/output` 全部 `0.0`

因此当前可以把下面这条结论升级为已证实：

- 当前最早的漂移点是：
  - `slot 256`
  - `token 9370`
  - `position 128`

并且它的特征是：

- 在自己作为 current row 的那一轮，
  `layer1 current_q/k/v/output`
  就已经全部不是 exact match

这意味着后续调试方向必须继续收缩：

- 不再横向扫描更多 prefix slots
- 直接围绕
  `slot 256 / token 9370`
  回到它自己的 `layer0 -> layer1` block 边界
- 目标是定位：
  - `layer0` 的哪一段输出开始让
    `layer1 current_q`
    首次出现非零差异

### 恢复会话后的单一待验证假设

状态：`待验证`

恢复会话后，不再新增宽泛猜测，当前只保留一个最窄假设：

- `slot 256 / token 9370 / position 128`
  已经是当前最早的已证实漂移 token
- 因而下一步不再回头怀疑：
  - 更晚的 `layer1 attention impl`
  - 更远的 prefix 大范围错位
  - 或新的 async/runtime state 猜测
- 当前只验证一件事：
  - 它自己作为 current row 的那一轮，
    `layer0` block 路径里
    到底是哪一个字段先从 `0.0` 变成非零

执行顺序固定为：

1. 先只解析现有 block trace：
   - `/tmp/baseline_block_trace_late_v2.jsonl`
   - `/tmp/tree_block_trace_v1.jsonl`
   - `/tmp/baseline_block_trace_l1_v1.jsonl`
   - `/tmp/tree_block_trace_l1_v1.jsonl`
2. 若这些现有 trace 足够回答问题，则直接据此收敛根因
3. 只有现有 trace 不足时，才补新的最小化定向 trace

这样做的目的很明确：

- 避免继续靠记忆推断
- 避免无谓重跑 4 卡任务
- 把“第一次漂移究竟发生在 token 9370 的 `layer0` 哪一段”
  钉成一个可以直接核对的日志问题

### 现有 block trace 已回答：`slot 256` 的首个非零字段在 `layer0 attn_output`

状态：`已验证`

恢复会话后先没有重跑，而是直接解析了现有 block trace：

- baseline:
  - `/tmp/baseline_block_trace_late_v2.jsonl`
  - `index = 68`
- tree:
  - `/tmp/tree_block_trace_late_v2.jsonl`
  - `index = 61`
  - `qpos = 0`

这对样本严格对齐到：

- `token 9370`
- `position 128`
- `slot 256`

逐字段结果如下：

- `layer0`
  - `layer_input_hidden = 0.0`
  - `layer_input_residual = 0.0`
  - `pre_attn_hidden = 0.0`
  - `attn_output = 0.0009765625`
  - `post_attn_hidden = 0.000732421875`
  - `post_attn_residual = 0.0009765625`
  - `mlp_output = 0.00390625`
- `layer1`
  - `layer_input_hidden = 0.00390625`
  - `layer_input_residual = 0.0009765625`
  - `pre_attn_hidden = 0.000732421875`

因此可以把结论再前移一层：

- `slot 256`
  的首个已证实非零字段
  不是 `layer1 current_q`
- 而是它自己那一轮的
  `layer0 attn_output`

这条结论非常关键，因为它把剩余问题面压缩成了两个更窄方向：

1. `layer0` 在这一步读到的某些更老 prefix K/V
   已经先轻微漂移
2. `layer0` attention exact path
   在同输入面下仍存在 residual 数值路径差异

下一步必须围绕这两条做验证，
而不是再回到泛化的 `layer1` 或 async state 猜测。

### 现有日志不足点与下一轮最小化重跑计划

状态：`待执行`

现有 `/tmp` 证据已经够回答：

- `slot 256` 的首个 block-level 非零字段是 `layer0 attn_output`

但还不够回答：

- 这一轮 `layer0` 消费到的 prefix K/V
  是否已经先漂
- 还是 exact path 在同输入面下自己算出了不同输出

缺口很具体：

- 现有 `layer0` prefix trace
  没有覆盖晚步数 `slot 256..261`
- 现有 `tree_exact*.jsonl`
  也没有覆盖这一段的 `layer0` 多行细节

因此下一轮重跑必须保持最小化：

1. 不改业务功能逻辑
2. 继续复用现有 instrumentation 开关
3. 串行重跑同一个 `32B eager noasync` 长文本 case
4. 只抓：
   - baseline `model.layers.0.self_attn.attn`
     的 `decode_fused` prefix trace
   - tree `model.layers.0.self_attn.attn`
     的 exact debug
5. 把 `tail/head` 拉大到能直接覆盖
   `prefix_len = 128`

这样下一轮就能直接回答：

- 若 `layer0 prefix K/V` 已先非零，
  根因继续前移到更老历史 token
- 若 `layer0 prefix K/V` 仍全等，
  则 `layer0 exact attention` 本身仍有 residual 差异

### `layer0 prefix K/V` 已被证伪：remaining drift 收缩到 exact attention 本身

状态：`已验证`

按上一节计划补完定向重跑后：

- baseline:
  - `/tmp/layer0_prefix_compare_v1.jsonl`
  - `step = 68`
- tree exact:
  - `/tmp/tree_exact_layer0_256_v1.jsonl`
  - `index = 61`
  - `row0`

对齐点：

- `slot 256`
- `token 9370`
- `position 128`
- `prefix_len = 128`

直接比较结果：

- `prefix_tail_slots 128..255` 全部一一对齐
- `prefix_tail_k/v` 全部 `0.0`
- `current_q = 0.0`
- `current_k = 0.0`
- `current_v = 0.0`
- `current_output = 6.103515625e-05`

同时 tree 自证也仍然成立：

- `current_k_max_diff(tree direct vs gathered) = 0.0`
- `current_v_max_diff(tree direct vs gathered) = 0.0`

这条结果的含义非常明确：

- 当前 remaining drift
  已经不是：
  - 更老 `layer0 prefix K/V` 先漂
  - cache gather 读错
  - current row `q/k/v` 写错
- 而是：
  - 在可见输入完全一致的前提下，
    `layer0 exact attention`
    自己已经算出了一个与 baseline 不同的 output

补充旁证：

- 同一条 tree exact 日志还给出了：
  - `row0_output_max_diff_vs_isolated = 0.0001220703125`

### 三向比较已完成：不是 multi-row exact 独有问题，而是 tree 单行 root decode 就已偏离 baseline

状态：`已验证`

继续只用刚生成的新日志，对三条路径做直接对账：

- baseline `decode_fused`
  - `/tmp/layer0_prefix_compare_v1.jsonl`
  - `step = 68`
- tree `tree_row0_decode`
  - `/tmp/layer0_prefix_compare_v1.jsonl`
  - `step = 61`
- tree exact `row0`
  - `/tmp/tree_exact_layer0_256_v1.jsonl`
  - `index = 61`

关键结果：

- `baseline vs tree_row0_decode current_output = 6.103515625e-05`
- `baseline vs tree exact row0 current_output = 6.103515625e-05`
- `tree_row0_decode vs tree exact row0 current_output = 0.0`

并且对 `baseline vs tree_row0_decode` 再核一次输入面：

- `prefix_tail_k/v = 0.0`
- `current_q = 0.0`
- `current_k = 0.0`
- `current_v = 0.0`

所以当前结论必须改写为：

- 这不是 multi-row exact attention
  独有的问题
- tree 侧单行 root decode 路径
  在同输入面下就已经与 baseline 有稳定微差

因此接下来最窄、最值得查的地方不再是：

- tree exact 的 row interaction

而是：

- `decode_fused` vs `tree_row0_decode`
  的 metadata / kernel-call 差异
- 尤其要看：
  - `_build_tree_root_decode_metadata()`
  - `actual_seq_lengths_q/kv`
  - `attn_state`
  - `block_tables / slot_mapping` 的切片方式

### 新的最强怀疑点：tree row0 被硬编码走 paged path，而 baseline 同一步走的是 decode_fused

状态：`待验证`

继续回看代码与刚落盘的新日志后，当前最强怀疑点已经比“metadata 某字段不同”更直接：

- baseline 在 `slot 256` 这一步的 trace source 是：
  - `decode_fused`
- tree row0 的 trace source 是：
  - `tree_row0_decode`

而对应代码正是：

- `forward_tree_exact_attention()`
  在覆盖 row0 output 时，
  当前直接调用：
  - `forward_paged_attention(...)`

这意味着：

- 即便 `prefix/current_qkv` 完全一致
- tree row0 仍然可能只是因为走了不同 kernel
  才产生稳定的 `6.103515625e-05` 微差

因此下一步最合理的功能验证已经固定为：

1. 让 tree row0 override
   复用 baseline decode 的同一 dispatch 逻辑
2. 若 baseline 该条件下会走 `decode_fused`
   tree row0 也走 `forward_fused_infer_attention`
3. 只有命中 `using_paged_attention(...)`
   时才继续走 paged path

若这样一改后：

- `slot 256` 的 `current_output`
  恢复 exact match

则当前主根因就可以升级为：

- tree row0 override 误用了与 baseline 不同的 attention kernel path

### 针对上述怀疑点的代码改动已经落地，等待复跑验证

状态：`待验证`

当前代码基线已变更为：

- 文件：
  - `vllm_ascend/attention/attention_v1.py`
- 改动：
  - `forward_fused_infer_attention()` 支持自定义 `trace_source`
  - tree row0 override 不再硬编码
    `forward_paged_attention()`
  - 现在会按与 baseline decode 相同的 dispatch 条件，
    选择：
    - `tree_row0_fused`
    - 或 `tree_row0_paged`

这样下一轮复跑能直接回答：

- 若最早那条 `6.103515625e-05` 消失，
  则当前主根因基本成立
- 若仍存在，
  则需要继续下钻 fused path 内部或 metadata 更深字段

### 第一轮 fused 验证失败，但错误点已经明确：row0_metadata 缺 `attn_mask`

状态：`已验证失败因子`

按上节改动做第一轮复跑后：

- baseline 正常完成
- tree 在：
  - `forward_tree_exact_attention()`
  - `-> forward_fused_infer_attention()`
  - `-> torch_npu.npu_fused_infer_attention_score(...)`
  处直接报错

关键错误：

- `When attnMask is not provided, sparseMode must be 0`

这条错误的价值很高，因为它说明：

- tree row0 已经真正进入 fused path
- 当前失败不是 dispatch 条件没命中
- 而是：
  - `row0_metadata`
    还没有带 baseline decode 所需的
    `attn_mask`

因此下一步不应回退当前方向，
而应继续补最小必要 metadata：

1. 优先给 `row0_metadata` 带上正确的 `attn_mask`
2. 若仍报错，再继续补 fused path 还依赖的其余 decode metadata
3. 然后再回到 `slot 256 current_output` 的 exact 对账

### `row0_metadata.attn_mask` 修正已落地，等待第二轮 fused 验证

状态：`待验证`

当前代码基线又往前一步：

- `row0_metadata.attn_mask`
  已改为复用 `AttentionMaskBuilder`
- `row0_metadata.seq_lens_cpu`
  也同步补齐

下一轮最小化复跑的目标很单一：

- 先确认 tree row0 fused path 不再因为缺 mask 而报错
- 然后再看 `slot 256` 的 `current_output`
  是否比上一轮更接近 baseline
