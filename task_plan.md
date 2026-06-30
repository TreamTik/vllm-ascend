# Task Plan

## Goal

在 `speculative_token_tree` 已恢复端到端正确性的前提下，系统化识别并优化 Ascend 侧的主要性能瓶颈，形成可执行的优化路线，并在每一轮优化后复验正确性。

## Phases

- [x] Phase 1: 收敛 proposer first-pass 问题，确认下一轮 root 输入不再夹带旧树 rows。
- [x] Phase 2: 补 target 侧 trace，建立 draft / verify / target 三向对齐证据。
- [x] Phase 3: 在 attention 层实现 tree-only 精确执行路径，绕开当前 fused path 的分叉树语义缺陷。
- [x] Phase 4: 用真实 32B eager 用例复验并更新文档。
- [x] Phase 5: 修正 row0 root override 的 kernel-path mismatch，恢复端到端正确性。
- [x] Phase 6: 做 speculative token tree 性能热点归因，给出优化优先级与验证口径。
- [ ] Phase 7: 按优先级逐项优化性能，并在每一步后复验正确性。

## Current Focus

- 2026-06-30 已新增正式 `2048 / 2048` 基线口径：
  - 用户已确认：
    - 输入长度按“测试集平均 2048 token”
    - 输出长度按“测试集平均 2048 token”
    - 主结论允许采用离线脚本
    - 服务化压测作为辅证
  - 当前已落地统一资产：
    - `/root/qwen32_2048_bench_common.py`
    - `/root/run_qwen32_2048_offline_bench.py`
    - `/root/run_qwen32_2048_service_bench.py`
    - `/root/qwen32_2048_bench_dataset.json`
  - 当前对比口径固定为三组：
    - `baseline`
      - 不开 speculative decode
    - `tree_before`
      - 开 tree，`VLLM_ASCEND_TREE_FAST_IMPL=batched_legacy`
    - `tree_after`
      - 开 tree，默认 `batched`
  - 当前输入集已生成完成：
    - 两条请求 prompt token 长度分别为 `2048 / 2048`
    - 平均输入 token 为 `2048.0`
  - 下一步：
    - 先跑离线 `baseline/tree_before/tree_after`
    - 再跑服务化三组辅助验证
    - 最后汇总 QPS / tokens/s / TTFT / TPOT / 精度一致性

- 2026-06-30 当前性能结论又新增了一层“prompt 形态”证据：
  - 已新增多 prompt 基准脚本：
    - `/root/run_qwen32_tree_prompt_bench.py`
  - 默认 5 个复杂 case 的合并结果：
    - `/tmp/qwen32_tree_prompt_bench_default_merged.json`
  - 新增超长强结构 JSON case 结果：
    - `/tmp/qwen32_tree_prompt_bench_json_long.json`
  - 当前 benchmark 结论已明确分层：
    - 自然长文 / 混合说明文：
      - tree 仍常被 proposer/verify 额外成本抵消
    - 时间线 / 固定编号输出：
      - tree 已接近 baseline
    - 强 schema JSON：
      - tree 已出现稳定正向收益
      - `json_dataset`：
        `43.60s -> 41.88s`
      - `json_long_grid`：
        `63.73s -> 50.37s`
        `speedup = 1.265x`
        `acceptance_per_pos ~= [0.526, 0.378]`
  - 这意味着：
    - “attention 已优化完但 tree 还是普遍不快”
      这件事现在已经有更细答案：
      - 不是 attention 主链路还没打通
      - 而是不同 prompt 的 acceptance 天花板差异很大
    - Phase 7 后续若继续推进，
      更值得做的是：
      - proposer / acceptance 路线
      - 面向生产吞吐的非 eager 形态

- 2026-06-30 当前这一轮 attention 大结构优化已拿到阶段性收口：
  - unified all-row exact batched path
    已经在真实 `32B eager + noasync` case 上完成双重验证
  - 显式实验名 `batched_all_rows` 下：
    - baseline `10.10s`
    - tree `10.65s`
    - baseline/tree `108` token 全量一致
  - 默认主线路径 `batched` 下：
    - baseline `10.25s`
    - tree `10.62s`
    - baseline/tree `108` token 全量一致
  - 对应 profile 又进一步证明：
    - `avg_row0_ms = 0.0`
    - 旧的 row0 单独 override 主瓶颈已被移除
    - `avg_total_ms` 已从 `v6` 的 `1.014`
      压到约 `0.206`
  - 当前代码语义也已正式收敛为：
    - `batched`
      - unified all-row exact fast path
    - `batched_legacy`
      - 旧 row0 override 路径
    - `batched_split`
      - 实验路径
    - `reference`
      - debug/exact fallback
  - 这意味着：
    - attention 主线的“大结构”这一轮已经打通
    - 当前 eager/noasync 验证形态里，
      tree 相对 baseline 只剩约 `3.7%` 开销
    - 下一轮若继续追更大收益，
      优先级应更多转向：
      - proposer / acceptance
      - 非 eager / async / graph 生产形态
    - Phase 7 仍保持进行中，
      但 attention 侧不再是第一主矛盾

- 新目标切换说明：
  - 用户已明确选择：
    - 优先优化 `生产形态` 性能
    - 目标是尽量接近真实服务吞吐
    - 不把当前 `eager + async_scheduling=False` 验证路径
      当作唯一优化目标
    - 进一步明确：
      - 优先清理“为了验证正确性而妥协出来的实现路径”
      - 即：先识别 correctness-first 设计导致的结构性慢点，
        再把这些路径和真正的生产路径分离
  - 正确性主路径当前已满足：
    - 32B eager/noasync 长文本 case 下 baseline/tree `108` token 全量一致
    - `slot 256 / token 9370 / pos 128` 的 `layer0 current_output` 漂移已清零
- 当前主问题转为性能：
  - 2026-06-30 最新 `32B eager + noasync` 长文本复验：
    - baseline `generate_seconds = 10.16`
    - tree `generate_seconds = 26.16`
    - baseline/tree `108` token 全量一致
  - 这说明：
    - 当前 batched fast-path 已经在真实长文本 case 上保持 correctness
    - 但 tree 仍约慢 `2.57x`
    - 当前主任务仍是性能，不是 correctness
    - 最近一次带重型 trace 的实测：
      - baseline `generate_seconds = 13.77`
      - tree `generate_seconds = 162.71`
      - 同为 `108` token
    - 同一阶段较轻量的历史实测也表明 tree 仍显著慢于 baseline：
      - baseline `10.06s`
      - tree `28.71s`
  - 因而接下来要区分：
    - 调试 instrumentation 带来的额外开销
    - correctness-first 实现本身的结构性瓶颈
    - 以及哪些优化只对验证路径有益，哪些优化会真实传导到服务吞吐
  - 当前这一轮实现先按“低风险、真实热路径”的顺序落地：
    - 缓存 tree runtime 的不变量与 device tensor 视图
    - 去掉 per-step `build_tree_target_attn_mask()` /
      `build_tree_query_depth_offsets()` /
      `mask.clone().to(device)` 的重复构造
    - 向量化 `_calc_tree_spec_decode_metadata()` 里的
      draft-token 重排
    - 把 accepted-path slot copy 的 block/offset tensor
      从“每个 cache tensor 重建一次”
      收敛成“每次 materialize 只建一次”
    - 收紧 `tree_first_pass` 里的 `.cpu().clone()` 临时量
  - 但基于 2026-06-30 新实测，下一层更大的结构优化优先级已抬升：
    - trace 证明当前模型走的是 `AscendAttentionBackendImpl`
      而不是 `AscendC8AttentionBackendImpl`
    - 因而剩余主瓶颈更可能集中在：
      - `gather_paged_kv_to_ragged()`
      - batched FIA 前对每个 speculative row 复制 prefix
      - row0 额外 decode override
    - 下一步应优先研究：
      - prefix/shared-history 与 tree-local current attention 的分解式执行
      - 即：尽量避免把整段 prefix 为每个 speculative row 重复 materialize
  - 2026-06-30 最新 profile v2 已进一步坐实上述方向：
    - visible-row template cache 生效后
      `nr_plan_ms` 已降到 `0.028ms`
    - 但 `gather_ms` 与 `nr_pack_ms` 仍维持在
      `0.61ms` 量级
    - 因而下一轮实现不再继续停留在
      gather+pack 这一版 batched 路径内部做小修
    - 而要直接推进：
      - `prefix-only FIA`（paged KV + repeated block_table）
      - `current-only FIA`（仅树上当前 token）
      - `lse merge`
    - 目标是从结构上拿掉：
      - full-prefix gather
      - per-row prefix duplication
  - 2026-06-30 新增 split-path 验证已给出阶段性结论：
    - `prefix-only FIA + current-only FIA + lse merge`
      的确能把 non-root 主耗时继续压低
    - 但它与正确 legacy 路径之间仍有
      `1e-3 ~ 1e-2` 级别 attention 输出漂移
    - 该漂移足以让 speculative decode
      在 `first_diff_index = 11` 处提前分叉
    - 因而当前主线不能切到 split path
    - split path 先保留为实验实现
    - 正确优化主线重新回到：
      - 继续压 legacy batched 路径中的
        gather / pack / copy
  - 2026-06-30 恢复上下文后的代码复核又把这条主线进一步收紧：
    - 当前 legacy batched path 里，
      non-root attention 仍是：
      - `paged KV -> ragged dense KV -> packed visible KV`
    - 其中 `ragged dense KV` 只是中间态，
      最终只服务于下一步 visible pack
    - 因而本轮优先级已明确更新为：
      - 直接把 non-root batched path 改成
        `paged KV -> packed visible KV`
      - 跳过 `gather_paged_kv_to_ragged()`
        这层中间 materialization
    - 这条改动的成功标准是：
      - baseline/tree token 全量一致不变
      - `gather_ms` 明显下降
      - tree 端到端生成耗时继续下降
  - 2026-06-30 当前这一轮结果已满足上述成功标准：
    - `gather_ms` 已降为 `0.0`
    - `32B` 真实 compare 下
      tree `20.04s -> 16.50s`
    - baseline/tree `108` token 继续全量一致
  - 当前新的下一刀已更新为：
    - direct paged path 不再每层重建 visible slot 计划
    - 把
      `output_indices / packed_slots / actual_seq_lengths_kv`
      缓存在当前 step 的 runtime / attn_metadata 上
    - 目标：
      - 把 `non_root_stats.plan_ms`
        从当前约 `0.157ms`
        再压回接近旧版 `0.03ms` 量级
  - 2026-06-30 第二轮缓存优化已完成真实验证：
    - `/tmp/qwen32_tree_batched_compare_v5.json`
      下：
      - baseline `10.05s`
      - tree `14.96s`
      - baseline/tree `108` token 继续全量一致
    - `/tmp/tree_fast_profile_v5.jsonl`
      下：
      - `total_ms ≈ 1.010`
      - `non_root_stats.plan_ms ≈ 0.066`
      - `non_root_ms ≈ 0.136`
  - 相对 `v3` 正确主路径，
    当前总效果已达到：
    - tree `20.04s -> 14.96s`
    - 总体约再降 `25.4%`
  - 当前热点判断也随之更新：
    - `gather` 已经被拿掉
    - non-root target tree attention
      已不再是主要瓶颈
    - 若继续追更大收益，
      下一阶段更可能要转向：
      - 更强的 exact fast kernel / 修正 split-path 数值漂移
      - proposer / draft / acceptance 侧
  - 2026-06-30 runtime-shared plan 这轮也已完成：
    - `model_runner_v1` 现在会在 patch 阶段
      每个 tree step 预建一次 shared `TreePagedPlan`
    - attention 层不再负责做 shared-plan owner
    - `/tmp/tree_fast_profile_v6.jsonl`
      已证明：
      - `non_root_stats.plan_ms = 0.0`
  - 这轮端到端结果：
    - baseline `10.23s`
    - tree `15.02s`
    - baseline/tree 继续全量一致
  - 当前因此进一步确认：
    - 若下一轮继续做“大结构”，
      最值得做的 attention 主线是：
      - 消掉 row0 单独 override
      - 做 unified all-row exact tree attention

- 当前主假设已更新：
  - 新增已验证结论：
    - row0 fused metadata owner 修复后，
      `Qwen3-32B-W8A8 + eagle3 + speculative_token_tree`
      在长文本 `noasync + eager` case 上已重新达到
      baseline/tree `108` token 全量一致
    - 同轮 prefix trace 已确认 root override 的真实 trace source 为：
      - baseline: `decode_fused`
      - tree: `tree_row0_fused`
    - 这意味着：
      - 先前的 paged-vs-fused kernel-path mismatch
        极大概率就是晚步数 residual drift 的主根因
      - 该结论现又被 `128-step` 晚步数 trace 直接加固：
        `slot 256 / token 9370 / pos 128`
        在 `layer0` 上的
        `prefix_last_k/v`
        `current_q/k/v`
        `current_output`
        已全部回到 `max_diff = 0.0`
  - 新增运行时待验证假设：
    - `forward_tree_exact_attention()` 给 row0 root override 补 fused 路径时，
      `_build_tree_root_decode_metadata()` 直接访问
      `self.attn_mask_builder / self.model_config`
      触发 `AttributeError`
    - 从 `attention_v1.py` 代码归属看，
      这两个对象属于 `AscendAttentionMetadataBuilder`，
      不属于 `AscendAttentionBackendImpl`
    - 因而 row0 fused metadata 更合理的来源应当是：
      - 直接复用已有 `attn_metadata.attn_mask`
      - 或者在 backend impl 内显式调用
        `AttentionMaskBuilder(device)` 与
        `self.vllm_config.model_config`
        现取标准 causal mask
      - 总之不能再假设 backend impl 自带 builder 侧成员
    - 该假设现已通过端到端复验 + 晚步数局部 trace 双重支持
  - `row0 isolated full-forward` 已证明 tree target row0 在同一 cache 状态下可被单独精确复现，
    batched row0 vs isolated row0 的 hidden/logits 在 16 个 step 上均为 `0.0`
  - 因而“sibling rows 引入的 non-attention batch-dependent drift”不再是当前主假设
  - 新证据又进一步削弱了“更早 prefix/cache 已提前漂移”这个方向：
    - 按 `seq_len=71` 对齐时，baseline/tree 在 `layer0` 上的
      - `current_input_id / position / slot`
      - `prefix_last_k/v`
      - `current_q/k/v`
      - `current attention output`
      都是逐元素一致
  - 剩余偏差更可能位于：
    - `layer0 attention` 之后的同层残差 / `o_proj` / MLP / block output
    - 更深层 transformer block 内部开始出现漂移
    - 或者 layer 之后到 sampler 之前的 hidden/logits 链路仍有状态泄漏
- `accepted path` 未 materialize 回 async `input_batch` 仍保留为弱怀疑点，但不再是当前第一优先级。
  - 当前动作：
    - 新增已验证对齐结论：
      - baseline block trace `index = 73`
        （`current_input_ids = [9370]`, `position = 133`, `slot = 261`）
        确实对应 tree block trace `index = 65` 的第二个 query row
        （sample trace 里的 speculative `row [0]`）
      - 更重要的是，这个对齐点在 `layer0` 上仍完全一致：
        - `layer_input_hidden`
        - `pre_attn_hidden`
        - `attn_output`
        - `post_attn_hidden`
        - `mlp_output`
        以及对应 norm 都是 `0.0`
      - 因而新的 residual divergence
        还没有在“输入 token = 9370 的这一行”发生
    - 新增已证伪对齐：
      - baseline `index = 74` (`104963`, `position = 134`, `slot = 262`)
        对 tree `index = 66` bonus row (`105355`, `position = 134`, `slot = 262`)
        虽然 `position/slot` 相同，但 token 已经不同
      - 这组样本的 `layer_input_hidden` 就已明显分叉，
        只能说明 sample 分叉已经传导到下一轮 root，
        不能用于定位首个根因
    - 当前正确的下一步已改成：
      - 保持真正有效的对齐点不变：
        baseline `index = 73`
        vs tree `index = 65 row [0]`
      - 不再停留在 `layer0`
      - 继续补 `layer1+` block trace，
        找出这对同输入调用的首个深层非零差异
    - `layer1` 的首条结果现已拿到：
      - baseline `index = 73`
        (`9370`, `position = 133`, `slot = 261`)
      - tree `index = 65 row [0]`
        (`9370`, `position = 133`, `slot = 261`)
      - 在 `layer1` 上：
        - `layer_input_hidden = 0.0`
        - `layer_input_residual = 0.0`
        - `pre_attn_hidden = 0.0`
        - `attn_output` 首次出现非零差异：`max_diff = 0.0068359375`
        - `post_attn_hidden / residual / mlp_output`
          继续保持同量级微差
      - 这意味着：
        - 当前 residual issue 已经再次收缩
        - 首个可观测深层分叉不在 `layer0`
        - 而在这对同输入调用的 `layer1 self_attn` 输出处
    - 新增下一条待验证分流：
      - 用 `layer1` 的 attention prefix trace
        直接比较这对对齐调用的：
        - `current_q/k/v`
        - `prefix_tail_k/v`
        - `current_output`
      - 目标是区分：
        - `layer1` 在读入阶段（qkv 或 prefix cache）就已不同
        - 还是 `layer1 self_attn` 在同输入下算出了不同输出
    - 上述分流首轮重跑已发现 instrumentation 条件问题：
      - `baseline/tree` 两侧运行都成功结束
      - 但 `/tmp/baseline_prefix_trace_l1_v1.jsonl`
        与 `/tmp/tree_prefix_trace_l1_v1.jsonl` 都没有生成
      - 代码复核显示：
        - decode 路径的真实 `trace_source` 是 `decode_fused`
        - 先前环境里误写成了
          `fused_infer,decode_only,common`
      - 因而下一轮不应解释模型，
        先修正 trace filter 再重跑
    - 新增更直接的 instrumentation 方案：
      - `prefix trace` 天生只能看到 `decode_fused` / `tree_row0_decode`
      - 它拿不到晚步数里真正要对齐的 speculative `row [0]`
      - 因而下一步改成：
        - 给 `tree exact debug` 增加按层过滤
        - 只抓 `model.layers.1.self_attn.attn`
        - 用它直接导出 multi-row speculative rows 的
          `current_q/k/v/current_output/prefix_tail`
      - 这样就能和 baseline `layer1` prefix trace
        做同一步的 row-level 对账
    - 该 instrumentation 现已接入：
      - `tree_attention.py::maybe_dump_tree_exact_debug()`
        新增了：
        - `VLLM_ASCEND_TREE_EXACT_DEBUG_LAYERS`
        - `VLLM_ASCEND_TREE_EXACT_DEBUG_ALL_TP`
      - 下一步直接用它只抓
        `model.layers.1.self_attn.attn`
        的 speculative rows
    - 新增代码级观察：
      - `vllm_ascend/spec_decode/tree_first_pass.py::build_tree_first_pass_inputs()`
        当前只是对上一轮 target batch 的
        `positions / hidden_states / slot_mapping`
      做 `index_select(terminal_row_indices)` 复用，
      没有显式做 accepted-path 的 canonical re-materialize
    - 这不等于它一定错误，
      但“promoted sibling row 只是原样提成 root”
      已经成为当前最具体的可疑点
  - 围绕对齐后的 `baseline step 10/11` vs `tree step 9/10` 继续缩圈
  - 基于已打通的 row0 isolated trace，转去对账 baseline/tree 在相同逻辑前缀下的 cache / prefix 状态
  - 扩展现有 debug，比较不止最后一个 prefix token，而是整段 prefix tail 乃至完整 row0 历史视图
  - 继续核对 `_EXTRA_CTX` / `set_ascend_forward_context` / runtime state 是否会让 tree 与 baseline 共享不同的历史状态
  - `tree_state_trace` 保留用于旁证，但不再把 `prev_sampled_token_ids / spec_token_ids` 直接当作当前主根因
  - 新增 baseline decode-prefix trace：
    让 baseline 在 decode-only 路径上也打出每层的 `prefix_tail_k/v`、`current_k/v`、slot/block-table 视图，
    然后与 tree exact debug 做同前缀对账
  - 本轮优先落统一 schema 的 `prefix-cache trace`：
    同时覆盖 baseline decode-only 与 tree row0，
    避免后续继续靠不同日志格式人工拼接对账
  - 已确认 baseline eager 当前实际走 `forward_fused_infer_attention` 而不是 `forward_paged_attention`，
    所以下一轮需把统一 trace 再补到 fused decode 路径
  - 新主结论：
    按 `seq_len` 对齐后，baseline/tree 在 `layer0` 上直到 `seq_len=71` 都完全一致，
    包括当前轮 attention output；
    `seq_len=72` 的分叉只是前一轮最终 token 已不同的后果
  - 下一轮最小验证改为：
    不急着新增 hook，
    先复用现有 `prefix-cache trace` 去看更深层 attention：
    - 如果 `layer1 self_attn.attn` 的 `current_q` 已分叉，
      说明问题发生在 `layer0 attention` 之后、`layer1 attention` 之前
    - 如果 `layer1` 仍一致，就继续向更深层推进
    - 这样可以先用“attention 边界探针”缩小范围，再决定是否要对
      `o_proj / post_attention_layernorm / mlp / block output` 单独加 hook
  - `layer1` 探针现已得到首条有效结果：
    - `seq_len=71` 时 baseline/tree 的
      `current_input_id / position / slot` 与 `current_q/k/v` 仍完全一致
    - 但 `current_output_heads` 已出现极小非零差异：
      `max_diff = 1.8310546875e-04`
    - 下一步继续沿 attention 边界看更深层，
      判断这种微小 drift 会在哪一层开始显著放大
  - 新增更强约束：
    - `layer1` 的 drift 不是只出现在 `seq_len=71` 当前轮 output
    - `seq_len=70` 时，prefix tail 中 `slot 196` 的历史 K/V 就已经轻微漂移
    - 该 `slot 196` 已确认就是 accepted-path 写入前缀的 token `3837`
    - 因而下一步优先级应改成：
      - 先确认该历史 token 在 `layer0` 是否仍一致
      - 若 `layer0` 一致，再验证“tree accepted row 自身的高层表示”是否已偏离 baseline
      - 若 accepted row 本身已偏离，再定位 `layer0 attention output` 之后到 `layer1 attention input` 之前的最早分叉
      - 只有这一步查不出来时，才继续盲目向更深层扩 trace
    - 上述第一步现已完成：
      - `slot 196 / token 3837` 在 `layer0 current_k/v` 上与 baseline 完全一致
      - 同一 token 到 `layer1 current_k/v` 才首次出现微小差异
    - 新增 `v5` prefix-tail 对账结果：
      - 在已对齐的 `tp_rank=0` shard 上，
        `seq_len=73` 视角下 accepted row 会消费到的 recent generated prefix
        `positions 60..67`（`slots 188..195`）
        现在已经可以直接逐项与 baseline 对齐
      - 结果是：
        - `layer0` 上 `slots 188..194` 的 prefix K/V 全部 `max_diff = 0.0`
        - `layer1` 上 `slots 188..194` 的 prefix K/V 也全部 `max_diff = 0.0`
        - `slot 195` 作为当前 batch 里的 row0，
          在 `layer0/layer1` 的 `current_q/k/v/output` 也都是 `0.0`
      - 这意味着：
        - 最近这段 generated prefix 已不再是主怀疑点
        - 当前最窄的问题面就是 `slot 196 / token 3837` 自身
          在 `layer0 current_output` 出现 `6.103515625e-05` 的微漂移，
          并在 `layer1 current_q/k/v/output` 放大
    - 因而下一步已经明确收缩成：
      - 不再继续怀疑 recent generated prefix `60..67`
      - 优先直接补 block-level hook
      - 专门查 `layer0 output -> layer1 input` 之间的哪一段先分叉
      - 备选再补更老 prompt prefix 的摘要/hash，对“更老历史是否已漂移”做一次性排除
    - 新增修正实验方向：
      - 既然 `row0` 单 row exact 是完全一致的，
        而 accepted row 在 batched multi-row exact 下才出现 `layer0 output` 级微小漂移，
        那么优先尝试把 tree exact attention 改成“逐 row 独立计算”
      - 若这样能消除 accepted row 的 `layer0 output` 漂移，
        就说明当前主问题是 batched exact path 的数值路径而不是 tree 语义本身
    - 该实验已完成并被证伪：
      - accepted row 的 `layer0 current_output` 与 `layer1 current_q/k/v` 差异完全不变
      - 因而下一步改成：
        - 撤回逐 row 功能改动
        - 保留 `current_q/current_output/tp_rank` instrumentation
        - 扩大 tree exact debug 的 prefix tail，
          直接检查更早的 generated prefix（尤其 positions 60..67）是否已经在 `layer0/1` 漂移
  - 当前新增的 instrumentation 级怀疑点：
    `_EXTRA_CTX` 代理白名单未包含 `input_ids / positions`，
    可能导致 trace 里这两个字段始终为 `null`
  - 最新结论：
    `seq_len=70/71` 时 baseline/tree 的 `current_input_id / position / slot` 完全一致，
    `seq_len=72` 才分叉成 `115995` vs `100210`
  - 因而下一轮最小验证改为：
    在 `seq_len=71` 对账 `layer0 attention output`，判断同一组 q/k/v 输入下 decode kernel 是否已分叉
  - 新增 block-level 结论：
    - `baseline seq68` 对 `tree seq73 row0(slot195)` 在 `model.layers.0/1` 的
      `layer_input -> pre_attn -> attn_output -> post_attn -> mlp_output`
      全链路仍是 `0.0`
    - 但 `baseline seq69` 对 `tree seq73 row1(slot196, token 3837)` 时：
      - `layer0 layer_input_hidden` 仍是 `0.0`
      - `layer0 pre_attn_hidden` 仍是 `0.0`
      - `layer0 attn_output` 首次出现非零差异
      - 之后 `post_attn / residual / mlp / layer1 input` 继续逐层放大
    - 结合已有 `attention_v1` trace：
      - `layer0 current_q/k/v` 已确认一致
      - `layer0 current_output_heads` 在 attention impl 输出处已有微小漂移
      - block trace 的 `attn_output` 是 `Qwen3Attention.forward()` 经 `o_proj` 后返回给 decoder block 的值
    - 因而当前最窄问题面更新为：
      - 非 `row0` speculative row 的 `run_exact_tree_attention` 路径
      - 或该路径消费到的更老 prefix K/V
      - 已不再优先怀疑 `layer0` 之后的 async batch state / residual materialization
  - 下一步：
    - 先读 `tree_attention.py` 和 exact attention helper
    - 然后补一轮针对 `slot196` 的定向验证：
      区分“exact attention 数学路径本身”与“更老 prefix K/V 已先漂移”
  - 上述定向验证现已完成到更窄一层：
    - baseline:
      `/tmp/baseline_prefix_trace_fullpref_v1.jsonl`
    - tree:
      `/tmp/tree_exact_fullpref_v1.jsonl`
    - 对 `baseline seq69` vs `tree seq73 row1(slot196)` 的 `layer0` 全 prefix head0 对账结果：
      - older prefix `slots 128..194` 的 `K/V` 全部 `0.0`
      - `slot195` 的 `K/V`（baseline prefix-last vs tree current row0）全部 `0.0`
      - `slot196` 自身 `current_k/v` 也全部 `0.0`
      - 但 `slot196 current_output_heads` 仍有 `max_diff = 6.103515625e-05`
    - 这说明：
      - 在已观测 head 上，`slot196` 的可见输入面已经完全对齐
      - remaining 分叉已经进一步收缩到 non-row0 attention 执行路径本身
      - 更老 prefix/cache 漂移不再是当前第一怀疑点
  - 下一步改成：
    - 设计一个 correctness-first 的 non-row0 tree attention fallback
    - 优先复用 Ascend 自带 decode/fused kernel 的数值路径，而不是继续依赖当前 Python dense softmax exact path
  - 上述 fallback 第一版现已接入：
    - non-row0 speculative row 改为
      `compact visible KV + npu_fused_infer_attention_score`
    - row0 仍保留 decode override
    - Python dense softmax exact path 作为异常兜底
  - 第一轮长文本主复现结果：
    - baseline token count: `108`
    - tree token count: `112`
    - `first_diff_index` 从旧版本的 `11` 后移到 `20`
    - 前 20 个 token 现已与 baseline 完全一致
  - 这说明：
    - non-row0 attention 数值路径确实是主问题之一
    - 当前 vendor rowwise fallback 已经实质性修正了早期分叉
    - 但后续更深位置仍存在剩余偏差，需要继续缩圈
  - 下一步：
    - 围绕新的 `first_diff_index=20` 重新挂 trace
    - 判断剩余偏差是否仍来自 later non-row0 attention，还是已经转移到别的链路
  - 这一步现已得到首条关键结论：
    - baseline sample trace:
      `/tmp/baseline_sample_after_vendor_v1.jsonl`
    - tree sample trace:
      `/tmp/tree_sample_after_vendor_v1.jsonl`
    - 对齐后发现：
      - `tree step16` 一次产出 `[20412, 99489]`
      - 其中 `bonus` 行与 baseline `step18` 一致
      - `row_label=[1]` 行与 baseline `step19` 也一致
      - 但紧接着的 `tree step17 bonus/row0` 就开始偏成 `31235`
      - 对应 baseline `step20 sample` 应为 `97145`
    - 这说明：
      - 当前剩余问题已经不再是“non-row0 attention 先漂，再传染 row0”
    - 更像是“一次多 token 接受之后，next root 的 first-pass / state materialization 仍有错位”
  - accepted-path KV materialization v1 的最新验证结果：
    - 已在 `propose_draft_token_ids()` 之前接入 canonical slot copy
    - `step 15` 的 trace 已明确看到四个 TP rank 都执行了：
      - `token_id=20412`
      - `source_slot=207`
      - `destination_slot=206`
    - 但同一轮复跑里：
      - `step 16 sampled_token_ids = [[20412, 99489]]`
      - `step 17 sampled_token_ids = [[31235]]`
      - 与 materialize 之前完全一致
    - 因而“accepted token 没有被 copy 回 canonical prefix slot”已经从主假设降级为：
      - copy 动作本身不是缺失的
      - 但仍需验证 copied slot 是否真的被下一轮 attention 读取，
        或是否在后续路径中被覆盖/绕过
  - 下一轮验证优先级已更新为：
    - 不再只看“有没有计划 copy”
    - 直接对账 materialize 之后的 next-step prefix/cache 观测
    - 目标是回答：
      - `slot 206` 的 prefix K/V 是否真的变成了 `slot 207` 的 accepted token 状态
      - 若已变对，是否存在更后置的 overwrite / alternate cache path / hidden-state reuse 错位
  - 上述 next-step prefix/cache 验证现已给出更强结论：
    - 新 trace：
      - `/tmp/tree_prefix_after_materialize_v1.jsonl`
      - `/tmp/tree_sample_after_materialize_v2.jsonl`
    - 关键对齐点：
      - tree `prefix-trace step 16`
      - `current_input_ids = [99489]`
      - `seq_len = 80`
      - `current_slots = [207]`
      - 同轮 sample 仍是 `sampled_token_ids = [[20412, 99489]]`
    - 观测结果：
      - `current_k/v` 已与 baseline `step19 current_k/v` 对齐
      - `prefix_tail slots 203/204/205` 也与 baseline 对齐
      - 但 `prefix slot 206` 仍不对：
        - baseline `prefix_last_k/v` 是 accepted token `20412` 的 canonical 状态
        - tree `prefix_last_k/v` 仍是另一份错误值
    - 因而问题面进一步收缩为：
      - 不是“没执行 copy”
      - 也不是“copy 后又把 current row 算错了”
      - 而是：
        - copied value 没有进入 attention gather 真正读取的物理 cache
        - 或 copied destination slot 在 step16 前被后续路径覆盖回错误值
  - 下一轮实际动作更新为：
    - 直接核对 `model_runner` 里被 copy 的 `self.kv_caches`
      与 attention backend 实际读取的 `key_cache/value_cache`
      是否共享同一底层 storage / data_ptr
    - 若是同一 storage，再追 `slot 206` 在 copy 之后、step16 attention 之前的覆盖路径
  - 代码级复核已给出新的强根因候选：
    - `attention/tree_attention.py::gather_paged_kv_to_ragged()` 证明
      paged KV cache 的 layout 是：
      - 第 0 维：`block`
      - 第 1 维：`block_size` 内 offset
    - 但当前 `_copy_tree_accepted_path_kv_slots()` 却把
      `slot_mapping = block * block_size + offset`
      直接拿去做 `cache_tensor.index_select(0, slot_ids)`
    - 这等价于把 flat slot id 错当成 block id
    - 因而当前 materialize copy 很可能一直在改错误的 cache block，
      难怪 trace 里“copy 执行了”但 `prefix slot 206` 仍完全不变
  - 下一步已明确为：
    - 直接修 `_copy_tree_accepted_path_kv_slots()`：
      - 先把 flat slot 拆成 `block_index` 与 `block_offset`
      - 再按 `[block_index, block_offset]` 做 gather/scatter
    - 然后立刻重跑 step16/17 定向 trace 验证
  - 上述修复现已被定向 trace 直接验证：
    - 新结果：
      - `/tmp/tree_prefix_after_materialize_v2.jsonl`
      - `/tmp/tree_sample_after_materialize_v3.jsonl`
      - `/tmp/tree_kv_materialize_v2.jsonl`
    - 关键点：
      - `step 15` 仍然执行 `207 -> 206`
      - `step 16` 时：
        - `current_input_ids = [99489]`
        - `prefix slot 206` 的 `k/v`
          已与 baseline `step19 prefix_last_k/v` 完全一致
      - `step 17` sample 已从错误的 `31235`
        变为 baseline 对齐的 `97145`
      - 同轮 accepted token 数也从 `3`
        提升到了 `5`
    - 当前阶段性结论：
      - materialize copy 的 flat-slot 物理索引错误就是 late-step 主根因之一，
        且该点已经修通
  - 下一步切换为：
    - 跑完整 `32B eager` compare
    - 看 `first_diff_index` 是否消失，或是否还有更晚位置的残余分歧
  - 完整 `32B eager` compare 现已给出新的全局状态：
    - 结果文件：
      - `/tmp/qwen32_tree_eager_quick_check_noasync.json`
    - baseline `token_count = 108`
    - tree `token_count = 115`
    - `first_diff_index = 74`
    - diff token:
      - baseline `104963`
      - tree `105355`
    - tree metrics 中最终 `Accepted = 6 tokens`
  - 这说明：
    - 旧的 late-step `slot 206` 问题已经不再是端到端首个分歧点
    - Phase 5 的目标要更新成：
      - 围绕新的第 `74` 个 token 重新缩圈
      - 明确这是“新的更晚残余问题”，而不是旧 root cause 没修透
  - 新的待验证判断先登记为：
    - 第 `74` 个 token 的分歧发生在
      `“从古代的四大发明到现代的...”`
      这类自由续写空间较大的语义位置
    - baseline/tree 当前都仍输出通顺中文
    - 因而优先怀疑：
      - 长程累积后的轻微 logits/state drift
      - 而不是再出现一次旧式的 canonical prefix continuity 硬错误
    - 下一轮验证顺序：
      - 先抓该点附近的 sample/top-logits margin
      - 再决定是否要回头挂更重的 prefix/cache trace
  - 新增 instrumentation 约束：
    - `VLLM_ASCEND_SAMPLE_TRACE_STEPS` 默认值只有 `32`
    - 所以如果不显式放大窗口，
      sample trace 根本覆盖不到新的第 `74` 个 token
    - 下一轮执行要求：
      - 带 `VLLM_ASCEND_SAMPLE_TRACE_STEPS=128` 重跑
  - 上述 sample trace 放大后又给出更窄定位：
    - baseline `step 74` 对应 token `104963`
    - tree `step 66` 已直接产出 `[9370, 105355]`
    - 问题不是发生在 step66 之后的 replay
    - 而是发生在 step66 内部的 `row [0]`
  - 因而当前 Phase 5 的最小验证切换为：
    - 不先追 next-root replay
    - 先对账 baseline `step 74`
      vs tree `step 66 row [0]`
      的 prefix / cache / hidden/logits 输入面
  - 更细的当前对齐假设：
    - baseline `current_input_id = 9370`
      这一轮
    - 应对应 tree block trace `index 65`
      的第二个 query row
    - 下一步先验证这个对齐是否成立，再做 block 级差异比较
  - 下一步改成：
    - 重跑 `tree_first_pass_trace` / `tree_state_trace`
    - 专门盯 `tree step16 -> step17` 这次双 token 推进后的 root 重建链

## Errors Encountered

- `pytest` 受现有 mixed `PYTHONPATH` / `conftest` 导入问题影响，当前不作为 tree runtime 根因判断依据。
- `async_scheduling=False` 对当前分叉树偏差无帮助，不能继续在调度维度重复排查。
- 仅靠 attention trace 不足以说明“attention 本身无误”；因此本轮又补了 exact K/V 对账与 row0 isolated 对账。新日志已表明 cache/gather/row0 batching 不是当前主矛盾。
- root/bonus 行 isolated exact attention 实验导致 `first_diff_index` 从 11 回退到 2，说明这不是正确修正方向，不能继续重复。
- 当前 sample trace 已证明分歧发生在 raw hidden/logits 阶段；后续若要继续怀疑 sampler/verifier，必须拿出更强的反证，不能再凭直觉回退。
- 新增 `tree_state_trace` 首轮曾因 `tree_terminal_row_indices` 尚未初始化而崩溃；已改成 `getattr(..., None)` 防御式读取，后续 trace 可稳定复用。
- 当前 `tree_state_trace` 已直接看到 `prev_sampled_token_ids` 仍是单 token 语义、`spec_token_ids` 长期全 `-1`、`token_ids_cpu` 仍充满 placeholder；在没有反证前，不能再把注意力主要放回 attention 侧。
- 接续会话后的当前主假设再次收缩为：
  - non-row0 tree attention 的 vendor fallback 已经把早期分叉从 `first_diff_index = 11` 推后到 `20`
  - `tree step16` 一次接受并输出 `[20412, 99489]` 后，
    `tree step17` 的 bonus/root 才首次偏成 `31235`
  - 对齐 baseline 后，对应应该比较的是：
    - baseline `step18 sample == 20412`
    - baseline `step19 sample == 99489`
    - baseline `step20 sample == 97145`
  - 因而当前第一优先级不再是扩 attention trace，
    而是核对一次多 token accept 之后 next-root replay / async `input_batch` materialization
- 当前最小验证清单已经固定：
  - 重跑带以下日志的长文本 32B eager case：
    - `VLLM_ASCEND_TREE_FIRST_PASS_TRACE_PATH`
    - `VLLM_ASCEND_TREE_STATE_TRACE_PATH`
    - `VLLM_ASCEND_SAMPLE_TRACE_PATH`
  - 只盯 `tree step16 -> step17`
  - 核对字段：
    - `accepted_paths`
    - `terminal_row_indices`
    - `rebuilt_token_ids / rebuilt_positions / rebuilt_seq_lens`
    - `prev_sampled_token_ids`
    - `spec_token_ids`
    - `token_ids_cpu_*tail`
    - `num_computed_tokens`
    - `optimistic_seq_lens_cpu`
  - 如果这些字段已经 self-consistent，
    再回头怀疑更深层 hidden/logits；否则先修 runtime state 链
- 上述验证的最新反证已经出现：
  - 通过定向 `tree target trace` 复现，当前代码在问题点附近的 target 输入是：
    - step `16` row0 `input_id = 99489`
    - `position = 79`
    - `slot_mapping = 207`
  - 这与 baseline 第 `20` 个生成 token之前应消费的 current token 语义一致
  - 因而“next-root 的 token / position / slot 直接喂错”这条假设需要降权
- 当前主怀疑点进一步收窄为：
  - terminal row 被提升为新 root 时，
    `input_id / position` 已对，
    但 canonical prefix KV / slot 收束可能仍有问题
  - 特别是 `terminal_row_indices = [2]`（也就是接受的不是第一个 depth-1 row）时，
    promoted row 的物理 slot 可能没有被正确压回连续前缀
  - 这能解释：
    - tree `step16` 的 bonus 与 row `[1]` 仍可分别对齐 baseline `step18/19`
    - 但紧接着以 `99489` 为 current token 的下一轮 row0 bonus 就偏成 `31235`
  - 下一轮最小验证继续细化为：
    - 不只看 `rebuilt_token_ids / rebuilt_positions / rebuilt_slot_mapping`
    - 还要直接对账：
      - baseline 对应步 row0 的 full-prefix K/V
      - tree `step17` row0 的 full-prefix K/V
    - 如果 promoted sibling row 的 canonical continuity 真有缺口，
      这一步应该会在 `step16 -> step17` 间直接看到 prefix 视图先分叉
- 上述 full-prefix 验证现已完成，结果比原假设更具体：
  - 对齐：
    - baseline `step19`
    - tree `step16`
    - `current_input_id = 99489`
    - `seq_len = 80`
    - `current_slot = 207`
  - 结论：
    - `current_input_id / position / slot`
    - `current_q / current_k / current_v`
    都仍一致
    - 但 `prefix slot 206 / position 78` 的 K/V 已明显分叉
  - 这说明：
    - continuity 缺口不在 current root 本身
    - 而在多 token accept 里的第一个 accepted draft token
      没有被落到 canonical prefix slot
  - 当前动作随之切换为：
    - 实现 accepted-path KV materialization
    - 具体实现约束固定为：
      - `rejection_sampler` 挂完整 `accepted_paths` 到 runner，而不只保留 `terminal_row_indices`
      - `model_runner_v1.propose_draft_token_ids()` 在 `drafter._propose(...)` 前执行
        tree-only KV materialization
      - source row 由 accepted runtime node 映射回当前 target execution row：
        `request_row_offset + runtime_node_to_execution_index[runtime_node_index] + 1`
      - destination slot 由 accepted token 自身的语义 `position`
        配合 block table 计算 canonical slot，而不是沿用 speculative physical slot
      - KV 拷贝采用 gather-then-scatter，避免 `src/dst` 重叠时自覆盖
      - bonus token 不 materialize 成 prefix，仍保留给 next-root current token
    - 再复跑同一轮 `current_input_id = 99489`
      检查 `prefix slot 206` 是否恢复一致
  - accepted-path KV materialization `v1` 已接入后，
    长文本 32B eager 端到端结果仍是：
    - `first_diff_index = 20`
    - baseline/tree 在 diff 点仍是 `97145` vs `31235`
  - 这说明：
    - 问题不再停留在“猜到 root cause 但还没下手”
    - 而是要继续确认：
      - helper 是否真的执行了预期 copy
      - copy 的 `source_slot / destination_slot` 是否正确
      - copy 后是否又被后续链路覆盖
  - 下一步固定为：
    - 给 materialization helper 增加独立 trace
    - 重跑问题步附近的 tree trace
    - 先验证 `207 -> 206` 这类 copy 是否真实发生
- baseline prefix-cache trace 第二轮曾因 instrumentation 接线错误失败：
  - `forward_impl()` 内转发 `forward_fused_infer_attention(...)` 时引用了未定义的 `layer`
  - 已补齐 `forward_impl(layer, ...)` 签名和调用链
  - 这属于 trace 接线错误，不改变当前 root-cause 判断
- 新增 exact continuity 对账后已确认：
  - `step 7` accepted draft token `3837` 写入 `slot 196`
  - 下一轮 prefix tail 对同一 `slot 196` 的 K/V 读取在 64 层上 `max diff = 0.0`
  - bonus continuity 也正常
  因而不能再把“accepted token cache continuity 错误”当作主假设。
- 新增线性层 isolated probe 首次运行失败：
  - 使用了系统 `/usr/bin/python3`
  - 环境中缺少 `torch`
  - 后续统一改用 `/root/venvs/vllm-ascend/bin/python`
- 正确 venv 上还存在一层启动约束：
  - 必须带 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`
  - 否则 `torch` 导入时会提前 auto-load `torch_npu` 并在脚本 bootstrap 之前因 `libhccl.so` 缺失失败
- linear isolated probe 的第一版实现有边界缺陷：
  - sample 阶段的某些 drafter 线性层不在 `forward_context` 保护区间内
  - 直接读 `_EXTRA_CTX` 会触发 `Forward context is not set`
  - 已修正为“仅在 target forward 活跃区间内采样”
- linear isolated probe 的第二版条件仍偏严：
  - tree-only 实跑成功，但没有任何 trace 输出
  - 当前判断不是“差异为零”，而是 top-level `tree_spec_decode_enabled` 检测不可靠
  - 已改为用 `num_tokens<=16` 作为小 batch target forward 的主筛选条件
- v5 trace 首轮人工复盘时曾误把观测字段从顶层读取：
  - 实际上 `current_input_ids / current_q_heads / current_output_heads` 等都位于
    `requests[0]` 下
  - 这只是日志解析失误，不是模型行为变化
- 新增运行时问题：
  - 为了把 baseline/tree 的 prefix-trace 脚本做成可独立复用的 bootstrap 版本，
    这轮在脚本内补了 CANN/ATB 环境设置
  - 随后首次重跑 `layer1` deeper-layer trace 时，baseline/tree 都在 worker 初始化阶段报：
    `ParallelOpenMP.cpp:64 Invalid thread pool`
  - 该错误发生在 engine startup，还没进入任何本轮模型对账逻辑
  - 当前最可能方向：
    - 脚本 bootstrap 与 multiprocess worker 启动方式的组合有冲突
    - 特别是缺少先前稳定脚本里的 `VLLM_WORKER_MULTIPROC_METHOD=spawn`
    - 或其他与 worker 启动相关的环境项未对齐
  - 上述启动环境问题已修正：
    - 补齐了 `VLLM_WORKER_MULTIPROC_METHOD=spawn`
    - 补齐了 ATB/LCCL 相关环境项
  - 但紧接着又暴露出第二个更直接的问题：
    - baseline/tree 两个 4 卡任务并行跑会触发 HCCL 端口冲突
    - 典型报错：
      `Failed to enable listening for the NPU network adapter socket`
      `The IP address 192.46.2.199 and port 16666 have already been bound`
  - 因而后续 deeper-layer trace 不能并行跑 baseline/tree，
    必须改成串行，或显式拆开 `HCCL_NPU_SOCKET_PORT_RANGE`

## Resume 2026-06-29 Late

- 新待验证问题：
  - `baseline index = 73`
    vs
    `tree index = 65 row [0]`
    这对已确认同 `token / position / slot = 9370 / 133 / 261`
    的 `layer1` 对齐样本里，
    首个 residual divergence
    到底是：
    - `current_q/k/v` 已先漂移
    - 还是 `current_q/k/v` 仍一致、但 `current_output` 才开始漂移
- 当前执行纪律：
  - 先解析现有日志：
    - `/tmp/baseline_prefix_trace_l1_v2.jsonl`
    - `/tmp/tree_exact_layer1_late_v1.jsonl`
  - 不先重跑 4 卡任务
  - 只有现有日志不足以回答上述分流时，才继续补 trace
- 判定规则：
  - 若 `current_q/k/v` 已漂移：
    - 下一步前移到 `layer1` 之前的 block 链路
  - 若 `current_q/k/v` 仍一致、但 `current_output` 已漂移：
    - 下一步优先下钻 `layer1 attention impl`
      或更老 prefix K/V
- 上述分流现已被现有日志直接回答：
  - baseline:
    - `/tmp/baseline_prefix_trace_l1_v2.jsonl`
  - tree:
    - `/tmp/tree_exact_layer1_late_v1.jsonl`
  - 对齐样本：
    - baseline `step = 73`
      `current_input_ids = [9370]`
      `current_input_positions = [133]`
      `current_slots = [261]`
    - tree `index = 65 row[0]`
      的 `query_pos = 1`
      对应同一 `slot = 261`
  - 结果：
    - `current_q.max_diff = 0.0`
    - `current_k.max_diff = 0.0`
    - `current_v.max_diff = 0.0`
    - `current_output.max_diff = 0.00013399124145507812`
  - 但同一行可见 prefix 中，
    `layer1` 的最近几格历史 K/V 已不再全等：
    - `slot 253..255` 仍为 `0.0`
    - `slot 256..260` 开始出现轻微 drift
      - `k.max_diff` 约 `0.03125 .. 0.125`
      - `v.max_diff` 约 `0.00225830078125 .. 0.00341796875`
  - 因而当前结论更新为：
    - 这条 residual issue 不是
      “layer1 attention impl 在完全同输入面下凭空算出不同 output”
    - 而是：
      - 当前 token `9370` 的 `q/k/v` 仍一致
      - 但它消费到的最近几格 `layer1 prefix K/V`
        已经在更早历史中轻微漂移
      - `current_output` 的 tiny diff 是这段已漂历史输入的直接后果
- 下一步切换为：
  - 不再优先下钻 `layer1 attention impl`
  - 先把 `slot 256..260` 映射回具体历史 token / step
  - 再顺着这些历史 token 去找：
    - 是哪一个 token 首次把 `layer1 K/V` 写偏
    - 它在 `layer0 -> layer1` 之间的哪一段先分叉
- `slot 256..260` 的 baseline 历史映射现已拿到：
  - `slot 256 -> step 68 -> token 9370 -> position 128`
  - `slot 257 -> step 69 -> token 104978 -> position 129`
  - `slot 258 -> step 70 -> token 102832 -> position 130`
  - `slot 259 -> step 71 -> token 26939 -> position 131`
  - `slot 260 -> step 72 -> token 100390 -> position 132`
- 下一步最小验证更新为：
  - 利用现有 `tree_exact_layer1_late_v1` 中
    `index 61..65`
    对应的 current rows
  - 逐个对齐上述 5 个 slot
  - 找出哪个 token 在作为 current row 时，
    首次出现 `layer1 current_k/v/output` 非零差异
- 上述逐 token 对齐现已给出新的更强候选：
  - `slot 256 -> token 9370`
    在其作为 current row 的那一轮，
    就已经出现：
    - `q_max = 0.04296875`
    - `k_max = 0.125`
    - `v_max = 0.00225830078125`
    - `out_max = 0.000732421875`
  - `slot 257..260` 也都继续非零
- 当前新的待验证问题改成：
  - `slot 256` 是否就是目前已观测到的最早漂移 token
  - 还是更早的 `slot 253..255`
    在它们自己作为 current row 的那一轮也已经有同类 `layer1` 差异，
    只是此前作为 prefix 被误判为 `0.0`
- 上述前验现已完成：
  - `slot 253 -> token 1773`
  - `slot 254 -> token 45181`
  - `slot 255 -> token 102640`
  在它们自己作为 current row 的那一轮，
  `layer1 current_q/k/v/output` 全部仍为 `0.0`
- 因而当前最早的已证实漂移点正式收敛为：
  - `slot 256`
  - `token 9370`
  - `position 128`
- 下一步最小动作更新为：
  - 围绕 `slot 256 / token 9370`
    补或复用 block-level trace
  - 直接比较它在 `layer0 -> layer1` 边界上的：
    - `layer_input_hidden`
    - `pre_attn_hidden`
    - `attn_output`
    - `post_attn_hidden`
    - `mlp_output`
  - 目标是回答：
    - `layer1 current_q` 为何从这个 token 开始首次非零
- 恢复会话后的执行约束已固定：
  - 先只解析现有 block trace
    - `/tmp/baseline_block_trace_late_v2.jsonl`
    - `/tmp/tree_block_trace_v1.jsonl`
    - `/tmp/baseline_block_trace_l1_v1.jsonl`
    - `/tmp/tree_block_trace_l1_v1.jsonl`
  - 不先补新的 4 卡重跑
  - 只有现有 trace 不能回答
    `token 9370` 在自己那一轮的
    `layer0` 首个非零字段时，
    才允许继续加最小化定向 trace
 - 上述 block-trace 对账现已直接给出答案：
   - baseline `step 68`
     (`token 9370`, `position 128`, `slot 256`)
   - 对齐 tree `index 61 qpos 0`
   - 在 `layer0` 上：
     - `layer_input_hidden = 0.0`
     - `layer_input_residual = 0.0`
     - `pre_attn_hidden = 0.0`
     - `attn_output` 首次非零：`max_diff = 0.0009765625`
     - `post_attn_hidden = 0.000732421875`
     - `post_attn_residual = 0.0009765625`
     - `mlp_output = 0.00390625`
   - 对应 `layer1` 上：
     - `layer_input_hidden = 0.00390625`
     - `layer_input_residual = 0.0009765625`
     - `pre_attn_hidden = 0.000732421875`
   - 这说明：
     - `slot 256` 的首个已证实非零字段
       已从“`layer1 current_q`”继续前移到
       “它自己那一轮的 `layer0 attn_output`”
 - 因而当前下一步已再次收缩为：
   - 不再先查 `layer1`
   - 优先回答：
     - 在 `slot 256` 这一轮，
       `layer0 current_q/k/v` 已知一致的前提下，
       `attn_output` 为何仍首次非零
   - 当前两个最窄待验证方向：
     - `layer0` 读到的更老 prefix K/V 已先轻微漂移
     - 或 `layer0` attention exact path 在同输入面下仍有 residual 数值路径差异
 - 现有 `/tmp` 日志的覆盖边界也已确认：
   - `layer0` 的 block trace 足够定位到
     `attn_output`
   - 但没有现成的 `layer0 prefix/exact` 晚步数日志
     覆盖 `slot 256`
 - 因而下一步最小重跑计划固定为：
   - 继续用现有 instrumentation
   - 不改功能逻辑
   - 串行重跑一轮 `32B eager noasync` 长文本 case
   - 只抓：
     - baseline `layer0 prefix trace`
     - tree `layer0 exact debug`
   - 并把 `tail/head` 拉大到足够覆盖
     `prefix_len = 128`
 - 上述定向重跑现已完成，并把问题进一步压缩：
   - baseline:
     `/tmp/layer0_prefix_compare_v1.jsonl`
     `step = 68`
   - tree exact:
     `/tmp/tree_exact_layer0_256_v1.jsonl`
     `index = 61 row0`
   - 在 `slot 256 / token 9370 / prefix_len 128` 上：
     - `prefix_tail_slots 128..255` 全部对齐
     - `prefix_tail_k/v` 全部 `0.0`
     - `current_q = 0.0`
     - `current_k = 0.0`
     - `current_v = 0.0`
     - `current_output = 6.103515625e-05`
   - 同时：
     - `current_k_max_diff(tree direct vs gathered) = 0.0`
     - `current_v_max_diff(tree direct vs gathered) = 0.0`
     - `row0_output_max_diff_vs_isolated = 0.0001220703125`
 - 当前结论因此再次前移：
   - `slot 256` 的 remaining drift
     不是更老 `layer0 prefix K/V` 先漂
   - 而是：
     - 在可见输入完全一致的前提下，
       `layer0 exact attention` 自身已经出现 residual 数值差异
 - 下一步最窄验证更新为：
   - 比较：
     - baseline output
     - tree `tree_row0_decode`
     - tree exact row0
   - 目标是回答：
     - 漂移来自 multi-row exact attention 本身
     - 还是 isolated exact / vendor rowwise 路径也已偏离 baseline
 - 上述三向比较现已完成，结果比预期更窄：
   - baseline `step 68`
   - tree `tree_row0_decode step 61`
   - tree exact `index 61 row0`
   - 满足：
     - `baseline vs tree_row0_decode current_output = 6.103515625e-05`
     - `baseline vs tree exact row0 current_output = 6.103515625e-05`
     - `tree_row0_decode vs tree exact row0 current_output = 0.0`
   - 同时 `baseline vs tree_row0_decode` 的：
     - `prefix_tail_k/v = 0.0`
     - `current_q/k/v = 0.0`
 - 因而当前最窄问题面再次改写为：
   - 不是 multi-row exact attention 特有问题
   - 而是 tree 侧单行 root decode 路径
     在同输入面下就已经与 baseline 有稳定微差
 - 下一步优先级更新为：
   - 先核对 `decode_fused` vs `tree_row0_decode`
     的 metadata / call-path 差异
   - 包括但不限于：
     - `_build_tree_root_decode_metadata()`
     - `actual_seq_lengths_q/kv`
     - `attn_state`
     - `block_tables / slot_mapping` 子视图
 - 继续读 `attention_v1.py` 后，当前最强嫌疑点已进一步具体化：
   - baseline 这一步的 trace source 是 `decode_fused`
   - tree row0 当前却在
     `forward_tree_exact_attention()`
     中被硬编码走
     `forward_paged_attention(...)`
   - 即使输入面完全一致，
     两边仍然会因为 kernel path 不同而产生稳定微差
 - 当前下一步实现计划固定为：
   - 把 tree row0 override 改成复用 baseline decode 的同一 dispatch 逻辑
   - 若该条件下 baseline 会走 `decode_fused`
     tree row0 也必须走 `forward_fused_infer_attention`
   - 只有命中 `using_paged_attention(...)`
     时才继续走 paged path
 - 上述功能改动现已落地到：
   - `vllm_ascend/attention/attention_v1.py`
   - 具体包括：
     - `forward_fused_infer_attention()` 新增 `trace_source`
     - tree row0 override 现在按 baseline 同一 dispatch 逻辑
       在 `tree_row0_fused` / `tree_row0_paged` 间分流
 - 下一步只做复跑验证，
   不再继续扩改动面。
 - 复跑后的首个失败点已经暴露：
   - tree row0 确实进入了 `forward_fused_infer_attention()`
   - 但 `row0_metadata.attn_mask` 为空
   - Ascend fused kernel 直接报错：
     - `When attnMask is not provided, sparseMode must be 0`
 - 因而下一步修正已更新为：
   - 不回退 dispatch 方向
   - 继续补齐 tree row0 fused 所需 metadata
   - 第一优先级就是给 `row0_metadata` 带上正确的 `attn_mask`
 - 该修正现已落地：
   - `row0_metadata.attn_mask`
     现已复用 `AttentionMaskBuilder`
   - 同时补上了 `row0_metadata.seq_lens_cpu`
 - 下一步继续复跑最小化验证，
   先看 fused path 是否能正常跑通。
