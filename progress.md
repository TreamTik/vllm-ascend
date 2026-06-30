# Progress

## 2026-06-29

## 2026-06-30

- 已新增正式 `2048 / 2048` benchmark 工具链：
  - `/root/qwen32_2048_bench_common.py`
  - `/root/run_qwen32_2048_offline_bench.py`
  - `/root/run_qwen32_2048_service_bench.py`
- 已扩展服务启动脚本：
  - `/root/start_qwen32_w8a8_service.sh`
  - 当前新增：
    - 可通过 `SPECULATIVE_CONFIG_JSON` 显式打开 speculative decode
    - 默认行为仍保持 baseline，不影响原有启动方式
- 已生成统一输入数据集：
  - `/root/qwen32_2048_bench_dataset.json`
  - 当前两条样本：
    - `json_grid`
    - `numbered_catalog`
  - prompt token 长度：
    - `2048`
    - `2048`
  - 平均输入：
    - `2048.0`
- 中间问题：
  - 首版数据集拟合采用大范围暴力 token 计数，准备阶段 CPU 开销过高
  - 后已改成：
    - 行数二分
    - filler 次数二分
  - 同时修正了 `apply_chat_template(tokenize=True)` 在当前 tokenizer 上的返回结构误判
  - 现在改为：
    - 先渲染字符串
    - 再 `tokenizer.encode()` 计数
- 下一步进入真实 benchmark：
  - 先跑离线 `baseline/tree_before/tree_after`
  - 再跑服务化辅助验证

- 已新增多 prompt 复验脚本：
  - `/root/run_qwen32_tree_prompt_bench.py`
  - 作用：
    - 一次加载 baseline / tree
    - 批量跑复杂 prompt
    - 对每个 case 记录：
      - `generate_seconds`
      - `token_count`
      - `same_text / same_token_ids`
      - tree 的 `num_drafts`
      - tree 的 `acceptance_per_pos`

- 默认 5 个复杂 prompt 的基准已完成：
  - 合并结果：
    - `/tmp/qwen32_tree_prompt_bench_default_merged.json`
  - 所有 case 都满足：
    - `same_text = true`
    - `same_token_ids = true`
  - 关键时间：
    - `natural_longform`
      - baseline `17.03s`
      - tree `18.32s`
    - `structured_report`
      - baseline `28.10s`
      - tree `30.68s`
    - `timeline_list`
      - baseline `26.56s`
      - tree `26.82s`
    - `markdown_table`
      - baseline `37.68s`
      - tree `39.39s`
    - `json_dataset`
      - baseline `43.60s`
      - tree `41.88s`
  - 当前阶段性判断：
    - 复杂 prompt 里已经出现
      baseline 被 tree 反超的真实 case
    - 当前最显著的正向样本是
      `json_dataset`
    - 格式越规则，tree 越容易把 acceptance 转成真实收益

- 为了把这种趋势再放大，额外补了一个超长强结构 JSON case：
  - 结果文件：
    - `/tmp/qwen32_tree_prompt_bench_json_long.json`
  - case:
    - `json_long_grid`
    - `760` token
    - 固定 `16` 个对象、每个对象固定 `8` 个字段
  - 结果：
    - baseline `63.73s`
    - tree `50.37s`
    - `same_text = true`
    - `same_token_ids = true`
    - `speedup = 1.265x`
    - `tree_num_drafts = 399`
    - `acceptance_per_pos ~= [0.526, 0.378]`
  - 直接结论：
    - 当前实现已经在复杂 prompt 上拿到明确加速
    - 而且是完全同文输出下的加速，不是以正确性换速度
    - 所以后续如果继续追整体吞吐，
      优先级更应放到：
      - 提升 acceptance 的 proposer/prompt 形态
      - 以及更贴近服务的非 eager 运行形态

- `batched_all_rows` 这一轮真实 32B 长跑已完成并成功：
  - 命令等价于：
    - `VLLM_ASCEND_TREE_FAST_IMPL=batched_all_rows /root/venvs/vllm-ascend/bin/python /root/tmp_tree_eager_quick_check_noasync.py`
  - 结果文件：
    - `/tmp/qwen32_tree_batched_all_rows_v1.json`
    - `/tmp/tree_fast_profile_all_rows_v1.jsonl`
  - baseline:
    - `generate_seconds = 10.10`
  - tree:
    - `generate_seconds = 10.65`
    - `token_count = 108`
  - comparison:
    - `same_token_ids = true`
    - `same_text = true`
  - profile 关键结论：
    - `avg_total_ms = 0.2067`
    - `avg_row0_ms = 0.0`
    - `avg_non_root_ms = 0.1329`
  - 直接结论：
    - 统一 all-row exact path 已经在真实 32B case 上同时满足
      correctness 和大幅提速
    - row0 单独 override 的 attention 主瓶颈已被拿掉

- 紧接着已把实现语义正式收敛：
  - `batched`
    - 新的 unified all-row exact fast path
  - `batched_legacy`
    - 旧的 row0 override 路径
  - `batched_all_rows`
    - 继续兼容，但内部规范化为 `batched`
  - `batched_split`
    - 继续只保留为实验路径
  - `reference`
    - debug/exact fallback

- 默认主线路径的 32B 复验也已经跑完：
  - 命令特征：
    - `env -u VLLM_ASCEND_TREE_FAST_IMPL ... /root/tmp_tree_eager_quick_check_noasync.py`
  - 结果文件：
    - `/tmp/qwen32_tree_default_batched_v1.json`
    - `/tmp/tree_fast_profile_default_batched_v1.jsonl`
  - baseline:
    - `generate_seconds = 10.25`
  - tree:
    - `generate_seconds = 10.62`
    - `token_count = 108`
  - comparison:
    - `same_token_ids = true`
    - `same_text = true`
  - profile:
    - `last_tree_impl = "batched"`
    - `all_rows_unified = true`
    - `avg_row0_ms = 0.0`
  - 当前阶段性结论：
    - 默认 fast path 已经稳定切到 unified all-row 主线
    - eager/noasync 32B case 下 tree 只比 baseline 慢约 `3.7%`
    - attention 侧已不再是最值得继续深挖的第一瓶颈
    - 后续若继续追性能，更可能转向：
      - proposer / acceptance
      - 非 eager / async / graph 生产形态

- 2026-06-30 新一轮 `32B eager + noasync` 长文本复验已完成：
  - 命令等价于：
    - `VLLM_ASCEND_TREE_FAST_IMPL=batched /root/venvs/vllm-ascend/bin/python /root/tmp_tree_eager_quick_check_noasync.py`
  - 结果文件：
    - `/tmp/qwen32_tree_eager_quick_check_noasync.json`
  - baseline:
    - `generate_seconds = 10.16`
    - `token_count = 108`
  - tree:
    - `generate_seconds = 26.16`
    - `token_count = 108`
  - compare:
    - `same_token_ids = true`
    - `same_text = true`
  - 直接结论：
    - 当前长文本 case 的端到端正确性继续成立
    - tree 相比旧的 `28.71s` 已再下降一截
    - 但仍约为 baseline 的 `2.57x`

- 同轮还补了 `tree attention trace`：
  - 文件：
    - `/tmp/tree_attn_trace_long.jsonl`
  - 当前最重要的新发现：
    - `impl_name = AscendAttentionBackendImpl`
    - 没有出现 `AscendC8AttentionBackendImpl`
  - 这直接排除了刚刚那条新怀疑：
    - 当前 32B W8A8 case 不是被 C8 attention 子类拖回去的
  - 性能优化主线因此重新收敛到：
    - 通用 tree attention 路径本身

- trace 里还暴露了一个记录口径问题：
  - `path = "exact"` 只是 coarse gate 名称
  - 它还不能证明当前实际走的是：
    - `forward_tree_exact_attention`
    - 还是 `forward_tree_fast_attention`
  - 这件事后续要么靠补 dispatch trace，
    要么靠 `reference` / `batched` 双跑做性能对照来坐实

- 基于以上结论，下一轮实现方向已经重新固定：
  - 不是去追 C8
  - 也不是继续做元数据级小优化
  - 而是直接冲下一层大结构优化：
    - 尽量把 non-root rows 的 shared prefix 从“每行复制一份”
      改成“paged prefix + 小型 tree-local current attention”的分解式执行
    - 目标是继续削掉：
      - full prefix gather
      - per-row prefix duplication
      - rowwise visible KV materialization

- 任务目标已从“正确性收口”切换到“性能优化规划”。
- 已先把当前可用的两档性能现状写回文档：
  - 正确性对齐后的重型 trace 实测：
    - baseline `13.77s`
    - tree `162.71s`
  - 较轻量的历史实测：
    - baseline `10.06s`
    - tree `28.71s`

- `2048 / 2048` 离线 benchmark 首轮重跑结果：
  - `baseline` 成功：
    - `load_seconds = 81.85`
    - `elapsed_seconds = 182.97`
    - `output_token_throughput = 22.39 tok/s`
    - `qps = 0.01093`
  - `tree_before/tree_after` 都在 mixed batch 上越界崩溃：
    - `tree_before`
      - `IndexError: index 5 is out of bounds for dimension 0 with size 5`
    - `tree_after`
      - `IndexError: index 6 is out of bounds for dimension 0 with size 6`
  - 已落地针对性修正：
    - `AscendMetadata` 新增 `tree_fast_mixed_batch_fallback`
    - `model_runner_v1.py`
      - mixed batch 时不再构建 shared tree fast plan
    - `attention_v1.py`
      - mixed batch 直接回退到普通 chunked-prefill attention
      - 但保留 tree mask patch，先保证正确性和 benchmark 可跑
    - `/root/run_qwen32_2048_offline_bench.py`
      - 新增 `QWEN32_2048_MODES`
      - 现在可以只重跑 `tree_before,tree_after`

- 已完成 `2048 / 2048` 离线正式对比：
  - baseline：
    - `output_token_throughput = 22.386 tok/s`
  - tree_before：
    - `output_token_throughput = 15.615 tok/s`
  - tree_after：
    - `output_token_throughput = 22.502 tok/s`
  - 关键结论：
    - `tree_after / tree_before = 1.441x`
    - `tree_after / baseline = 1.005x`

- 已完成服务化辅证 benchmark：
  - baseline：
    - `output_token_throughput = 22.268 tok/s`
  - tree_after：
    - `output_token_throughput = 21.444 tok/s`
  - 关键结论：
    - 服务化 `tree_after / baseline = 0.963x`
    - 服务化输出 `same_text = false`
    - 当前 async / service path 仍需单独做精度与性能收口

- 已新增主报告：
  - `/root/qwen32_2048_benchmark_report_2026-06-30.md`
- 当前接下来的工作顺序固定为：
  - 先定位结构性热点
  - 再给出优化分层与优先级
  - 获得确认后再逐项落地并复验正确性
- 首轮代码热点梳理已经拿到三个强候选：
  - 每层 dense gather paged KV
  - Python request/row 双层循环的 exact attention
  - row0 再单独重算一遍 decode attention
- 下一步继续补 runner/spec_decode 侧的高频 patch/copy 开销，
  看有没有和 attention 主热点并列的第二梯队问题
- 第二梯队热点也已确认：
  - per-step tree mask build + `mask.clone()`
  - per-step depth offset tensor 构造
  - first-pass 元数据重建中的 `index_select + cpu().clone()`
  - accepted-path slot copy 中重复创建 block/offset tensor
- 另外，当前 acceptance 也偏低：
  - `draft_tokens = 455`
  - `accepted_tokens = 16`
  - 这意味着后续性能优化不能只盯算子，还要把 acceptance 作为单独维度观察
- 用户已选定后续优化目标：
  - 优先做“生产形态”性能
  - 不把 `eager + async_scheduling=False` 验证链路作为最终目标
  - 因而后续规划会优先区分：
    - 可直接传导到真实吞吐的优化
    - 只对 debug/validation 路径有帮助的优化
- 用户进一步强调：
  - 重点不是泛泛地做小优化
  - 而是优先清理“为了验证正确性而妥协出来的慢路径”
  - 这会成为下一版设计的主轴
- 已完成正式性能设计文档：
  - [2026-06-30-speculative-token-tree-performance-design.md](/root/worktrees/vllm-ascend-clean/docs/superpowers/specs/2026-06-30-speculative-token-tree-performance-design.md)
  - 设计主线已固定为：
    - 分离 `reference path` 与 `production fast path`
    - 先清理 correctness-first 的结构性慢路径
    - 再做 metadata 小热点与 acceptance 评估
  - 文档已做占位项/一致性自检，通过
- 用户已再次明确要求：
  - 不能把这轮工作收缩成“低风险小优化”
  - 必须优先推进 tree attention 的大结构优化 / fast path
  - 该做的新实现、批量化执行、乃至专用 kernel 路线，都应作为主线
- 因而当前实现策略已切换为：
  - 先立新的 `tree fast path`
  - 再让缓存 / 临时 tensor 缩减作为配套清理
  - 不再把小热点优化当作主交付
- 当前已完成的第一步大改：
  - `attention_v1.py` 新增 `forward_tree_fast_attention()`
  - `tree_attention.py` 新增 batched speculative-row FIA 路径
  - 现在 tree path 默认将：
    - row0 继续走已验证正确的 baseline decode dispatch
    - non-root speculative rows 改成单次 batched fused attention
    - reference rowwise exact path 仅保留为 `VLLM_ASCEND_TREE_FAST_IMPL=reference`
      时的兜底
  - 下一步固定为：
    - 先核实 Ascend `TND` 多 request `actual_seq_lengths(_kv)` 语义
    - 再直接跑 32B compare，看这条大改路径的正确性和性能
- 第一轮 32B `batched fast path` 实跑已完成，结论如下：
  - baseline：
    - `load_seconds = 82.78`
    - `generate_seconds = 1.11`
    - `token_count = 5`
    - 输出为 `7/18`
  - tree：
    - 模型和 draft 都能成功加载
    - 但首次真正进入 `forward_tree_fast_attention()`
      时 worker 直接报错退出
  - 根因已收敛到新 fast path 的 FIA 参数组织：
    - 文件：
      - [tree_attention.py](/root/worktrees/vllm-ascend-clean/vllm_ascend/attention/tree_attention.py)
      - [attention_v1.py](/root/worktrees/vllm-ascend-clean/vllm_ascend/attention/attention_v1.py)
    - 错误：
      - `When layout is TND and PA not enabled, keyT/valueT must be equal to the last element of actualSeqenceLengthKV`
  - 这说明：
    - batched fast path 已真实接到运行链上
    - 失败点不是调度没命中
    - 而是 Ascend `TND` batched FIA 的 seqlen 契约仍没对齐
  - 当前下一步已经非常具体：
    - 对照仓内其它 `TND` / `actual_seq_lengths_kv` 用法
    - 把当前 `visible_lens` 改成 kernel 期望的 cumulative 形式
    - 再次直接跑 32B tree case 验证

- 恢复上下文后，先复核了 `attention_v1.py` 中 row0 root override 的现状：
  - 当前 tree exact path 已经按 baseline dispatch 分流：
    - paged 时走 `tree_row0_paged`
    - 否则走 `tree_row0_fused`
  - 但 `_build_tree_root_decode_metadata()` 里新增的
    `self.attn_mask_builder.get_attention_mask(...)`
    运行时不可用
- 新增代码归属结论已写回文档：
  - `attn_mask_builder / model_config` 属于 metadata builder，
    不是 backend impl
  - 当前优先修法是：
    在 backend impl 内显式取
    `AttentionMaskBuilder(device)` +
    `self.vllm_config.model_config`
    重建标准 causal mask
  - 下一步将按这个 ownership 修 row0 fused metadata，
    然后再做最小串行 trace 复验
- `attention_v1.py` 已按上述修法落地：
  - `_build_tree_root_decode_metadata()` 不再访问
    `self.attn_mask_builder / self.model_config`
  - 改为使用：
    `AttentionMaskBuilder(device).get_attention_mask(..., self.vllm_config.model_config)`
  - 同时补齐了：
    - `query_start_loc`
    - `max_query_len`
    - `num_decode_tokens`
    - `num_decodes`
    - `num_prefills`
  - `python3 -m py_compile` 已通过
  - 随后已完成一轮串行 `32B eager + noasync` 长文本复验
- 本轮复验结果：
  - baseline 生成：
    - `load_seconds = 81.96`
    - `generate_seconds = 10.92`
    - `token_count = 108`
  - tree 生成：
    - `load_seconds = 83.66`
    - `generate_seconds = 155.31`
    - `token_count = 108`
  - `/tmp/qwen32_tree_eager_quick_check_noasync.json`
    中 baseline/tree `token_ids` 已全量一致
  - `/tmp/layer0_prefix_compare_v4.jsonl`
    已确认 trace source 为：
    - baseline `decode_fused`
    - tree `tree_row0_fused`
  - 当前剩余动作只剩：
    把 trace 步数从 `32` 放大到 `128`，
    再核销旧的 `slot 256` 微漂移结论
- `128-step` 晚步数核销现已完成：
  - `/tmp/layer0_prefix_compare_v5.jsonl`
    中 trace source 统计为：
    - `decode_fused = 107`
    - `tree_row0_fused = 91`
  - 旧定位点重新对齐结果：
    - baseline `step 68`
    - tree `step 61`
    - token / position / slot / seq_len 全一致：
      `9370 / 128 / 256 / 129`
    - `prefix_last_k/v`
    - `current_q/k/v`
    - `current_output`
      全部 `max_diff = 0.0`
  - 这说明旧的 `6.103515625e-05` root-row 漂移已被彻底清掉

- 已完成 materialization 修复后的完整 `32B eager` compare：
  - 结果文件：
    - `/tmp/qwen32_tree_eager_quick_check_noasync.json`
  - baseline:
    - `token_count=108`
    - `generate_seconds=10.06`
  - tree:
    - `token_count=115`
    - `generate_seconds=28.71`
    - metrics 最终打印到：
      - `Accepted = 6 tokens`
      - `Drafted = 180 tokens`
  - compare:
    - `first_diff_index=74`
    - `baseline_token_id_at_diff=104963`
    - `tree_token_id_at_diff=105355`
- 这轮完整复跑的直接结论：
  - 旧的 `first_diff_index=20` 已被清掉
  - `slot 206` continuity 修复已经真实传导到端到端输出
  - 但还有一个更晚的新 residual divergence，需要围绕第 `74` 个 token 重新挂 trace
- 为了避免下一轮又凭印象对齐 trace，
  先把新的待验证判断写回磁盘：
  - baseline block trace 中 `current_input_ids = [9370]`
    的那一轮
  - 暂时视作 tree block trace `index = 65`
    第二个 query row 的候选对应项
  - 下一步只做两件事：
    - 先验证 `position / slot` 是否真的对齐
    - 再看 `layer0` 是否已经在这一行开始分叉
- 这条对齐验证现已完成，且结果和先前预期不同：
  - baseline `index = 73`
    (`9370`, `position 133`, `slot 261`)
  - 确实对应 tree `index = 65` 的第二个 query row
  - 但这对候选行在 `layer0` 上仍是全链路 `0.0`
    - `layer_input_hidden`
    - `pre_attn_hidden`
    - `attn_output`
    - `post_attn_hidden`
    - `mlp_output`
- 这说明：
  - 先前的 mapping 判断是对的
  - 但真正首个 residual divergence
    不是发生在 `9370` 这一行
  - 后续立即验证了
    `baseline index = 74` vs `tree index = 66 bonus row`
  - 结果是：
    - `position/slot` 虽然相同
    - 但 token 已经不同：`104963` vs `105355`
    - 且 `layer_input_hidden` 就明显分叉
  - 这组比较现已被定性为“无效根因定位样本”：
    它只能证明 sample 分叉已经传导到下一轮 root，
    不能说明首个根因发生在 `layer0`
- 当前下一步已改成：
  - 不再比较分叉后的下一轮
  - 而是围绕真正对齐的前一轮
    `baseline index = 73` vs `tree index = 65 row [0]`
    去补 `layer1+` block trace
  - 目标是找到这对同输入调用的首个深层非零差异
- 串行补完 `layer1` block trace 后，首条有效 deeper-layer 结果已经拿到：
  - baseline trace:
    `/tmp/baseline_block_trace_l1_v1.jsonl`
  - tree trace:
    `/tmp/tree_block_trace_l1_v1.jsonl`
  - 对齐点仍然是：
    - baseline `index = 73`
    - tree `index = 65 row [0]`
    - token/position/slot 都是 `9370 / 133 / 261`
  - 结果：
    - `layer_input_hidden = 0.0`
    - `layer_input_residual = 0.0`
    - `pre_attn_hidden = 0.0`
    - `attn_output.max_diff = 0.0068359375`
    - `post_attn_hidden.max_diff = 0.0078125`
    - `post_attn_residual.max_diff = 0.0068359375`
    - `mlp_output.max_diff = 0.008026123046875`
- 这轮结果把问题再次压缩：
  - `layer0` 仍完全一致
  - `layer1` 的首个非零差异出现在 block hook 的 `attn_output`
  - 根据 hook 定义，这个字段是
    `self.self_attn(...)` 的返回值，
    也就是 `layer1 self_attn` 之后、`post_attention_layernorm` 之前
- 当前下一步不再泛泛地说“继续 deeper”
  - 而是优先围绕 `layer1 self_attn` 做更细粒度验证：
    attention impl 本身 vs 它读到的历史 prefix K/V
  - 具体动作已经固定为：
    - 重跑 `layer1` attention prefix trace
    - 对齐 `baseline 73` vs `tree 65 row[0]`
    - 先看 `current_q/k/v` 与 `prefix_tail_k/v`
    - 再决定是继续扩历史窗口，还是直接下钻 `layer1 attention impl`
- 这轮 `layer1` attention prefix trace 首次尝试没有产生日志文件：
  - baseline run 正常结束，但
    `/tmp/baseline_prefix_trace_l1_v1.jsonl` 不存在
  - 随后核源码发现：
    decode 路径使用的 `trace_source` 实际是 `decode_fused`
  - 而我这轮环境变量误写成了：
    `fused_infer,decode_only,common`
- 当前结论：
  - 这是 instrumentation filter 配置错误
  - 下一步先改成包含 `decode_fused`
    再重跑 baseline/tree 的 `layer1` prefix trace
- 修正 `decode_fused` 后，baseline `layer1` prefix trace 已成功落盘：
  - `/tmp/baseline_prefix_trace_l1_v2.jsonl`
- 但 tree 侧新拿到的文件再次说明了工具边界：
  - `/tmp/tree_prefix_trace_l1_v2.jsonl`
  - 其中记录的是 `tree_row0_decode`
  - 仍然看不到要对齐的 speculative `row [0]`
- 因而当前下一步已经切换为：
  - 不再指望 `prefix trace` 直接看到 speculative row
  - 改为给 `tree exact debug` 增加 layer filter
  - 只抓 `layer1`
  - 再用它和 baseline prefix trace 做 row-level 对账
- 这层 instrumentation 已经接入代码：
  - [tree_attention.py](/root/worktrees/vllm-ascend-clean/vllm_ascend/attention/tree_attention.py)
  - `maybe_dump_tree_exact_debug()` 新增：
    - `VLLM_ASCEND_TREE_EXACT_DEBUG_LAYERS`
    - `VLLM_ASCEND_TREE_EXACT_DEBUG_ALL_TP`
- 在进入下一轮 trace 之前，先把新的待验证判断记到磁盘：
  - 当前第 `74` 个 token 的分歧点位于
    `“从古代的四大发明到现代的...”`
    这一自然语言自由续写空间较大的位置
  - baseline 续成：
    `高铁、航天、信息技术等领域的飞速发展`
  - tree 续成：
    `科技创新，中国在历史长河中不断演进`
  - 当前先把它视为“可能是 logits margin 很近时的 residual drift”，
    不是直接复用旧 `slot206` 根因
- 紧接着对 sample trace 做了第一轮晚步数抓取，但先暴露出 instrumentation 自身的窗口限制：
  - `vllm_ascend/worker/model_runner_v1.py` 中
    `VLLM_ASCEND_SAMPLE_TRACE_STEPS` 默认是 `32`
  - 因而 `/tmp/qwen32_tree_eager_sample_late.jsonl`
    去重后只剩：
    - baseline `32` step
    - tree `32` step
  - 这份日志无法覆盖新的 `first_diff_index=74`
  - 下一步不是解释模型，而是把 trace 窗口显式放大后重跑
- 放大到 `VLLM_ASCEND_SAMPLE_TRACE_STEPS=128` 之后，新的关键定位结果已经拿到：
  - baseline:
    - `step 73 -> sampled 9370`
    - `step 74 -> sampled 104963`
  - tree:
    - `step 66 -> sampled [9370, 105355]`
    - `tree_terminal_row_indices = [1]`
  - 同步对账表明：
    - tree `bonus` 行仍大体对齐 baseline `step 73`
    - 但 tree `row [0]` 已在同一 step 内把
      `104963` / `105355` 的排序翻转
- 这把问题进一步前移到：
  - 不是 next-root replay 之后才出错
  - 而是 verify batch 内非 bonus speculative row 的 hidden/logits
    已经开始残余漂移
- 下一步固定为：
  - 挂 prefix/cache trace
  - 直接对账 baseline `step 74`
    vs tree `step 66 row [0]`
    的 logical prefix / slot / cache 视图
- 接续上一轮修复后，先不做新猜测，直接把已证实的 v2 materialization 结果补回磁盘文档。
- 已复核当前代码中的 `_copy_tree_accepted_path_kv_slots()`：
  - 现在已经按 `block_index/block_offset` 访问 paged KV cache
  - 不再把 flat `slot_id` 误当成第 0 维 block 索引
- 已核对修复后的定向 trace：
  - `/tmp/tree_prefix_after_materialize_v2.jsonl`
  - `/tmp/tree_sample_after_materialize_v3.jsonl`
  - `/tmp/tree_kv_materialize_v2.jsonl`
- 修复后的关键结果：
  - `step 15` 仍执行 `207 -> 206` copy
  - `step 16` 的 canonical `prefix slot 206` 已与 baseline 对齐
  - `step 17` sample 从 `31235` 修正为 `97145`
  - 同轮 accepted token 数从 `3` 提升到 `5`
- 当前动作切换为：
  - 先重跑完整 `32B eager` compare
  - 再根据新的 `first_diff_index` 判断是否还存在更晚的残余问题
- 接续上一轮 trace 会话，读取：
  - `/tmp/tree_kv_materialize_v1.jsonl`
  - `/tmp/tree_first_pass_after_materialize_v1.jsonl`
  - `/tmp/tree_sample_after_materialize_v1.jsonl`
- 已确认 accepted-path KV materialization v1 并非“根本没跑”：
  - `step 15` 四个 TP rank 都执行了
    `token_id=20412, source_slot=207, destination_slot=206`
- 同一轮结果没有改善：
  - `step 16` 仍为 `[[20412, 99489]]`
  - `step 17` 仍为 `[[31235]]`
- 这轮结论已写回 `task_plan.md` / `findings.md` / 专项调试文档：
  - 缺的不是“有没有做 copy”
  - 下一步要直接查 copy 之后 next-step attention 实际看到的 prefix/cache 是否变化，
    以及是否存在后续 overwrite / alternate-path state
- 已新增并完成一轮更窄的 materialize 后 prefix trace：
  - `/tmp/tree_prefix_after_materialize_v1.jsonl`
  - `/tmp/tree_sample_after_materialize_v2.jsonl`
- 当前最关键的新结果：
  - tree `step 16` 时
    - `current_input_ids=[99489]`
    - `current_slots=[207]`
    - `seq_len=80`
  - 对齐 baseline `step 19` 后发现：
    - `current_k/v` 已正确
    - `prefix slots 203/204/205` 已正确
    - 只有 `prefix slot 206` 仍保持错误值
- 这把问题进一步压缩到：
  - accepted token canonical slot `206` 没有出现在 attention 真正读取的 cache 视图里
  - 或者该 slot 在 copy 之后、step16 attention 之前被重新覆盖
- 随后做了代码级 layout 复核，发现当前 materialize v1 的实现本身就有物理索引错误：
  - paged KV cache 第 0 维是 `block`
  - 第 1 维才是 `block_size` 内 offset
  - 但 `_copy_tree_accepted_path_kv_slots()` 把 flat `slot_id`
    直接拿去索引第 0 维
- 这与现有 trace 完全吻合：
  - copy 日志存在
  - 但 canonical `slot 206` 在 attention gather 视图中没有变化
- 下一步改动已收敛为：
  - 把 materialize copy 改成 `[block_index, block_offset]` 级别的 gather/scatter
  - 然后重跑 step16/17 定向 trace
- 恢复上下文，确认当前工作树为 `/root/worktrees/vllm-ascend-clean`，分支为 `feat/ascend-tower-connector-lora`。
- 复核已有证据：proposer first-pass 污染已修复，主矛盾切到 target 侧 tree execution。
- 已补 tree-only target trace，并确认 target batch 已真正走到 `ChunkedPrefill + tree mask` 路径。
- 重跑 `async_scheduling=False` 用例，分歧点不变，排除调度因素。
- 重跑 chain-tree 用例，链式 tree 与 baseline 完全一致，说明剩余问题只在真分叉树语义。
- 已在 `vllm_ascend/attention/` 里实现 tree-only 精确 attention 执行路径，并接入标准 attention/C8 路径。
- 分数题 prompt 验证结果：`baseline` 与 `tree-eagle3` 逐 token 完全一致，说明新 attention fallback 已经真正修到一部分真分叉树场景。
- 长文本介绍中国 prompt 复跑结果：旧问题未完全消失，但分歧模式已变化，`first_diff_index` 从 7 变为 2。
- 进一步确认长文本 case 是“前两个生成 token 一致、第三个才开始分叉”，所以排查重点继续后移到第一轮 verify 之后的状态重建链路。
- 下一步：把排查重点从“attention mask 是否生效”转到 verifier / logits 行对齐与 accepted/bonus 回灌语义。
- 新增 Qwen3 attention backend trace，确认真实运行路径是 `Attention -> AscendAttentionBackendImpl -> CUSTOM`。
- 新增 tree exact debug trace，并在同一长文本 case 上完成实跑：
  - cache gather 回来的 current K/V 与本轮前向 direct K/V 完全一致
  - row0 在“整棵树一起算”和“只拿 prefix + row0 单独算”之间只剩很小数值噪声
- 这轮日志把怀疑面从 attention/gather/cache 进一步收缩到了 attention 之后的 logits/sampler/verifier 链路。
- 已把上述结论同步回 `findings.md` 与开发调试文档，避免后续重复围绕 attention 打转。
- 新增 `VLLM_ASCEND_TREE_FIRST_PASS_TRACE_PATH`，把 `accepted_paths -> terminal_row_indices -> rebuilt token/position/seq_len` 链路打通到日志里。
- 用新增 first-pass trace 重跑 32B eager 长文本 case 后确认：
  - 第一次非空 accept 出现在 step 7
  - step 8 的 next root token 为 `97145`
  - step 8 bonus 仍然对齐为 `29490`
  - 真正的首次 target 偏离被压缩到 step 9
- 做过一轮 root/bonus 行 isolated exact attention 的功能实验：
  - 改动后 `first_diff_index` 从 11 退回 2
  - 已撤回该改动
  - 再次复跑确认工作树恢复到 `first_diff_index = 11`
- 这轮结论是：
  - first-pass immediate replay 不是当前主根因
  - root paged decode kernel 单点也不是当前主根因
  - 下一步继续查 branching 场景下 target 侧的 cache/logical-prefix 或 hidden/logits 偏移
- 新增 sample-stage trace，并在同一长文本 case 上完成 baseline/tree 对账：
  - baseline `step 11` 与 tree `step 10` 在逻辑前缀都结束于 `...97145, 29490`
  - 但 raw hidden/logits 已经不同，tree row0 top-1 从 baseline 的 `115995` 变成 `100210`
  - 因而当前主分歧已被确认发生在 sampler/verifier 之前
- 扩展 exact debug，新增 `current_slots / prefix_tail_slots / prefix_last_k_head/v_head / current_k_heads/v_heads`：
  - 已确认“最后一个已接受 prefix token”的 cache 写回/回读在 64 层上一致
  - 当前不再优先怀疑 last-slot cache corruption
- 下一步：
  - 补整段 prefix tail continuity trace
  - 直接比较 baseline/tree 在首次分歧前一轮的 token/position/slot/block-table 视图
- 新增 `VLLM_ASCEND_TREE_STATE_TRACE_PATH` instrumentation，并在：
  - `_bookkeeping_sync()` async 分支
  - `_copy_valid_sampled_token_count()`
  - `_prepare_inputs()` 的 `update_num_computed_tokens_for_batch_change` 之后
  - `_prepare_inputs()` 的 `_correct_optimistic_seq_lens_cpu` 之后
  打出 tree runtime state。
- 首次 trace 运行时因为 `tree_terminal_row_indices` 早期不存在而报错；已补防御式 `getattr(..., None)`，问题已解除。
- 用 state trace 对 `pid=705018` 完成首轮复盘后确认：
  - `prev_sampled_token_ids` 始终只像单 token
  - `spec_token_ids` 长期为全 `-1`
  - `token_ids_cpu_*tail` 仍以 placeholder 为主
  - 即使 `num_computed_tokens` 和 `optimistic_seq_lens_cpu` 已推进到 `64/70` 左右，也没看到 tree accepted path 被真正 materialize
- 新增 tree-only exact continuity 对账后确认：
  - `step 7` accepted draft token `3837` 位于 row1 `slot 196`
  - round 8 / 9 prefix 对该 `slot 196` 的 K/V 读取在 64 层上完全一致
  - accepted draft continuity 与 bonus continuity 都正常
- 进一步对齐 baseline/tree sample trace：
  - baseline `step 10` vs tree `step 9` 虽然 sampled token 仍同为 `29490`
  - 但 hidden/logits 已经开始漂移
  - baseline `step 11` vs tree `step 10` 才出现可见 token 分叉
- 这轮结论把主矛盾从“accepted path 没 materialize”进一步上提为：
  - non-attention 路径的 batch-dependent drift
  - 或 forward context 额外状态泄漏
- 新增临时脚本 `/root/tmp_tree_only_step11.py`，用于固定 tree-only step11 复跑并生成 continuity / sample / verify / target 联合日志。
- 新增临时脚本 `/root/tmp_linear_iso_probe.py`，用于只跑 tree-only 长文本并挂载线性层 isolated probe。
- 首次运行 `/root/tmp_linear_iso_probe.py` 失败：
  - 原因不是模型逻辑，而是误用了系统 `/usr/bin/python3`
  - 该解释器没有安装 `torch`
  - 正确环境已定位到 `/root/venvs/vllm-ascend/bin/python`
- 在正确 venv 上做最小 `torch` 自检时又发现：
  - 直接起 `/root/venvs/vllm-ascend/bin/python` 会触发 `torch_npu` backend auto-load
  - 若未显式关闭，会报 `libhccl.so` 缺失并提示 `source set_env.sh`
  - 后续统一带 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 启动，交给脚本自己的 CANN bootstrap 处理剩余环境
- 首轮 linear isolated probe 真正跑进模型后又暴露出 probe 自身的一处边界错误：
  - 部分线性层调用发生在 `forward_context` 生命周期之外
  - 典型位置是 `sample_tokens` 阶段 drafter `combine_hidden_states` 路径
  - 旧 probe 直接访问 `_EXTRA_CTX`，会因 `Forward context is not set` 把整个 engine 打死
  - 现已改成：
    - 先 `try/except AssertionError` 探测 `get_forward_context()`
    - 只在 `forward_context` 存在、且 `is_draft_model=False`、且 `tree_spec_decode_enabled=True` 时采样
- 第二轮 probe 成功跑完整个 tree-only 用例，但没有生成任何 trace 文件。
  - 这不代表“线性层差异为 0”
  - 只代表 probe 条件没有命中
  - 当前更可能的问题是：
    - 在线性层入口处，top-level `forward_context.attn_metadata` 不一定还能稳定暴露 `tree_spec_decode_enabled`
    - 旧条件把真正的小 batch target forward 过滤掉了
  - 已进一步调整为：
    - 保留 `is_draft_model=False`
    - 不再默认强依赖 `tree_spec_decode_enabled`
    - 改用 `num_tokens<=16` 抓小 batch target forward
- 下一步：
  - 更新专项调试文档，明确 accepted-path continuity 已排除
  - 在线性/量化路径上增加 isolated probe
  - 如果 isolated probe 命中，再决定是修量化线性路径还是做 correctness-first override
- 基于 linear probe 长时间未命中的情况，新增了更上层的定位分流：
  - 在 `sample_tokens` 处直接对 tree target row0 做 isolated full forward
  - 计划同时导出：
    - batched row0 hidden/logits
    - isolated row0 hidden/logits
    - top-k token/logit
  - 目的不是立即修功能，而是先回答一个更关键的问题：
    - 当前漂移是否在“同一 row0、同一 prefix、仅因 batch 形态不同”时就已经存在
    - 还是要继续回到 state reconstruction / row selection 方向
- 这条新 trace 代码已接入 `vllm_ascend/worker/model_runner_v1.py`，但截至当前记录点还没有完成首轮有效落盘验证。
- 已完成 row0 isolated full-forward trace 的首轮真实运行，但结果不是有效对账，而是 helper 自身暴露出新的执行语义错误：
  - 运行脚本：`/root/tmp_tree_only_step11.py`
  - 环境必须补齐 CANN 的 `PYTHONPATH/LD_LIBRARY_PATH/PATH`，否则 `import acl` 会直接失败
  - 进入首轮 sample 后，worker 在 `_maybe_dump_tree_row0_isolated_trace()` 内崩溃
  - 栈上显示 isolated rerun 走进了：
    - `attention_v1.py -> forward_fused_infer_attention -> torch_npu.npu_fused_infer_attention_score`
  - NPU 报错为：
    - `When attnMask is not provided, sparseMode must be 0`
  - 当前解释：
    - helper 构造出的 isolated metadata 还不满足 Ascend fused attention 的约束
    - 因而这次失败只能说明 instrumentation 路径不正确，不能说明 tree root row0 本身语义有问题
- 下一步先修 helper：
  - 要么强制 isolated rerun 进入 exact / masked 路径
  - 要么构造完全合法的 decode-only metadata
  - 修好后再重新执行 row0 batched-vs-isolated 对账
- 已对 `model_runner_v1.py` 的 row0 isolated helper 做第一轮修正：
  - 不再把 isolated row0 伪装成 `DecodeOnly + attn_mask=None`
  - 改为构造：
    - 单请求
    - 单 query token
    - `ChunkedPrefill`
    - `tree_spec_decode_enabled=True`
    - `1x1 zero tree mask`
  - 目的：强制 isolated rerun 走已有的 tree exact attention path，而不是重新落回 fused decode
- 同时把 helper 包成“失败只记日志、不打崩主跑”的调试逻辑：
  - 任何 isolated rerun 异常都会落到 `/tmp/tree_row0_iso.jsonl`
  - 主 32B 推理本身继续跑完，便于同时保留：
    - tree 输出结果
    - isolated helper 的错误上下文
  - 这样后续不会再因为 instrumentation 自己有问题而丢掉整轮主实验
- 第二轮实跑又补出一条关键实现约束：
  - 我把 isolated rerun 只放在 `TP rank 0` 上执行，结果不是“更省成本”，而是直接把多卡同步挂住
  - 日志表现：
    - 长时间无 `tree_row0_iso.jsonl`
    - 无主结果文件
    - EngineCore 报 `No available shared memory broadcast block found in 60 seconds`
  - 结论：
    - nested `_model_forward` 本身包含 TP collectives
    - 不能只让一个 TP rank 进入这段 rerun
  - 正确做法应改成：
      - 所有 TP ranks 共同参与 isolated rerun
      - 只有 rank0 负责写 trace
- 这轮挂起进程已人工清理，避免残留 worker 干扰后续实验。
- 第三轮实跑已经把 row0 isolated helper 完整打通：
  - 所有 TP ranks 共同参与 isolated rerun
  - 只有 rank0 落盘 `/tmp/tree_row0_iso.jsonl`
  - 主 tree-only 32B eager 推理也正常结束，结果落在 `/tmp/tree_only_result_step11.json`
- 这轮最关键的新结论：
  - `step 0..15` 的 batched row0 vs isolated row0
    - hidden 完全一致
    - logits 完全一致
    - top-k 完全一致
  - 关键偏离步 `step 9` 也一致：
    - `row0_input_ids=[29490]`
    - `root_seq_len=71`
    - batched/isolated top-1 都是 `100210`
    - batched/isolated top-2 都是 `115995`
- 由此把主矛盾进一步收缩为：
  - tree row0 本身不是被 sibling rows 在 batched non-attention 路径上污染
  - 更可能是 tree 与 baseline 在进入 row0 之前，历史 cache/prefix 状态已经不同
- 下一步不再继续扩张 row0 isolated helper，而是转向：
  - 给 baseline decode-only 路径补同类型 prefix/cache trace
  - 用真实 baseline/tree 同前缀样本做逐层对账
- 当前已确认下一轮具体动作：
  - 先改 `attention_v1.py`，补统一 schema 的 `prefix-cache trace`
  - 再分别重跑 baseline / tree 32B eager
  - 最后按 `...97145, 29490` 这一逻辑前缀做逐层 prefix-cache 对账
- 已完成统一 schema 的 `prefix-cache trace` 第一版接线，并做了 baseline 首轮实跑。
- baseline 首轮实跑结果：
  - `/tmp/baseline_prefix_result.json` 正常生成
  - `/tmp/baseline_prefix_trace.jsonl` 未生成
  - 代码核对后确认 baseline eager 实际走的是 `forward_fused_infer_attention`，不是 `forward_paged_attention`
  - 下一步改动从“继续跑”切换为“把 trace 补到 fused decode 路径后重跑”
- baseline 第二轮实跑没有进入有效对账：
  - trace 已补到 fused decode
  - 但 `forward_impl()` 里漏传 `layer`，导致 worker 在 `attention_v1.py` 抛 `NameError: name 'layer' is not defined`
  - 当前已修正 `forward_impl(layer, ...)`，准备第三轮 baseline 重跑
- baseline 第三轮与 tree 同步 trace 已完成：
  - `/tmp/baseline_prefix_trace.jsonl`
  - `/tmp/tree_prefix_trace.jsonl`
  - `/tmp/tree_prefix_sample.jsonl`
- 基于这三份日志已确认：
  - 按 step 对齐会被 tree 的多 token accept 误导
  - 按 `seq_len` 对齐后，`layer0` 在 `seq_len=72` 才首次出现真实分叉
  - 分叉顺序是：
    - `prefix_last_k/v` 仍一致
    - `current_k/v` 先分叉
  - 下一步改成补 `input_id / position / slot` 观测，不再继续怀疑更早的 prefix cache
- 已补最小二次观测：
  - `set_ascend_forward_context()` 现在保存 `positions`
  - `prefix-cache trace` 新增：
    - `current_input_ids`
    - `current_input_positions`
    - tree row0 decode 的 `current_slots`
  - 目标只针对 `seq_len=72` 的首次 current K/V 分叉做输入侧核对
- 首轮二次观测结果还不完整：
  - `current_input_ids / current_input_positions` 仍然是 `null`
  - 原因已缩到 instrumentation 自身：
    `_EXTRA_CTX.extra_attrs` 未开放这两个字段
  - 下一步是补代理白名单后再重跑 baseline/tree
- 代理白名单补完后的 v3 trace 已完成：
  - `/tmp/baseline_prefix_trace_v3.jsonl`
  - `/tmp/tree_prefix_trace_v3.jsonl`
- 当前最新定位：
  - `seq_len=70/71`：
    - baseline/tree 的 `current_input_id / position / slot` 一致
  - `seq_len=72`：
    - baseline 当前 token 已是 `115995`
    - tree 当前 token 已是 `100210`
  - 因此 `seq_len=72` 的 K/V 分叉并不是根因本身，而是前一轮采样分叉的后果
  - 下一步转向补 `current_q_heads`，专门看 `seq_len=71`
- 当前又补了一层最小观测：
  - `prefix-cache trace` 新增 `current_q_heads`
  - 目标只验证 `seq_len=71` 这一轮 query 是否已先分叉
- `current_q_heads` 对账结果已经出来：
  - `seq_len=71` 的 baseline/tree `current_q_heads` 在当前 trace 切片上完全一致
  - 因而当前最小下一步改成补 `current_output_heads`，直接比较 layer0 attention output
- 当前已补 `current_output_heads`，并准备把 trace `head_dim` 提到整头大小再复跑
- `baseline v5 / tree v5` 已完成：
  - `/tmp/baseline_prefix_trace_v5.jsonl`
  - `/tmp/tree_prefix_trace_v5.jsonl`
- 首轮 v5 复盘一度误判为“观测字段全是 null”：
  - 根因不是 trace 失败，而是我把字段从顶层读取了
  - 实际字段位于每条记录的 `requests[0]` 下
  - 该解析错误已纠正，并同步写回文档
- 纠正后对 `seq_len=71` 完成程序化逐元素对账：
  - baseline `step 10` 对 tree `step 9`
  - `current_input_id / position / slot` 一致
  - `prefix_last_k/v` 一致
  - `current_q/k/v` 一致
  - `current_output_heads` 也一致，且 `max_diff = 0.0`
- 当前最新结论：
  - `layer0 attention` 已被排除
  - 下一步先不急着新加 block hook，
    而是复用现有 `prefix-cache trace` 去看更深层 attention 边界：
    - 先看 `layer1 self_attn.attn`
    - 用 `current_q_heads/current_output_heads` 判断分叉是否已跨过 `layer0`
    - 再决定是否需要对 `o_proj / post_attention_layernorm / mlp` 单独加观测
- 为了让后续 `layerN` trace 可重复执行，已把运行时 bootstrap 补进：
  - `/root/tmp_baseline_prefix_trace.py`
  - `/root/tmp_tree_only_step11.py`
- 这次修改只处理 CANN/ATB 的 `PYTHONPATH/LD_LIBRARY_PATH/PATH` 和进程自重启，
  不涉及模型逻辑；目的只是避免再次卡在 `import acl`
- 紧接着首轮 `layer1` deeper-layer 复跑失败：
  - baseline / tree 都在 engine worker 初始化阶段报
    `ParallelOpenMP.cpp:64 Invalid thread pool`
  - 尚未进入本轮 `seq_len=71` 的模型对账
  - 下一步先修启动环境对齐，再继续 deeper-layer trace
- 对齐旧稳定脚本的 worker 环境项后，`Invalid thread pool` 已解除。
- 但随后我把 baseline/tree 两个 4 卡任务并行跑，触发了 HCCL 端口冲突：
  - `port 16666 have already been bound`
  - 这轮依旧没有拿到可用的 `layer1` 语义对账结果
- 恢复接续会话上下文，重新核对 `task_plan.md / findings.md / progress.md / speculative_token_tree_debug_log.md` 与工作树状态。
- 确认当前不应再重复扩 attention trace：
  - `tree_attention.py` 的 vendor rowwise fallback 已经把长文本 `first_diff_index` 从 `11` 推后到 `20`
  - 最新主怀疑点是 `tree step16 -> step17` 的 next-root replay / async `input_batch` state materialization
- 已把这条新焦点和待抓字段同步回 4 份文档，避免后续又回到已被降权的 attention 假设。
- 当前下一步已经固定：
  - 串行重跑 baseline/tree 长文本 case
  - 打开 `first-pass/state/sample` 三类 trace
  - 只对齐 `tree step16 -> step17` 与 baseline `step18 -> step20`
- 已完成这轮串行复现并得到新反证：
  - baseline 结果仍为 `108` tokens
  - tree 结果仍为 `112` tokens
  - `sample + first-pass + state` 证据表明：
    - tree `step16` 仍输出 `[20412, 99489]`
    - tree `step17` 仍偏成 `31235`
- 随后又做了一轮定向 `tree target trace` 复现，专门只看问题点附近的 row0 输入：
  - 当前代码下的问题批次已经确认是：
    - row0 `input_id = 99489`
    - row0 `position = 79`
    - row0 `slot_mapping = 207`
  - 因而“next-root 的 token/position/slot 直接喂错”被当前证据削弱
- 当前新的主线改成：
  - 不再泛泛地说“state materialization 有问题”
  - 改成验证：
    - accepted terminal row 被提升为 root 后，
      canonical prefix KV / slot 是否真的被压回连续前缀
    - 尤其是 `terminal_row_indices = [2]` 这种 branch 不是第一个子节点的场景
- 重新串行补跑 block trace 后，`/tmp/tree_block_trace_v1.jsonl` 已正常生成并完成首轮程序化对账。
- 新结果：
  - `baseline seq68` 对 `tree seq73 row0(slot195)`：
    - `model.layers.0/1` 全链路 `0.0`
  - `baseline seq69` 对 `tree seq73 row1(slot196)`：
    - `layer0 layer_input_hidden = 0.0`
    - `layer0 pre_attn_hidden = 0.0`
    - `layer0 attn_output` 首次非零 `max_diff = 0.001953125`
    - 之后在 `post_attn/residual/mlp/layer1` 持续放大
- 这说明：
  - `row0` 之所以完全对齐，是因为 tree exact path 里本来就对 row0 做了 decode override
  - 真正还没对齐的是 non-row0 speculative row
  - 目前首个可见分叉点已经精确缩到：
    `slot196` 的 `layer0 self_attn` 返回值
- 下一步不再盲目扩 block hook，而是先回看：
  - `vllm_ascend/attention/tree_attention.py`
  - `forward_tree_exact_attention()`
  - `run_exact_tree_attention()` 的 row/prefix/mask 语义
  先区分是 exact attention 数学路径本身，还是更老 prefix K/V 的输入面仍有隐蔽漂移
- 已继续用现有 trace 做了一轮更便宜的 full-prefix 验证，而不是马上改功能代码：
  - baseline 重跑：
    `/tmp/baseline_prefix_trace_fullpref_v1.jsonl`
  - tree 首轮 full-prefix 重跑时，`VLLM_ASCEND_TREE_EXACT_DEBUG_STEPS=16` 只覆盖到 `seq66`
  - 这不是模型逻辑问题，是 instrumentation 的步数配太小
  - 已把 `STEPS` 提到 `700` 后重跑 tree：
    `/tmp/tree_exact_fullpref_v1.jsonl`
- full-prefix 对账结果：
  - `baseline seq69` vs `tree seq73 row1(slot196)` 的 `layer0` 上：
    - older prefix `slots 128..194` 的 `K/V` 全部 `0.0`
    - `slot195` 的 `K/V`（baseline prefix-last vs tree current row0）全部 `0.0`
    - `slot196 current_k/v` 也全部 `0.0`
    - 但 `slot196 current_output_heads` 仍有 `max_diff = 6.103515625e-05`
- 当前结论因此再次收缩：
  - 现在至少在已观测 head 上，older prefix / row0 prefix / current K/V 都对齐了
  - 剩余问题就是 non-row0 attention 执行路径本身与 baseline decode/fused kernel 不同值
  - 下一步应从“如何让 non-row0 复用更接近 baseline 的 Ascend attention 数值路径”来改，而不是继续扩 prefix 输入面日志
- 已按这个结论直接改功能代码，而不是继续只写调试：
  - 文件：
    `/root/worktrees/vllm-ascend-clean/vllm_ascend/attention/tree_attention.py`
  - 新实现：
    - `non-row0` 改成逐 row 抽取 compact visible KV
    - 然后走 `torch_npu.npu_fused_infer_attention_score`
    - 原 Python dense softmax path 保留为 fallback
- 语法检查已过：
  - `python3 -m py_compile vllm_ascend/attention/tree_attention.py`
- 随后直接重跑长文本主复现：
  - tree 结果文件：
    `/tmp/tree_only_result_step11.json`
  - baseline 结果复用：
    `/tmp/baseline_prefix_result.json`
- 最新对比：
  - `baseline_count = 108`
  - `tree_count = 112`
  - `first_diff_index = 20`
  - 旧版本这里一直是 `11`
  - 当前前 20 个 token 已与 baseline 完全一致
- 这轮结论：
  - 新 fallback 确认有效
  - early divergence 已显著被推后
  - 但 tree 长文本仍未与 baseline 完全同值，下一轮需要围绕新的 diff 点继续缩圈
- 已继续补 baseline/tree 的 sample trace，而不是只看最终 token：
  - baseline:
    `/tmp/baseline_sample_after_vendor_v1.jsonl`
  - tree:
    `/tmp/tree_sample_after_vendor_v1.jsonl`
- 基于 TP0 的 step 对账，当前最重要的新发现是：
  - `tree step16` 一次输出 `[20412, 99489]`
  - 其中：
    - `bonus` 行与 baseline `step18` 完全一致
    - `row_label=[1]` 行与 baseline `step19` 完全一致
  - 紧接着 `tree step17 bonus/row0` 才首次分叉成 `31235`
  - 对应 baseline `step20` 本应是 `97145`
- 所以当前问题已经发生了转移：
  - non-row0 attention 修复有效，已经把早期问题推走
  - 现在需要重点回到：
    - multi-token accept 之后的 first-pass root replay
    - input_batch/token_ids/seq_len/state materialization
  - 也就是从 `tree step16 -> step17` 这条链上找错位，而不是继续怀疑 attention math
- 随后改成 baseline/tree 串行重跑，`layer1` 探针终于拿到有效结果：
  - `seq_len=71` 时，
    `model.layers.1.self_attn.attn` 的 `current_input / q / k / v` 仍完全一致
  - 但 `current_output_heads` 已出现微小非零差异：
    - `max_diff = 1.8310546875e-04`
    - `mean_diff = 5.607865750789642e-05`
- 在同一组 `v6` 文件上继续程序化对账，又拿到一条更早的证据：
  - `seq_len=70` 时，
    `layer1` 的 `prefix_tail_slots=[193,194,195,196]` 中只有 `slot 196` 已开始漂移
  - 数值为：
    - `prefix_tail_k_heads.max_diff = 0.125`
    - `prefix_tail_v_heads.max_diff = 0.0018515586853027344`
  - 这说明：
    - `seq_len=71` 当前轮 attention output 的微小 drift，
      更像是来自更早历史 token 的高层 K/V 已经轻微分叉
    - 下一步要直接查这个历史 token 在 `layer0 -> layer1` 边界上的最早分叉点
- 继续把 slot/token 对上之后，已确认：
  - 这个发生漂移的 `slot 196` 对应 token `3837`
  - 它正是先前通过 accepted-path 进入前缀的那个非 row0 token
- 这条映射很关键，因为它说明：
  - 之前的 row0 isolated exact 对账并没有覆盖真正可疑对象
  - 当前更像是“accepted row 写入 cache 的高层表示”和 baseline 不一致
  - 而不是“row0 后续读取时临时出错”
- 继续复用现有日志做了一轮更直接的 baseline-vs-accepted-row 对账：
  - 先用 baseline `seq_len=68` 的 bonus row 去匹配 tree exact debug 的 shard，
    锁定 `pid=725850`
  - 再比较 accepted row `slot196/token3837`：
    - `layer0 current_k/v` 与 baseline `seq_len=69` 完全一致
    - `layer1 current_k/v` 才首次出现微小差异
- 这轮结果把问题进一步钉在：
  - 不是 accepted row 从一开始就错
  - 而是它在同一个 token 的 `layer0 -> layer1` 之间开始偏离 baseline
  - 下一步需要给 accepted row 补 `current_output_heads` 或 block-level 输出观测
- 已补 `current_q_heads / current_output_heads / tp_rank` 到 tree exact debug，
  并重跑拿到 accepted row 的完整对账：
  - `layer0` 上：
    - `row1 current_q/k/v` 与 baseline 仍完全一致
    - 但 `row1 current_output` 仍有极小差异：
      `max_diff = 6.103515625e-05`
  - `layer1` 上：
    - `row1 current_q.max_diff = 0.04296875`
    - `row1 current_k.max_diff = 0.125`
    - `row1 current_v.max_diff = 0.0018515586853027344`
- 同时做过一轮“逐 row 独立 exact attention”的功能实验：
  - 目标是排除 batched multi-row GEMM 的数值路径
  - 结果：上述 `layer0 output` / `layer1 qkv` 差异完全不变，只是生成吞吐下降
  - 结论：
    - 当前 root cause 不是 batched-q exact attention 的实现方式
    - 这条功能改动没有收益，应撤回，仅保留本轮 instrumentation
- 下一步改成串行跑 baseline 再跑 tree，不再并行占卡
- 已按串行方式完成 `layer1 self_attn.attn` 的 baseline/tree 对账：
  - baseline:
    `/tmp/baseline_prefix_trace_v6.jsonl`
  - tree:
    `/tmp/tree_prefix_trace_v6.jsonl`
- `seq_len=71` 的结果：
  - `current_input_id / position / slot` 一致
  - `current_q/k/v` 一致
  - `layer1 current_output_heads` 首次出现微小非零差异：
    `max_diff = 1.8310546875e-04`
- 当前下一步：
  - 继续沿 attention 边界看更深层
  - 判断这类 tiny drift 是从 `layer1` 开始逐层累积，
    还是在某个更深层才突然放大
- `v5` 这轮 tree-only exact debug 已成功落盘：
  - `/tmp/tree_only_exact_debug_step11_v5.jsonl`
  - `/tmp/tree_only_result_step11.json`
  - 运行结束前被外部 `SIGTERM` 收掉，但结果文件完整可用
- 随后对 `v5` 做程序化对账，拿到当前最强的新证据：
  - 对齐 shard 仍取 `tp_rank = 0`
  - 对齐点为：
    - `seq_len = 73`
    - `current_slots = [195, 196, 197, 198, 199, 200]`
    - accepted row `slot 196 / token 3837`
  - 在这个视角下，它会消费到的 recent generated prefix
    是 `slots 188..195`（位置 `60..67`）
  - 对账结果：
    - `layer0`：
      - `prefix slots 188..194` 的 K/V 全部 `0.0`
      - `current slot 195` 的 `q/k/v/output` 全部 `0.0`
      - `current slot 196` 仅 `output.max_diff = 6.103515625e-05`
    - `layer1`：
      - `prefix slots 188..194` 的 K/V 也全部 `0.0`
      - `current slot 195` 的 `q/k/v/output` 仍全部 `0.0`
      - `current slot 196` 开始出现：
        - `q.max_diff = 0.04296875`
        - `k.max_diff = 0.125`
        - `v.max_diff = 0.0018515586853027344`
        - `output.max_diff = 0.00048828125`
- 这轮结果的直接含义：
  - recent generated prefix `60..67` 已不再是主怀疑点
  - accepted token `3837` 的直接前驱 `slot 195 / position 67` 也还是 exact
  - 当前应把下一步改成：
    - 优先补 block-level hook
    - 专查 `layer0 attention output -> layer1 input` 之间的哪一段先分叉
    - 如果还需要查“历史 prefix”，就直接查更老 prompt prefix，而不是重复查最近几步 generated prefix
- 恢复本轮接续会话后，先重新读取：
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
  - `docs/source/developer_guide/Design_Documents/speculative_token_tree_debug_log.md`
  并核对了最新 trace 文件，确认当前不该再回到旧的 early attention 假设。
- 对 `tree_first_pass` 代码做了第一次直接复核：
  - `build_tree_first_pass_inputs()` 当前对 promoted row 的
    `positions / hidden_states / slot_mapping`
    都只是 `index_select(terminal_row_indices)` 后直接复用
  - 代码里没有显式的 accepted-path canonical re-materialize
- 基于这个代码观察，已把新的更窄验证目标同步回文档：
  - 不再泛泛地说“continuity 可能有问题”
  - 改成直接验证：
    - baseline 对应 `99489` current token 的 row0 full-prefix K/V
    - tree `step17` row0 full-prefix K/V
  - 如果这里已经分叉，就优先修 promoted sibling row -> next root 的 canonical continuity；
    如果这里仍一致，再把问题面收回到 row0 当前轮 decode/fused 路径或更后层
- 上述晚步数 trace 已经串行补完：
  - baseline:
    `/tmp/baseline_prefix_trace_late_v1.jsonl`
  - tree:
    `/tmp/tree_prefix_trace_late_v1.jsonl`
- 关键对齐点已经抓到：
  - baseline `step19`
  - tree `step16`
  - `current_input_id = 99489`
  - `seq_len = 80`
  - `current_slot = 207`
- 程序化对账结果：
  - `current_input_id / current_input_position / current_slot` 一致
  - `current_q / current_k / current_v` 一致
  - full-prefix 中只有最后一个 prefix slot 分叉：
    - `slot 206 / position 78`
    - `prefix_tail_k_heads.max_diff = 26.0`
    - `prefix_tail_v_heads.max_diff = 0.322265625`
  - 同时当前轮 `current_output_heads.max_diff = 0.023468017578125`
- 这轮结果把问题进一步钉死为：
  - 多 token accept 后，
    第一个 accepted draft token 没有被 materialize 到 canonical prefix slot
  - 当前不再需要继续证明 continuity 是否断掉，
    下一步直接改 accepted-path KV materialization
- 在真正改代码前，又把这次实现约束补回了 4 份磁盘文档：
  - `accepted_paths` 需要保留到 runner
  - source row 需要从 accepted runtime node 回推到当前 target execution row
  - destination slot 需要按 position + block table 重算 canonical slot
  - bonus token 不参与 prefix materialization
- 已实现 accepted-path KV materialization `v1`：
  - `rejection_sampler` 现在会把完整 `accepted_paths` 挂到 runner
  - `model_runner_v1.propose_draft_token_ids()` 在 `drafter._propose(...)` 前
    依据 accepted runtime node 规划 slot copy
  - 当前实现对所有已绑定 KV cache tensor 组件执行 gather-then-scatter
- 已用 `/root/tmp_tree_eager_quick_check_noasync.py` 做真实 32B eager 复跑：
  - baseline 仍正常
  - tree 仍然 `first_diff_index = 20`
  - diff token 仍是 `97145` vs `31235`
- 因而这轮代码虽然成功接入，但效果尚未传导到最终语义：
  - 下一步不再盲猜
  - 先给 materialization helper 加独立 trace
  - 直接核对问题步是否真的发生了 `207 -> 206` 的 copy

- 恢复本轮接续会话后，先重新核对：
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
  - `speculative_token_tree_debug_log.md`
  以及现有 trace 文件是否已经完整落盘。
- 当前恢复后的单一待验证问题已固定为：
  - `baseline index = 73`
    vs
    `tree index = 65 row [0]`
    这对 `layer1` 对齐样本里，
    到底是 `current_q/k/v` 先漂，
    还是 `current_output` 才漂。
- 本轮执行顺序也已固定：
  - 先解析：
    - `/tmp/baseline_prefix_trace_l1_v2.jsonl`
    - `/tmp/tree_exact_layer1_late_v1.jsonl`
  - 不先重跑 4 卡任务
  - 只有现有日志不足以回答该分流时，才继续补新的 trace
- 上述解析已经完成，并得到了新的收敛结论：
  - 对齐点：
    - baseline `step = 73`
      `current_input_ids = [9370]`
      `current_input_positions = [133]`
      `current_slots = [261]`
    - tree `index = 65 row[0]`
      的 `query_pos = 1`
      对应同一逻辑行
  - `current_q/k/v` 全部 `0.0`
  - `current_output.max_diff = 0.00013399124145507812`
  - 但同一行的可见 prefix slots `253..260` 中：
    - `253..255` 仍 `0.0`
    - `256..260` 的 `layer1 K/V` 已开始轻微漂移
- 这把问题重新前移成：
  - 当前 token `9370` 本身的 `q/k/v` 还没漂
  - 真正先漂的是它最近几格历史 prefix 的 `layer1 K/V`
  - 下一步要沿 `slot 256..260` 继续往回追具体 token 和更早 block 边界
- 已继续用现有 baseline `layer1` trace 反查 `slot 256..260`：
  - `slot 256 -> step 68 -> token 9370 -> position 128`
  - `slot 257 -> step 69 -> token 104978 -> position 129`
  - `slot 258 -> step 70 -> token 102832 -> position 130`
  - `slot 259 -> step 71 -> token 26939 -> position 131`
  - `slot 260 -> step 72 -> token 100390 -> position 132`
- 下一步会继续只用现有 `tree_exact_layer1_late_v1`，
  逐个对齐这 5 个 current rows，
  找出哪个 token 在自己那一轮就首次把 `layer1 current K/V/output` 写偏。
- 上述逐 token 对齐已经跑完一轮：
  - `slot 256 -> token 9370`
    已是当前最早的已观测非零样本
  - 它在自己作为 current row 的那一轮就已有：
    - `q_max = 0.04296875`
    - `k_max = 0.125`
    - `v_max = 0.00225830078125`
    - `out_max = 0.000732421875`
  - `slot 257..260` 也都继续非零
- 下一步继续用现有日志向前再验一轮 `slot 253..255`，
  确认 `slot 256` 是否就是当前最早漂移点。
- 对 `slot 253..255` 的前验现已完成：
  - `slot 253 -> token 1773`
  - `slot 254 -> token 45181`
  - `slot 255 -> token 102640`
  在它们自己作为 current row 的那一轮，
  `layer1 current_q/k/v/output` 仍全部 `0.0`
- 这意味着当前最早的已证实漂移点已经正式收敛到：
  - `slot 256 / token 9370 / position 128`
- 下一步不再继续扫 prefix slots，
  而是直接围绕这个 token 的 `layer0 -> layer1` block 边界继续下钻。

- 恢复当前会话后，先重新读取了：
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
  - `speculative_token_tree_debug_log.md`
- 本轮执行顺序已经明确固定：
  - 先把新的单一待验证判断写回文档
  - 然后只解析现有 block trace：
    - `/tmp/baseline_block_trace_late_v2.jsonl`
    - `/tmp/tree_block_trace_v1.jsonl`
    - `/tmp/baseline_block_trace_l1_v1.jsonl`
    - `/tmp/tree_block_trace_l1_v1.jsonl`
  - 不先启动新的 4 卡 trace 任务
- 当前唯一待验证问题是：
  - `slot 256 / token 9370 / position 128`
    在自己作为 current row 的那一轮，
    `layer0` 路径上的首个非零字段到底是什么
  - 只有现有 trace 不足以回答这个问题时，
    才继续补最小化定向 instrumentation
- 上述 block trace 解析已经完成，并拿到了目前最关键的新收敛结果：
  - baseline:
    - `/tmp/baseline_block_trace_late_v2.jsonl`
    - `index = 68`
  - tree:
    - `/tmp/tree_block_trace_late_v2.jsonl`
    - `index = 61`
    - `qpos = 0`
  - 对齐 token/position/slot：
    - `9370 / 128 / 256`
- 结果不是“`layer1` 才第一次非零”，而是：
  - `layer0 layer_input_hidden = 0.0`
  - `layer0 pre_attn_hidden = 0.0`
  - `layer0 attn_output = 0.0009765625`
  - `layer0 post_attn_hidden = 0.000732421875`
  - `layer0 post_attn_residual = 0.0009765625`
  - `layer0 mlp_output = 0.00390625`
  - 随后 `layer1 layer_input_hidden = 0.00390625`
- 这把问题再次前移：
  - 当前最早的已证实漂移点
    已经不是 `layer1 current_q`
  - 而是 `slot 256` 自己那一轮的
    `layer0 attn_output`
- 下一步不会直接重跑大任务，
  而是先判断：
  - `layer0` 这一轮读到的更老 prefix K/V
    是否已经先漂
  - 或 attention exact path 在相同 `q/k/v` 面下仍存在 residual 差异
- 继续盘点 `/tmp` 后已经确认：
  - `layer0` 没有现成的晚步数 prefix/exact 日志
    能直接覆盖 `slot 256`
  - 现有 `layer0` prefix trace 文件没有命中
    `current_slots = [256..261]`
  - `tree_exact*.jsonl` 也没有对应晚步数 `layer0` 命中
- 因而下一步动作已经固定：
  - 不改业务逻辑
  - 只用现有 env 开关
  - 串行重跑一轮 `32B eager noasync`
  - 补一份 `layer0` 定向 trace
- 该定向重跑现已完成：
  - baseline:
    - `load_seconds = 81.89`
    - `generate_seconds = 12.59`
    - `token_count = 108`
  - tree:
    - `load_seconds = 81.80`
    - `generate_seconds = 178.31`
    - `token_count = 115`
- 新生成的目标文件：
  - `/tmp/layer0_prefix_compare_v1.jsonl`
  - `/tmp/tree_exact_layer0_256_v1.jsonl`
- 对 `slot 256 / token 9370 / position 128 / prefix_len 128` 的新结果：
  - baseline `step 68`
  - tree exact `index 61 row0`
  - `prefix_tail_slots 128..255` 全部对齐
  - `prefix_tail_k/v` 全部 `0.0`
  - `current_q/k/v` 全部 `0.0`
  - 但 `current_output = 6.103515625e-05`
  - tree 内部 `current_k/v` 的 direct-vs-gather 仍是 `0.0`
  - `row0_output_max_diff_vs_isolated = 0.0001220703125`
- 这意味着：
  - 先前“更老 `layer0 prefix K/V` 已漂”的怀疑点已经被证伪
  - 当前剩余问题已经压到
    `layer0 exact attention` 自身
- 紧接着继续用同一轮新日志做了三向比较：
  - baseline `decode_fused step 68`
  - tree `tree_row0_decode step 61`
  - tree exact `index 61 row0`
- 结果比“multi-row exact 有问题”更窄：
  - `baseline vs tree_row0_decode current_output = 6.103515625e-05`
  - `baseline vs tree exact row0 current_output = 6.103515625e-05`
  - `tree_row0_decode vs tree exact row0 current_output = 0.0`
  - `baseline vs tree_row0_decode` 的
    `prefix_tail_k/v` 和 `current_q/k/v`
    仍然全部 `0.0`
- 这说明：
  - 问题不是 multi-row exact 独有
  - tree 单行 root decode 路径
    在同输入面下就已经与 baseline 有稳定微差
- 当前下一步已切到：
  - 查 `decode_fused` vs `tree_row0_decode`
    的 metadata 和调用参数差异
- 在读 `attention_v1.py` 时，已经抓到一个更具体的结构差异：
  - baseline 这一步走的是 `decode_fused`
  - tree row0 在
    `forward_tree_exact_attention()`
    里当前被直接送进 `forward_paged_attention(...)`
- 这把下一步从“继续看日志”改成了功能验证：
  - 先让 tree row0 override 与 baseline decode 共用 dispatch 逻辑
  - 再重跑最小化 compare 看最早微差是否消失
- 该功能改动现已完成：
  - 文件：
    - `vllm_ascend/attention/attention_v1.py`
  - 改动点：
    - `forward_fused_infer_attention()` 支持自定义 `trace_source`
    - tree row0 override 不再硬编码 `forward_paged_attention()`
    - 现在会按与 baseline 相同的 dispatch 条件，
      选择 `tree_row0_fused` 或 `tree_row0_paged`
- 当前下一步只剩复跑验证：
  - 看 `slot 256` 的最早 `6.103515625e-05`
    是否消失
- 第一轮复跑验证已经返回了更具体的失败信息：
  - baseline 正常完成
  - tree 在 `forward_tree_exact_attention() -> forward_fused_infer_attention()`
    处失败
  - 关键报错：
    - `When attnMask is not provided, sparseMode must be 0`
- 这说明：
  - tree row0 已经真正改走 fused path
  - 但 `row0_metadata` 还缺 fused kernel 所需的 `attn_mask`
- 当前动作已经切换为：
  - 不回退 dispatch 改动
  - 先补齐 `row0_metadata.attn_mask`
  - 然后再重跑最小化验证
- 上述 metadata 修正现已完成：
  - `row0_metadata.attn_mask`
    现已通过 `AttentionMaskBuilder` 构造
  - `row0_metadata.seq_lens_cpu`
    也已同步补齐
 - 当前下一步仍是最小化复跑：
   - 先确认 tree row0 fused path 能正常执行

## 2026-06-30

- 已按当前 `reset --mixed` 后的两个工作树状态，新增完整变更报告：
  - [speculative_token_tree_change_report.md](/root/worktrees/vllm-ascend-clean/speculative_token_tree_change_report.md)
  - 报告内容覆盖：
    - 上游 `v023` 改动
    - Ascend tree 实现主线
    - 调试/验证资产
    - 当前现状
    - 未完成项
  - 这份报告对应的是：
    - `/root/worktrees/vllm-spec-tree-v023`
      已回退到 `967c5c3bc`，tree 改动停留在工作区
    - `/root/worktrees/vllm-ascend-clean`
      已回退到 `5d4964c`，tree 改动停留在工作区

- 恢复上下文后，先完成了这一轮“还能继续优化哪里”的代码复核：
  - 对照了：
    - `vllm_ascend/worker/model_runner_v1.py`
    - `vllm_ascend/spec_decode/tree_first_pass.py`
    - `vllm_ascend/spec_decode/tree_verifier.py`
    - `vllm_ascend/sample/rejection_sampler.py`
    - `/root/worktrees/vllm-spec-tree-v023` 上游对标代码
  - 当前新增的几个关键结论：
    - tree attention 主链路已经不是第一瓶颈
    - 剩余优化空间主要转向：
      - 固定树形本身的 acceptance 浪费
      - verifier / metadata 的 Python/CPU 重排
      - graph / async 生产形态
    - 当前 tree verifier 仍是 greedy-only 特化，
      尚未和普通 speculative decode 的 sampling / rejection 语义统一
    - `_calc_tree_spec_decode_metadata()` /
      `select_tree_accept_paths()` /
      `build_tree_first_pass_inputs()`
      这三处仍保留每步 host-side 小开销
  - 这些结论已同步写回 `findings.md`，
    下一步会基于它们给出新的优化优先级建议。

- 恢复上下文后已先完成最新性能结果核对：
  - 上一个 tree-only batched profile 任务已自然结束
  - 结果文件已落地：
    - `/tmp/qwen32_tree_batched_profile_v2.json`
    - `/tmp/tree_fast_profile_v2.jsonl`
    - `/tmp/tree_attn_trace_dispatch_v2.jsonl`
- v1/v2 对比结论：
  - tree 端到端 `generate_seconds`
    从 `27.01s` 下降到 `23.45s`
  - `non_root_ms`
    从 `1.494ms` 下降到 `0.809ms`
  - 其中 `nr_plan_ms`
    从 `0.660ms` 下降到 `0.028ms`
  - 说明新增的 visible-row template cache
    已经真实生效
- 当前瓶颈再收敛：
  - `plan` 已不再是主要问题
  - 下一刀优先级改为：
    - 去掉 `gather_paged_kv_to_ragged()`
    - 去掉 prefix duplication / packed prefix copy
- 已据此确定下一轮主改动方向：
  - 为 non-root rows 新增
    `prefix-only FIA + current-only FIA + lse merge`
    的 split path
  - 目标是把 prefix attention 直接留在 paged cache 上算，
    只对树上当前 token 做小型 dense pack

- 上述 split path 已完成一轮真实实现与验证：
  - `tree-only` 冒烟可跑通：
    - `/tmp/qwen32_tree_batched_split_smoke.json`
    - `generate_seconds ≈ 16.8s`
  - profile 证明该路径的结构收益是真实的：
    - `gather_ms = 0`
    - `pack_ms ≈ 0.024ms`
    - non-root 主成本已变成
      两次 FIA + 一次 merge
  - 但完整 baseline/tree compare 仍不正确：
    - `/tmp/qwen32_tree_batched_split_compare.json`
    - `/tmp/qwen32_tree_batched_split_compare_v2.json`
    - 两轮都在 `first_diff_index = 11` 处分叉
    - baseline `108 token`
    - tree `104 token`
- 为了避免继续凭感觉猜，已新增 split-vs-legacy probe：
  - `/tmp/tree_split_compare_probe.jsonl`
  - 结论：
    - split 与 legacy 的 non-root attention 输出
      仅有 `1e-3 ~ 1e-2` 级别微差
    - `npu_attention_update` 合并结果
      与纯 `torch logsumexp` 合并完全一致
  - 因而当前判断已经更新为：
    - split 路径的数值漂移来自分段 attention 本身
    - 不是 merge 公式实现错误
- 当前策略已据此切换：
  - 保留 `batched_split` 作为实验路径
  - 正确主线回到当前已验证正确的 legacy batched 路径
  - 下一步继续压 legacy path 的 pack / copy / gather

- 恢复上下文后已完成一轮新的代码复核，结论比上轮更具体：
  - 当前正确主路径
    `attention_v1.py::_forward_tree_fast_attention_legacy()`
    仍然先做：
    - `gather_paged_kv_to_ragged()`
  - 随后
    `tree_attention.py::_run_vendor_tree_attention_batched()`
    又把 ragged dense KV
    再打包成 speculative rows 的最终可见 KV
  - 即 non-root 正确路径现在仍存在：
    - `paged KV -> ragged dense KV -> packed visible KV`
    这条双搬运链
- 由此确定本轮下一刀：
  - 不继续碰 `batched_split` 的数值问题
  - 而是直接把 legacy batched path 改成：
    - 从 paged cache 按最终 visible token 次序直接 pack
    - 跳过中间 ragged dense KV
- 预期收益口径也已固定：
  - 先看 `gather_ms` 是否显著下降
  - 再看端到端 `tree generate_seconds`
  - 同时必须继续维持 baseline/tree token 全量一致

- `paged KV -> packed visible KV` 这一版已经完成真实 32B compare + profile：
  - compare：
    - `/tmp/qwen32_tree_batched_compare_v4.json`
  - profile：
    - `/tmp/tree_fast_profile_v4.jsonl`
  - 结果：
    - baseline `10.12s`
    - tree `16.50s`
    - baseline/tree `108` token 全量一致
    - `same_text = true`
- 相比上一轮正确主路径：
  - `/tmp/qwen32_tree_batched_compare_v3.json`
    里的 tree `20.04s`
  - 这次又再下降了约 `3.54s`
- profile 结果与代码改动完全对上：
  - `gather_ms`
    从 `0.662ms` 直接降到 `0.0`
  - `total_ms`
    从 `1.837ms` 降到 `1.336ms`
  - 说明这次收益的主要来源，
    确实就是去掉了中间 ragged dense KV
- 同时也发现了新的热点迁移：
  - `non_root_stats.plan_ms`
    从 `0.029ms` 升到 `0.157ms`
  - 原因是 direct paged path 每层都要重新把
    `block_tables + seq_lens + tree_mask`
    展开成 visible `slot` 计划
- 因而下一轮动作已经固定：
  - 保留当前 direct paged 正确路径
  - 继续把
    `output_indices / packed_slots / actual_seq_lengths_kv`
    做成 step 级缓存
  - 目标是把 plan 重建从“每层一次”
    压成“每步一次”

- step 级 visible-plan 缓存这一轮也已完成真实 32B compare + profile：
  - compare：
    - `/tmp/qwen32_tree_batched_compare_v5.json`
  - profile：
    - `/tmp/tree_fast_profile_v5.jsonl`
  - 结果：
    - baseline `10.05s`
    - tree `14.96s`
    - baseline/tree `108` token 全量一致
    - `same_text = true`
- 这说明第二轮缓存优化继续成立：
  - 相对 `v4`：
    - tree `16.50s -> 14.96s`
    - 再下降约 `1.54s`
  - 相对 `v3`：
    - tree `20.04s -> 14.96s`
    - 总下降约 `5.08s`
- profile 结果也已经对上设计目标：
  - `gather_ms` 继续保持 `0`
  - `non_root_stats.plan_ms`
    从 `0.157ms` 回落到 `0.066ms`
  - `non_root_ms`
    从 `0.450ms` 进一步降到 `0.136ms`
  - `total_ms`
    从 `1.336ms` 再降到 `1.010ms`
- 到这一步，当前 attention 主路径的热点结构已经明显变化：
  - non-root tree attention 已不再是主要重头
  - 当前 remaining gap 更可能来自：
    - row0/baseline-equivalent attention 本身
    - proposer / draft / acceptance 链路
    - 或未来若要更进一步，则需要
      数值精确的新 kernel / 修正 split path

- 随后又做了一轮更偏架构清理的改动：
  - 把 shared `TreePagedPlan` 的 owner
    从 attention backend 层内，
    上提到 `model_runner_v1._patch_tree_spec_decode_attn_metadata()`
  - 现在每个 tree step 会先建一次 shared plan，
    再把同一个 plan 对象挂给所有 layer metadata
  - attention 层只消费 plan，不再默认自己构造
- 这轮验证结果：
  - `/tmp/qwen32_tree_batched_compare_v6.json`
  - `/tmp/tree_fast_profile_v6.jsonl`
  - baseline `10.23s`
  - tree `15.02s`
  - baseline/tree `108` token 继续全量一致
- 端到端收益这次没有继续明显扩大，
  但 profile 已把结构目标坐实：
  - `non_root_stats.plan_ms = 0.0`
  - 说明 per-layer plan build 已彻底离开 attention 热路径
- 这轮之后的当前判断已更明确：
  - target attention 路线上若还要继续做“大结构”，
    下一刀不该再压 non-root plan/gather/pack
  - 而应转向：
    - row0 与 non-root 的 single-pass all-row exact tree attention
    - 或者修通 split path 的数值精确版本
