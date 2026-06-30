# Findings

## Confirmed

- 2026-06-30 正式 `2048 / 2048` 离线 benchmark 已首次拿到有效 baseline，并同时暴露出一个此前单请求冒烟没有覆盖到的 mixed-batch 问题：
  - `baseline`
    - 成功完成
    - `2 x 2048 in / 2 x 2048 out`
    - `elapsed_seconds = 182.97`
    - `output_token_throughput = 22.39 tok/s`
    - `qps = 0.01093`
  - `tree_before`
    - 旧 `batched_legacy` 在第一轮 mixed scheduler step 直接崩溃
    - 根因：
      - shared non-root plan 误把“1 个 speculative 请求 + 1 个普通 prefill 请求”
        的整个 chunked-prefill batch 都当成 tree rows
      - 于是拿 `6x6` tree mask 去索引 `2048` 行 prefill request
      - 报错：
        `IndexError: index 5 is out of bounds for dimension 0 with size 5`
  - `tree_after`
    - 新 `batched` 也在同类场景崩溃
    - 根因同上，只是 all-row path 上报成：
        `IndexError: index 6 is out of bounds for dimension 0 with size 6`
  - 这说明：
    - 之前 tree 路径的 correctness 主要是在
      “单请求 / 无 mixed prefill 干扰”的验证口径下成立
    - 一旦进入真实长 benchmark，
      scheduler 会把 speculative request 与新 prefill request 混在同一 step
    - 当前 tree fast-path 需要显式识别这种 mixed batch
      并回退到普通 chunked-prefill attention，
      否则 plan 构造阶段就会越界

- 2026-06-30 服务化 benchmark 也补齐了一个重要限定条件：
  - `baseline` 服务结果：
    - `output_token_throughput = 22.268 tok/s`
  - `tree_after` 服务结果：
    - `output_token_throughput = 21.444 tok/s`
    - 约为 baseline 的 `0.963x`
  - 同时服务化文本 compare 显示：
    - `json_grid`
      - 首次差异在较后位置，tree 输出出现了额外续写段落
    - `numbered_catalog`
      - 首字符即偏移
  - 结论：
    - 离线 `tree_after ~= baseline`
      并不自动等价于
      服务化 `tree_after ~= baseline`
    - 当前 async / service path 仍存在输出漂移
    - 所以这轮服务 benchmark 只能作为性能辅证，
      不能作为精度验收通过证据

- 2026-06-30 已建立新的正式 benchmark 口径，不再沿用之前的短文本/108 token 冒烟结论：
  - 统一目标：
    - 测试集平均输入 `2048`
    - 测试集平均输出 `2048`
  - 统一模式：
    - `baseline`
    - `tree_before = batched_legacy`
    - `tree_after = batched`
  - 统一资产：
    - `/root/qwen32_2048_bench_common.py`
    - `/root/run_qwen32_2048_offline_bench.py`
    - `/root/run_qwen32_2048_service_bench.py`
    - `/root/qwen32_2048_bench_dataset.json`

- 当前 `2048` 输入集已经生成成功，且不再只是“平均接近”，而是逐条精确命中：
  - `json_grid`
    - `prompt_token_count = 2048`
  - `numbered_catalog`
    - `prompt_token_count = 2048`
  - 平均输入：
    - `2048.0`
  - 这意味着后续 benchmark 里的长度波动主要只剩输出侧与运行时调度，不再是输入样本本身不稳定。

- 2026-06-30 恢复代码复核后，当前“还能继续优化什么”已经可以分成三层，而不是继续泛泛地说性能：
  - 第一层：`acceptance / draft 形态`
    - 当前 tree 形态仍是固定：
      - `[(0,), (1,), (0, 0), (0, 1), (1, 0)]`
    - 但已有 benchmark 已经说明：
      - 自然长文 / 说明文场景下
        `acceptance_per_pos` 常常只在第一层有值，
        第二层几乎掉到 `0`
      - 强结构 JSON 场景下，
        第二层 acceptance 才明显抬高
    - 这意味着：
      - 现在的固定树对很多 prompt 来说
        在“多算 draft rows”
        但没有换回足够 acceptance
      - 一个很可能有真实收益的大方向是：
        - 做 `adaptive tree policy`
        - 让树的深度 / 分叉数随最近 acceptance 变化
        - 低 acceptance prompt 退化为更浅或更窄的树
        - 高 acceptance prompt 继续使用当前树或更激进的树
  - 第二层：`verifier / metadata runtime`
    - `model_runner_v1._calc_tree_spec_decode_metadata()`
      当前仍保留 Python 双层循环与每步 tensor 重建：
      - `bonus_logits_indices_list`
      - `draft_token_input_indices`
      - execution-order -> runtime-order draft token 重排
    - `spec_decode/tree_verifier.py::select_tree_accept_paths()`
      当前也仍是按 request / depth / node 的 Python 遍历，
      并在 host 侧构造 `AcceptedTreePath`
    - `spec_decode/tree_first_pass.py::build_tree_first_pass_inputs()`
      仍有：
      - `copy.copy(common_attn_metadata)`
      - 多个 `.cpu()` 镜像更新
      - `index_select` 后的小 tensor 重建
    - 这说明：
      - 当前 attention 主链路虽然已大幅收敛，
        但 tree runtime 仍有一批“每步都会发生”的
        host-side 重排 / 小 tensor 开销
      - 若要继续压固定树下的开销，
        这里仍有可做的中等收益优化
  - 第三层：`生产形态`
    - 当前已经证明 eager/noasync 下：
      - fixed tree 在强结构 JSON 上可以加速
      - 在自然长文上仍常被 proposer/verify 成本抵消
    - 但这还不等于真实服务吞吐结论，
      因为：
      - 还没有把当前 tree 方案真正搬到
        graph / async / 长驻服务形态下评估
    - 因而若目标是“真实部署收益”，
      后续仍必须把优化主线延伸到：
      - graph-friendly metadata
      - async scheduling
      - tree step 的稳定 shape / cache 复用

- 当前 tree verifier 的行为约束也已经更明确：
  - `sample/rejection_sampler.py` 中，
    `AscendTreeSpecDecodeMetadata` 分支现在只支持 greedy：
    - 直接对 `target_logits.argmax(dim=-1)` 做 tree verify
    - 然后走 `select_tree_accept_paths()`
  - 它不会复用普通 speculative decode 分支里的：
    - `bonus_sampler_output = self.sampler(...)`
    - `apply_sampling_constraints(...)`
    - `rejection_sample(...)`
  - 结论：
    - 当前 tree 路径的 verifier 语义和普通 speculative decode
      仍不是同一条抽象
    - 这既限制了通用采样能力，
      也使得自然语言场景下的 acceptance / 输出风格
      更依赖 strict greedy 匹配

- 与上游 `v023` 对照后，当前剩余的一部分 runtime 成本已可明确归因到 Ascend tree 实现本身：
  - 上游 `v023` 当前只支持：
    - `speculative_token_tree` 配置项校验与透传
  - 上游没有：
    - tree verifier
    - tree first-pass
    - tree attention execution backend
  - 结论：
    - 当前这轮看到的 tree runtime 成本
      不是“上游本来就这样”
    - 而主要是 Ascend 侧为了把 tree 执行链打通而新增的真实实现成本

- 2026-06-30 已新增可复用的多 prompt 基准脚本：
  - `/root/run_qwen32_tree_prompt_bench.py`
  - 用途：
    - 复用当前 `32B + eagle3 + speculative_token_tree`
      eager/noasync 环境
    - 对一组复杂 prompt 批量比较
      baseline/tree 的：
      - 输出一致性
      - 单 prompt 生成耗时
      - per-case acceptance
  - 输出文件：
    - 默认 5 case 合并结果：
      - `/tmp/qwen32_tree_prompt_bench_default_merged.json`
    - 超长 JSON case：
      - `/tmp/qwen32_tree_prompt_bench_json_long.json`

- 这轮复杂 prompt 基准把“什么类型的 prompt 更容易吃到 tree 收益”显式坐实了：
  - 默认 5 个复杂 case 全部满足：
    - `same_text = true`
    - `same_token_ids = true`
  - 结果分布：
    - `natural_longform`
      - baseline `17.03s`
      - tree `18.32s`
      - `speedup = 0.93x`
      - `acceptance_per_pos ~= [0.230, 0.006]`
    - `structured_report`
      - baseline `28.10s`
      - tree `30.68s`
      - `speedup = 0.92x`
      - `acceptance_per_pos ~= [0.235, 0.041]`
    - `timeline_list`
      - baseline `26.56s`
      - tree `26.82s`
      - `speedup = 0.99x`
      - `acceptance_per_pos ~= [0.304, 0.132]`
    - `markdown_table`
      - baseline `37.68s`
      - tree `39.39s`
      - `speedup = 0.96x`
      - `acceptance_per_pos ~= [0.288, 0.092]`
    - `json_dataset`
      - baseline `43.60s`
      - tree `41.88s`
      - `speedup = 1.04x`
      - `acceptance_per_pos ~= [0.349, 0.089]`
  - 结论：
    - tree 在自然长文、混合结构说明文中仍容易被额外 proposer/verify 开销吃掉
    - 输出 schema 越固定、分隔符和字段模式越稳定，
      acceptance 越容易抬高，
      tree 越容易真正快过 baseline

- 超长强结构 JSON case 已经给出明确正向加速：
  - case:
    - `json_long_grid`
    - 仅输出长度为 `16` 的 JSON 数组，
      每个对象固定 `8` 个字段
  - 结果文件：
    - `/tmp/qwen32_tree_prompt_bench_json_long.json`
  - 关键结果：
    - baseline `63.73s`
    - tree `50.37s`
    - `speedup = 1.265x`
    - `token_count = 760`
    - `same_text = true`
    - `same_token_ids = true`
    - `tree_num_drafts = 399`
    - `acceptance_per_pos ~= [0.526, 0.378]`
  - 结论：
    - 当前 unified all-row fast path 已经不只是“接近 baseline”
    - 在足够规则、长输出、可重复 schema 的复杂 prompt 上，
      tree 已能稳定转化成真实生成加速
    - 当前限制 tree 泛化收益的核心，
      已更偏向 acceptance / proposer，
      而不是 attention 主链路

- 2026-06-30 当前 attention 大结构优化已经拿到阶段性收口结果：
  - 新路径：
    - unified all-row exact batched tree attention
    - 即把旧的
      - `row0 override`
      - `non-root batched`
      合并成一条统一 `all rows -> paged KV -> packed visible KV -> single FIA`
  - 首轮显式实验：
    - `VLLM_ASCEND_TREE_FAST_IMPL=batched_all_rows`
    - 结果文件：
      - `/tmp/qwen32_tree_batched_all_rows_v1.json`
      - `/tmp/tree_fast_profile_all_rows_v1.jsonl`
    - 关键结果：
      - baseline `10.10s`
      - tree `10.65s`
      - `same_token_ids = true`
      - `same_text = true`
    - profile 均值：
      - `avg_total_ms = 0.2067`
      - `avg_row0_ms = 0.0`
      - `avg_non_root_ms = 0.1329`
  - 与上一版正确主线 `v6` 对比：
    - `v6 avg_total_ms = 1.0140`
    - `all_rows avg_total_ms = 0.2067`
    - `v6 tree = 15.02s`
    - `all_rows tree = 10.65s`
  - 结论：
    - 当前 attention 侧的最大历史瓶颈，
      即 `row0` 单独 override，
      已被结构性消掉
    - non-root FIA 本身已不是主要矛盾
    - 现在 tree 端到端时间已经基本贴近 baseline

- 上述实验已经足够支持把新路径正式收敛成主线，而不是继续挂在实验 env 名字上：
  - 当前 canonical 映射已改为：
    - `batched`
      - 新的 unified all-row exact fast path
    - `batched_legacy`
      - 旧的 `row0 override + non-root batched` 路径
    - `batched_split`
      - 继续保留为实验路径
    - `reference`
      - rowwise exact/debug 路径
  - 兼容性：
    - 历史实验名 `batched_all_rows`
      现在会被规范化到 `batched`
  - 结论：
    - 以后默认 `VLLM_ASCEND_TREE_FAST_IMPL`
      不设值时，直接走新的 unified 主线
    - 老路径仍可显式回退做对账

- `batched -> unified all-row` 的主线切换也已完成默认配置复验：
  - 命令特征：
    - 不设置 `VLLM_ASCEND_TREE_FAST_IMPL`
    - 直接运行
      `/root/tmp_tree_eager_quick_check_noasync.py`
  - 结果文件：
    - `/tmp/qwen32_tree_default_batched_v1.json`
    - `/tmp/tree_fast_profile_default_batched_v1.jsonl`
  - 关键结果：
    - baseline `10.25s`
    - tree `10.62s`
    - `same_token_ids = true`
    - `same_text = true`
    - profile 中：
      - `last_tree_impl = "batched"`
      - `all_rows_unified = true`
      - `avg_row0_ms = 0.0`
  - 结论：
    - 默认主线路径已经稳定切到 unified all-row exact fast path
    - 不是只有实验 env 名称下才成立
    - 当前 eager/noasync 32B case 中，
      tree 相对 baseline 只剩约 `3.7%` 额外开销

- 2026-06-30 的最新 `32B eager + noasync` 长文本复验，已经把“当前 tree fast path 到底有没有把 correctness 打穿”重新坐实：
  - 脚本：
    - `/root/tmp_tree_eager_quick_check_noasync.py`
  - 运行环境：
    - `VLLM_ASCEND_TREE_FAST_IMPL=batched`
    - `VLLM_ASCEND_TREE_ATTN_TRACE_PATH=/tmp/tree_attn_trace_long.jsonl`
  - 结果文件：
    - `/tmp/qwen32_tree_eager_quick_check_noasync.json`
  - 关键结果：
    - baseline:
      - `generate_seconds = 10.16`
      - `token_count = 108`
    - tree:
      - `generate_seconds = 26.16`
      - `token_count = 108`
      - `spec_decode_num_draft_tokens = 455`
      - `spec_decode_num_accepted_tokens = 16`
      - `num_drafts = 91`
      - `acceptance_per_pos[0] = 0.1758`
    - compare:
      - `same_token_ids = true`
      - `same_text = true`
  - 结论：
    - 当前 batched fast-path 代码在该真实 32B 长文本 case 上没有破坏正确性
    - 与之前历史基线相比，tree 生成耗时已从 `28.71s` 进一步降到 `26.16s`
    - 但仍然显著慢于 baseline `10.16s`
    - 当前真实差距大约是 `2.57x`

- 同一轮 trace 已经排除了一个新的误判方向：
  - 文件：
    - `/tmp/tree_attn_trace_long.jsonl`
  - 关键观测：
    - `impl_name = AscendAttentionBackendImpl`
    - 没有出现 `AscendC8AttentionBackendImpl`
  - 结论：
    - 这条 `Qwen3-32B-W8A8` 长文本 tree 路径，当前实际不在 C8 attention 子类里
    - 因而这轮性能瓶颈不能归咎为“W8A8 模型被量化分支绕回 C8 exact path”
    - 剩余优化主战场仍在通用 `AscendAttentionBackendImpl` 的 tree attention 路径

- 这轮 trace 也暴露了一个 instrumentation 层面的限制：
  - `attention_v1.py::_maybe_dump_tree_attn_trace()` 里的 `path` 字段，
    当前只区分：
    - `_should_use_tree_exact_attention(attn_metadata) == true`
    - vs fallback 普通 attention
  - 因而即使环境变量设成：
    - `VLLM_ASCEND_TREE_FAST_IMPL=batched`
    trace 里仍然会看到：
    - `path = "exact"`
  - 这不等于代码真的回到了 `forward_tree_exact_attention()`
  - 它只能说明：
    - 本轮命中了 tree-specialized attention gate
    - 但现有 coarse trace 还不能直接区分：
      - `forward_tree_exact_attention()`
      - `forward_tree_fast_attention()`
  - 结论：
    - 后续若要精确对比 `reference` vs `batched`，需要补更细的 dispatch trace
    - 或直接做同脚本双跑性能对照

- 基于最新长文本结果，当前剩余的主瓶颈画像已经更清楚了：
  - correctness 已再次确认收口
  - 生成性能虽然优于旧版，但仍明显落后 baseline
  - 又由于当前模型路径不走 C8 子类，
    剩余大头更集中在通用 tree attention 自身：
    - 每层 `gather_paged_kv_to_ragged()` 的完整 prefix gather
    - `_run_vendor_tree_attention_batched()` 里对每个 speculative row 复制一份 prefix
    - row0 仍需单独走一次 baseline decode path
  - 结论：
    - 下一轮不应再优先怀疑量化分支
    - 而应直接朝“去掉 prefix gather / prefix duplication”的更大结构优化推进

- 从当前 correctness-first 实现的代码结构看，已经有三处强热点，不需要等 profiler 才能先列入优化候选：
  - `vllm_ascend/attention/tree_attention.py::gather_paged_kv_to_ragged()`
    - 每层都会把 paged KV 按 request gather 成 ragged dense KV
    - 本质上是把完整 prefix `seq_len` 重新 materialize 一遍
  - `vllm_ascend/attention/tree_attention.py::run_exact_tree_attention()`
    - 先按 request 做 Python 循环
    - vendor path 下又按 speculative row 再做 Python 循环
    - 每一行都会重新 `torch.cat(prefix, visible_current)`，
      再单独调用一次 `npu_fused_infer_attention_score`
  - `vllm_ascend/attention/attention_v1.py::forward_tree_exact_attention()`
    - 先跑完整棵树的 exact attention
    - 然后对 row0 再单独走一次 baseline-style decode attention 覆写输出
    - 即 row0 存在明确的二次计算
  - 结论：
    - 即便不看 debug trace，这三处也足以解释当前 tree 相对 baseline 的大幅慢速
- runner / first-pass 侧还有一组第二梯队热点，会放大 attention 主热点的总开销：
  - `vllm_ascend/worker/model_runner_v1.py::_patch_tree_spec_decode_attn_metadata()`
    - 每次都会 `build_tree_target_attn_mask(runtime)`
    - 若已有 mask，还会 `mask.clone()` 再局部覆盖
    - 这是高频小批量分配/拷贝热点
  - `vllm_ascend/worker/model_runner_v1.py::_patch_tree_spec_decode_positions()`
    - 每次都新建 `depth_offsets` tensor，再做 expand/reshape
  - `vllm_ascend/spec_decode/tree_first_pass.py::build_tree_first_pass_inputs()`
    - 每轮 first-pass 都会：
      - `copy.copy(common_attn_metadata)`
      - 多次 `index_select`
      - 多次 `.cpu().clone()`
    - 这属于 accepted frontier 重建链上的高频元数据搬运
  - `vllm_ascend/worker/model_runner_v1.py::_copy_tree_accepted_path_kv_slots()`
    - 每次 slot copy 都重新创建
      `src_blocks/src_offsets/dst_blocks/dst_offsets`
      四组 tensor
  - 结论：
    - 第一梯队是 attention 主算子路径
    - 第二梯队是 per-step metadata/mask/slot-copy 重建开销
    - 两者都需要纳入优化路线，但优先级应先打第一梯队
- 本轮开始前再复核代码后，已确认有一组“无需改 attention 语义、可直接削掉”的热路径浪费：
  - `SpecTreeRuntime` / `build_tree_*` 当前仍反复把同一棵树的
    - execution order
    - depth offsets
    - target attn mask
    - per-depth draft attn mask
    重新算成 Python / CPU tensor
  - `NPUModelRunner._calc_tree_spec_decode_metadata()`
    当前每步都用 Python 双层循环把 execution-order draft ids
    重新排回 runtime order，且顺带重复创建：
    - `bonus_logits_indices`
    - `target_logits_indices`
    - draft input row indices
  - `_copy_tree_accepted_path_kv_slots()`
    当前对同一批 `slot_copies`
    会在每个 cache tensor 上都重新构造
    `src_blocks/src_offsets/dst_blocks/dst_offsets`
  - `tree_first_pass.build_tree_first_pass_inputs()`
    对同一个 `seq_lens / num_computed_tokens / slot_mapping`
    会多次生成 CPU 副本，其中一部分 clone 只是历史保守写法
  - 结论：
    - 这些点不会消除 rowwise tree attention 的主成本
    - 但它们都属于 correctness-first 路径上的真实热损耗，
      适合作为本轮第一批低风险优化
- batched tree fast path 的第一轮实跑结论已经拿到：
  - 已在 `attention_v1.py` 中把 tree 路径默认切到
    `forward_tree_fast_attention()`
  - 设计上：
    - row0 仍走已验证正确的 baseline decode path
    - non-root speculative rows 打包成一次 batched FIA 调用
  - `32B + eagle3 + speculative_token_tree` 首轮实跑时，
    fast path 已真实触发，但在 Ascend FIA TND 入口报错：
    - `When layout is TND and PA not enabled, keyT/valueT must be equal to the last element of actualSeqenceLengthKV`
  - 这说明当前 batched 实现的第一版参数组织仍不符合
    Ascend `npu_fused_infer_attention_score` 在 `TND` layout 下的长度约定
  - 当前最强判断：
    - 这里要求的不是“每个 request 的独立 KV 长度列表”
    - 而更可能是和现有 TND 路径一致的 cumulative/cumsum 风格长度描述
  - 结论：
    - fast path 主方向成立，且已经真实接入执行链
    - 当前阻塞点已经缩成一个明确的 kernel 接口契约问题，
      不是“fast path 根本不可行”
- 还有一个算法层面的性能瓶颈已经由实测直接给出：
  - 最新正确性对齐轮次里：
    - `num_drafts = 91`
    - `spec_decode_num_draft_tokens = 455`
    - `spec_decode_num_accepted_tokens = 16`
    - `acceptance_per_pos[0] ≈ 0.176`
  - 含义：
    - 当前接受率偏低
    - 即便算子侧完全优化，speculative decoding 的净收益也会被低 acceptance 明显压制
  - 因而性能规划里必须把“实现热点优化”和“acceptance 提升/无效 draft 降低”分开评估

- 当前性能问题已经是下一阶段主任务，不再只是附带现象：
  - 最近一次正确性对齐后的重型 trace 实测：
    - baseline `108` tokens / `13.77s`
    - tree `108` tokens / `162.71s`
  - 更早一轮较轻量的实测：
    - baseline `108` tokens / `10.06s`
    - tree `115` tokens / `28.71s`
  - 这说明：
    - debug trace 的确会显著放大总耗时
    - 但即使不看最重的 trace 轮次，tree 当前实现仍明显慢于 baseline
  - 下一步需要把“debug 开销”和“实现本身的结构性瓶颈”拆开看

- row0 root override 的 fused-path 修复已经得到首轮端到端验证：
  - 运行配置：
    - `Qwen3-32B-W8A8`
    - `RedHatAI/Qwen3-32B-speculator.eagle3`
    - `speculative_token_tree = [(0,), (1,), (0, 0), (0, 1), (1, 0)]`
    - eager
    - `async_scheduling=False`
  - 结果文件：
    - `/tmp/qwen32_tree_eager_quick_check_noasync.json`
  - 关键结果：
    - baseline `token_count = 108`
    - tree `token_count = 108`
    - baseline/tree `token_ids` 全量一致
  - 同轮 trace：
    - `/tmp/layer0_prefix_compare_v4.jsonl`
    - `trace_sources = {'decode_fused': 32, 'tree_row0_fused': 32}`
  - 结论：
    - tree row0 override 已不再错误地走 paged path
    - 修正后的 `tree_row0_fused` 与 baseline `decode_fused`
      已能把该长文本 case 拉回端到端一致
- 上述结论已被晚步数局部 trace 直接核销旧漂移点：
  - 文件：
    - `/tmp/layer0_prefix_compare_v5.jsonl`
  - 关键对齐点：
    - baseline `decode_fused step = 68`
    - tree `tree_row0_fused step = 61`
    - `current_input_ids = [9370]`
    - `current_input_positions = [128]`
    - `current_slots = [256]`
    - `seq_len = 129`
  - 结果：
    - `prefix_last_k_max_diff = 0.0`
    - `prefix_last_v_max_diff = 0.0`
    - `current_q_max_diff = 0.0`
    - `current_k_max_diff = 0.0`
    - `current_v_max_diff = 0.0`
    - `current_output_max_diff = 0.0`
  - 结论：
    - 旧的 `slot 256` `layer0 current_output = 6.103515625e-05`
      漂移已经消失
    - root row 的 late-step residual mismatch
      已被 `tree_row0_fused` 修复

- 新的 tree row0 fused rerun 失败不是“kernel 本身不支持”，而是 metadata 构造先断了：
  - 第一次失败：
    - fused path 真正被走到
    - 但报错：
      `When attnMask is not provided, sparseMode must be 0`
  - 第二次失败：
    - 给 row0 metadata 补 `attn_mask` 后
    - 运行时报：
      `AttributeError: 'AscendAttentionBackendImpl' object has no attribute 'attn_mask_builder'`
  - 代码归属复核结果：
    - `attn_mask_builder` 与 `model_config`
      当前都定义在 `AscendAttentionMetadataBuilder`
    - `AscendAttentionBackendImpl` 并没有这两个成员
  - 当前状态结论：
    - 这说明 row0 fused dispatch 改动已经真实进入 fused 路径
    - 但 `_build_tree_root_decode_metadata()` 的 mask 构造方式存在 owner 选错的问题
    - 下一步不是回退 fused dispatch，
      而是修 row0 decode metadata 的 mask 来源
    - 当前更偏向的实现方向是：
      - 在 backend impl 内用
        `AttentionMaskBuilder(device)` +
        `self.vllm_config.model_config`
        重建标准 causal mask
      - 这样更贴近 baseline `decode_fused`
        而不是直接复用 tree-patched `attn_metadata.attn_mask`

- 放大 `VLLM_ASCEND_SAMPLE_TRACE_STEPS=128` 后，新的第 `74` 个 token 分歧已经被精确定位到：
  - baseline:
    - `step 73 -> sampled 9370`
    - `step 74 -> sampled 104963`
    - `step 74 top logits`:
      - `104963: 30.0`
      - `105355: 29.625`
      - `99602: 29.5`
  - tree:
    - `step 66 -> sampled [9370, 105355]`
    - `tree_terminal_row_indices = [1]`
    - 同一 step 的 `bonus` 行 top logits 仍与 baseline `step 73`
      基本同序、同量级
    - 但同一 step 的 `row [0]` 已经变成：
      - `105355: 31.5`
      - `99602: 31.125`
      - `104963: 30.875`
  - 结论：
    - 新的首个分歧不是出现在 step 边界之后的 next-root replay
    - 而是在 tree `step 66` 内部，
      accepted token `9370` 对应的后继 speculative row `[0]`
      就已经把 baseline 的 `104963`
      翻成了 `105355`
    - 因而当前 residual issue 更靠近：
      - non-bonus speculative row 的 hidden/logits 计算
      - 而不是旧的 first-pass / canonical slot materialization
- sample trace 当前默认只记录前 `32` 个 step：
  - 代码位置：
    - `vllm_ascend/worker/model_runner_v1.py`
    - `max_steps = int(os.getenv("VLLM_ASCEND_SAMPLE_TRACE_STEPS", "32"))`
  - 现象：
    - `/tmp/qwen32_tree_eager_sample_late.jsonl` 只有 `256` 行
    - 去重后是 baseline `32` step、tree `32` step
    - 只覆盖到 baseline `32` token / tree `36` token
  - 结论：
    - 这份 trace 无法覆盖新的 `first_diff_index = 74`
    - 不是模型行为变化，而是 instrumentation 窗口默认值过短
- 32B eager 长文本完整 compare 已在 materialization block/offset 修复后重跑：
  - 结果文件：
    - `/tmp/qwen32_tree_eager_quick_check_noasync.json`
  - 关键结果：
    - baseline:
      - `token_count = 108`
      - `generate_seconds = 10.06`
    - tree:
      - `token_count = 115`
      - `generate_seconds = 28.71`
      - SpecDecoding metrics 中最终看到：
        - `Accepted = 6 tokens`
        - `Drafted = 180 tokens`
    - compare:
      - `first_diff_index = 74`
      - `baseline_token_id_at_diff = 104963`
      - `tree_token_id_at_diff = 105355`
  - 含义：
    - 之前围绕 `step16 -> step17` 的 late-step 主分歧
      已不再是端到端的首个分歧点
    - `first_diff_index` 已从修复前的 `20`
      大幅后移到 `74`
    - 当前 remaining issue 已经收缩成“更晚位置的 residual divergence”，
      不能再用旧的 `slot 206` 根因解释整轮结果
- accepted-path KV materialization 的 late-step 主问题已经被直接修正并验证：
  - 代码修正：
    - `_copy_tree_accepted_path_kv_slots()` 不再把 flat `slot_id`
      直接当成 cache 第 0 维索引
    - 改为先拆成：
      - `block_index = slot_id // block_size`
      - `block_offset = slot_id % block_size`
    - 再按 `[block_index, block_offset]` 做 gather/scatter
  - 验证 trace：
    - `/tmp/tree_prefix_after_materialize_v2.jsonl`
    - `/tmp/tree_sample_after_materialize_v3.jsonl`
    - `/tmp/tree_kv_materialize_v2.jsonl`
  - 关键观测：
    - `step 15` 仍记录到 `source_slot=207 -> destination_slot=206`
    - `step 16` 时：
      - `current_input_ids = [99489]`
      - canonical `prefix slot 206` 的 `K/V`
        已与 baseline `step19 prefix_last_k/v` 完全一致
    - `step 17` sample 已从错误的 `31235`
      变为 baseline 对齐的 `97145`
    - 同一窄 trace 中，accepted token 数也从 `3`
      提升到了 `5`
  - 结论：
    - 之前 v1 materialization “看似执行但不生效”的根因
      就是 paged KV 物理索引方式错误
    - 这个 late-step canonical prefix continuity 缺口已经修通
- accepted-path KV materialization v1 已被 trace 直接证实“代码确实执行了”：
  - trace:
    - `/tmp/tree_kv_materialize_v1.jsonl`
    - `/tmp/tree_first_pass_after_materialize_v1.jsonl`
    - `/tmp/tree_sample_after_materialize_v1.jsonl`
  - 在关键问题步：
    - `step 15`
    - 四个 TP rank 都记录到：
      - `token_id=20412`
      - `source_slot=207`
      - `destination_slot=206`
  - 但同一轮运行结果仍然是：
    - `step 16 sampled_token_ids = [[20412, 99489]]`
    - `step 17 sampled_token_ids = [[31235]]`
  - 与 materialize 之前没有任何改善
  - 结论：
    - “没有执行 accepted-path slot copy”已被证伪
    - 当前更合理的问题面是：
      - copied KV 没有被下一轮真正消费
      - 或 copied slot 在后续路径被覆盖
      - 或 next-root 的有效状态并不只依赖这一个 canonical prefix slot
- accepted-path KV materialization v1 的 follow-up prefix trace 已进一步锁定问题位置：
  - 新 trace：
    - `/tmp/tree_prefix_after_materialize_v1.jsonl`
    - `/tmp/tree_sample_after_materialize_v2.jsonl`
  - 关键点是 tree `step 16`：
    - `current_input_ids = [99489]`
    - `current_slots = [207]`
    - `seq_len = 80`
    - sample 仍为 `[[20412, 99489]]`
  - 与 baseline `step 19` 对账结果：
    - tree `current_k/v` 已与 baseline 当前 token `99489` 对齐
    - tree `prefix slots 203/204/205` 也都与 baseline 对齐
    - 唯独 `prefix slot 206` 仍不对
  - 结论：
    - 当前错误已经不是“整个 prefix tail 都错位”
    - 而是非常窄地落在：
      - canonical accepted slot `206`
      - 在 attention gather 可见面上仍保持错误值
    - 因此 materialize copy 不是没做，
      而是没写进 attention 实际读取的那份 cache，
      或写进去后在 step16 前被后续逻辑覆盖了
- 代码级复核已经发现当前 materialize v1 的一个明确实现错误：
  - `tree_attention.py::gather_paged_kv_to_ragged()` 表明 cache layout 是：
    - `key_cache.shape == [num_blocks, block_size, ...]`
    - gather 先按 `block_tables` 取第 0 维 block，再在第 1 维按 position 使用 offset
  - 但 `_copy_tree_accepted_path_kv_slots()` 当前实现是：
    - 直接拿 flat `slot_id = block * block_size + offset`
    - 去做 `cache_tensor.index_select(0, slot_ids)`
  - 这会把 flat slot id 错当成 block id
  - 结论：
    - materialize copy 逻辑当前确实写错了物理位置
    - 这与 trace 现象完全一致：
      - “copy 记录存在”
      - 但 attention 可见的 canonical `slot 206` 没变化
- row0 isolated full-forward 对账已经正式跑通：
  - 结果文件：`/tmp/tree_row0_iso.jsonl`
  - tree-only 主结果：`/tmp/tree_only_result_step11.json`
  - `step 0..15` 全部满足：
    - `hidden_max_abs_diff = 0.0`
    - `hidden_mean_abs_diff = 0.0`
    - `logits_max_abs_diff = 0.0`
    - `logits_mean_abs_diff = 0.0`
  - 且 top-k token/logits 完全一致，包括关键分歧步：
    - `step 9`
    - `row0_input_ids = [29490]`
    - `root_seq_len = 71`
    - top-1 均为 `100210`
    - top-2 均为 `115995`
  - 结论：
    - tree target row0 在当前 cache / state 条件下可被单独精确复现
    - sibling rows 并没有通过 batched non-attention 路径污染 row0 hidden/logits
    - “量化线性 / 通信 / MoE 的 batch-dependent drift”当前已被大幅降权
- tree-only 精确对账又补了一层关键证据：
  - `step 7` 首次非空 accept 为 `token_ids=[3837], bonus_token_id=97145, terminal_row_index=1`
  - target metadata 表明该 accepted draft token 位于 row1 的 `slot 196`
  - exact debug 对账显示：
    - round 7 当前 row1 `slot 196`
    - round 8 prefix `slot 196`
    - round 9 prefix `slot 196`
    - 全 64 层 K/V `max diff = 0.0`
  - 结论：accepted draft token continuity 正确
- bonus token continuity 也没有发现异常：
  - 已接受 prefix 最后 token 的 current/prefix K/V 对账为 `0.0`
  - row0 together vs isolated attention output 仍只剩极小 bf16 噪声
  - 因而 cache/gather/row0 attention isolation 已基本被排除为当前主根因
- 新增 `sample trace` 后已直接确认：
  - baseline `step 11` 与 tree `step 10` 在逻辑前缀都结束于 `... 97145, 29490`
  - baseline top logits 前二为：
    - `115995: 33.25`
    - `100210: 33.0`
  - tree row0 top logits 前二为：
    - `100210: 33.5`
    - `115995: 33.25`
  - 且 hidden states 头部与范数已经不同
  - 结论：当前主分歧发生在 sampler/verifier 之前，raw target hidden/logits 已经分叉
- sample trace 还确认：
  - baseline `step 10` 与 tree `step 9` 在产生 `29490` 时仍然对齐
  - 但这两个对齐 step 的 raw hidden/logits 已经开始出现偏移：
    - hidden head `max diff = 0.5`
    - baseline top-1 / tree top-1 仍同为 `29490`
  - 真正的首次 target 偏离是下一轮才出现
  - 因而问题不是“接受路径刚回灌完立刻错”，而是“再往后一轮的 target 前向语义开始漂移”
- 更强的含义是：
  - 可见 token 首次分歧发生在 `baseline step 11` vs `tree step 10`
  - 但不可见的 hidden/logits 漂移在前一轮 `baseline step 10` vs `tree step 9` 就已经开始
  - 这进一步支持“问题在 attention 之后或 attention 之外”而不是 verifier/sampler
- 新增 `tree_exact_debug` 进一步表明：
  - 当 prefix 从 `69 -> 70 -> 71` 增长时，下一轮 `prefix_last_k_head` 与上一轮 row0 `current_k_heads[0]` 完全一致
  - 快速汇总脚本已验证该 last-prefix-token 对账在 64 层上 `max diff = 0.0`
  - 因而“最后一个已接受 token 写错 cache / 读错 cache”当前没有证据支持
- 新增 `tree_first_pass` trace 后确认：
  - 第一次非空 accept 出现在 step 7：
    - `accepted_paths = [{"node_indices":[0],"token_ids":[3837],"bonus_token_id":97145,"terminal_row_index":1}]`
  - 下一轮 first-pass 重建结果是：
    - `rebuilt_token_ids = [97145]`
    - `rebuilt_positions = [68]`
  - 再下一轮 target bonus token 为 `29490`
  - 即：
    - `accept -> first_pass -> 下一轮 root 输入` 这条链在 step 7/8 上是连贯的，不是“第一次 accept 后立刻喂错 token”
- 长文本 case 在当前稳定代码下仍然是：
  - `first_diff_index = 11`
  - baseline token 前 20 个：
    - `[58695, 103987, 104320, 106565, 3837, 109068, 60686, 100490, 3837, 97145, 29490, 115995, ...]`
  - tree token 前 20 个：
    - `[58695, 103987, 104320, 106565, 3837, 109068, 60686, 100490, 3837, 97145, 29490, 100210, ...]`
- 首次分歧已经被进一步钉到：
  - step 8 bonus token 仍对齐，为 `29490`
  - step 9 bonus token 首次偏离，为 `100210`
  - 因而当前问题不是“第一次非空 accept 当轮立刻错”，而是“accept 之后再过一轮，target root/bonus 语义开始漂移”
- 做过一轮更激进的 root 修正实验：
  - 把 tree exact path 的 root/bonus 行从 `paged_attention` 改成 isolated exact attention
  - 真实 32B eager 复跑后，结果从 `first_diff_index = 11` 退化回 `2`
  - 已撤回该改动，当前工作树重新验证回 `first_diff_index = 11`
- 该实验的含义是：
  - “root paged decode kernel 本身就是主根因”这个怀疑点当前不成立
  - root 行改成 isolated exact 不但没修好，反而明显更差
- 因此当前主矛盾进一步收缩为：
  - branching tree 在 target 侧的后续轮次上，仍存在 cache/logical-prefix 或 hidden/logits 级别的语义偏移
  - 不是 first-pass immediate replay 错
  - 也不是 root row paged decode kernel 单点问题
- tree proposer first-pass 已改为 terminal-row 压缩，下一轮 root 输入不再携带旧 sibling / descendant rows。
- 32B eager 下 tree 输出质量已明显好转，但 deterministic greedy 仍在约第 7 个生成 token 首次分歧。
- 在 first-pass 修复后，early all-reject 轮次仍出现 bonus token 早期偏移，说明剩余问题更偏向 target 侧执行语义。
- `async_scheduling=False` 不改变问题，`first_diff_index` 仍为 7。
- tree target batch 现在已经被强制走成 `ChunkedPrefill` 风格路径：
  - `common_tree_spec_decode_enabled = true`
  - `metadata_attn_state = ChunkedPrefill`
  - `metadata_num_decodes = 0`
  - `metadata_num_prefills = 1`
  - `metadata_num_decode_tokens = 0`
  - `metadata_num_actual_tokens = 6`
- 即便强制走 masked prefill-style attention，输出完全不变，说明“误走 decode fast path”这个假设已被证伪。
- 链式 tree 与 baseline token 序列完全一致；只要出现真分叉，输出就仍在约第 7 个生成 token 首次偏离。
- 因此当前剩余问题是“branching-specific”，不是“multi-token target verification 普遍错误”。
- 已在 `vllm_ascend/attention/` 接入 tree-only exact attention fallback：
  - 显式 gather paged KV 到 ragged dense KV
  - 对 suffix speculative rows 按 tree mask 做精确 attention
  - 标准 attention 与 C8 路径均已接入
- 新 fallback 不是无效改动：
  - 数学分数题 prompt 下，`baseline` 与 `tree-eagle3` 已逐 token 完全一致
  - `first_diff_index = null`
  - 结果文件：`/tmp/qwen32_tree_compare_result.json`
- 但长文本介绍中国 prompt 下，旧复现并未完全消除：
  - 分歧从原来的 `first_diff_index = 7` 提前到了 `first_diff_index = 2`
  - baseline 第 2 个分歧 token 为 `104320`
  - tree 第 2 个分歧 token 为 `116037`
  - 结果文件：`/tmp/qwen32_tree_eager_quick_check_noasync.json`
- 从长文本 case 的 token 前缀看：

## Pending Validation

- 新的候选对齐已经完成验证：
  - baseline block trace `index = 73`
    - `current_input_ids = [9370]`
    - `position = 133`
    - `slot = 261`
  - tree block trace `index = 65` 的第二个 query row
    - `current_input_ids = 9370`
    - `position = 133`
    - `slot = 261`
  - 二者在 `layer0` 上的以下字段全部逐元素一致：
    - `layer_input_hidden`
    - `pre_attn_hidden`
    - `attn_output`
    - `post_attn_hidden`
    - `mlp_output`
    - 对应 norm
- 结论：
  - 先前的对齐假设成立
  - 但 residual divergence 还没有在
    “accepted token `9370` 对应的 speculative row”
    这一层发生
- 紧接着做过一轮 `baseline index = 74`
  vs `tree index = 66 bonus row` 的比较：
  - 这对行共享 `position = 134 / slot = 262`
  - 但 token 已经不同：
    - baseline `104963`
    - tree `105355`
  - 且 `layer_input_hidden` 就已明显分叉
- 这条比较已被定性为“无效定位样本”：
  - 它只说明 token 级分叉已经传导进下一轮 root
  - 不能回答首个根因在哪一层出现
- 因而当前正确下一步是：
  - 保持真正有效的对齐点不变：
    - baseline `index = 73`
    - tree `index = 65 row [0]`
  - 继续向 `layer1+` 扩 block trace，
    找到这对“同输入、同 position、同 slot”调用的首个深层分叉位置。
- `layer1` block trace 现已给出首条直接证据：
  - 对齐点仍是：
    - baseline `index = 73`
    - tree `index = 65 row [0]`
    - token/position/slot 都是 `9370 / 133 / 261`
  - `layer1` 上：
    - `layer_input_hidden = 0.0`
    - `layer_input_residual = 0.0`
    - `pre_attn_hidden = 0.0`
    - `attn_output.max_diff = 0.0068359375`
    - `post_attn_hidden.max_diff = 0.0078125`
    - `post_attn_residual.max_diff = 0.0068359375`
    - `mlp_output.max_diff = 0.008026123046875`
- 结合 `patch_qwen3vl.py` 的 hook 位置可知：
  - 这里的 `attn_output` 是 decoder block 中
    `self.self_attn(...)` 的返回值
  - 也就是 `layer1 self_attn` 之后、`post_attention_layernorm` 之前
- 结论：
  - 当前首个深层非零差异已经被缩到
    “同输入调用的 `layer1 self_attn` 输出”
  - 下一步如果继续 trace，应优先盯：
    - `layer1 attention impl`
    - 或 `layer1` 看到的更老 prefix K/V
  - 不应再回到 `layer0` 或分叉后的下一轮 root 去兜圈子。
- 当前下一条待验证分流已经明确：
  - 对同一对齐点补 `layer1` attention prefix trace
  - 直接比较：
    - `current_q/k/v`
    - `prefix_tail_k/v`
    - `current_output`
  - 若 `current_q/k/v` 已不同，问题会前移到 `layer1 qkv` 投影
  - 若 `current_q/k/v` 与可见 prefix 都一致、但 `current_output` 不同，
    则 root cause 更接近：
    - `layer1 attention impl`
    - 或 prefix 中更老、未被 tail 覆盖的历史 K/V
- `layer1` attention prefix trace 的首轮重跑没有生成任何日志文件：
  - `baseline` 运行完成，但
    `/tmp/baseline_prefix_trace_l1_v1.jsonl` 不存在
  - `tree` 运行还在进行时就已经确认
    baseline 侧没有落盘
- 代码复核给出的最直接原因是：
  - `attention_v1.py` 里的 decode trace source 实际是 `decode_fused`
  - 先前环境变量写成了
    `VLLM_ASCEND_PREFIX_CACHE_TRACE_SOURCES=fused_infer,decode_only,common`
  - 过滤器很可能把真实 decode 记录全挡掉了
- 结论：
  - 这一步当前首先是 instrumentation 配置错误
  - 不是新的模型行为证据
- 更关键的新认识是：
  - 即便 `prefix trace` 配置修正后，
    它能看到的 tree 侧记录仍主要是 `tree_row0_decode`
  - 也就是 root row 的 decode override
  - 它拿不到当前要对齐的 multi-row speculative `row [0]`
- 因而下一步更合适的工具不是继续挤 `prefix trace`
  - 而是给 `tree exact debug` 增加 layer filter
  - 只抓 `layer1`
  - 直接导出 speculative row 的
    `current_q/k/v/current_output/prefix_tail`
- 这层 instrumentation 现已补齐：
  - `tree_attention.py::maybe_dump_tree_exact_debug()`
    已支持：
    - `VLLM_ASCEND_TREE_EXACT_DEBUG_LAYERS`
    - `VLLM_ASCEND_TREE_EXACT_DEBUG_ALL_TP`
  - 可以在不改功能路径的前提下，
    直接只抓 `layer1` 的 speculative rows
  - 前两个生成 token 仍然一致
  - 第三个生成 token 才首次分叉
  - 这说明第一轮 tree attention / verifier 至少部分成立
  - 剩余问题更像是第一轮 verify 之后的“下一轮状态重建 / accepted-bonus 回灌”错误，而不是首轮 target attention 直接错误
- Qwen3 真实运行时走的是：
  - `Attention`
  - `AscendAttentionBackendImpl`
  - backend `CUSTOM`
  - 不是 DSA / DeepSeek 专用 backend。
- tree target 在 Qwen3 上已经 100% 进入 `attention_v1.py` 里的 tree exact path：
  - `tree_attn_trace` 共 2048 条记录全部为 `path=exact`
  - 因此“其实没走到 exact path”这个怀疑点已被证伪。
- 新增 `tree_exact_debug` 实跑结论：
  - 当前轮 direct current K/V 与 cache gather 回来的 current K/V 完全一致：
    - `current_k_max_diff = 0`
    - `current_v_max_diff = 0`
  - row0 在“整棵树一起算”和“只拿 prefix + row0 单独算”之间几乎一致：
    - 16384 条记录里最大差异 `0.015625`
    - 大多数层为 `0` 或极小 bf16 数值噪声
  - 因此：
    - cache 写回 / gather 不是当前主根因
    - row0 被 sibling rows 在 exact attention 内直接污染，也不是当前主根因
- 结合上面的新日志，attention 侧当前已经收敛到：
  - exact path 确实在跑
  - tree mask 在 exact path 内没有明显失效
  - 剩余大问题更可能位于 attention 之后的 hidden/logits/sampler 链路
- 新增 `tree_state_trace` 后，async tree runtime 已出现一组更强的状态侧信号：
  - 观测对象 `pid=705018`
  - 在 step `0..23` 的 trace 里，`prev_sampled_token_ids` 始终表现为单 token 形态，例如：
    - `[[58695]]`
    - `[[103987]]`
    - `[[104320]]`
    - `[[106565]]`
    - `[[3837]]`
  - 同时 `spec_token_ids` 长期仍是 `[-1, -1, -1, -1, -1]`
  - `request_output_tail`、`token_ids_cpu_computed_tail`、`token_ids_cpu_no_spec_tail` 中也长期保留大段 `-1` / `0` placeholder
  - 即便到 `step 19/20`，已经出现：
    - `prev_num_draft_tokens = [5]`
    - `num_computed_tokens = [64]`
    - `input_batch_num_computed_tokens_cpu = [69]`
    - `optimistic_seq_lens_cpu = [70]`
    - `prev_sampled_token_ids = [[3837]]`
    - `spec_token_ids` 仍全 `-1`
  - 结论：
    - async tree 路径当前没有把真实 accepted path 实体化回 `spec_token_ids / token_ids_cpu`
    - 当前 batch state 更像是在复用线性 speculative decode 的 placeholder 约定，而不是 tree accepted-path 语义
- `tree_state_trace` 的辅助结论：
  - 新 trace 首轮确实因为 `tree_terminal_row_indices` 尚未创建而崩过一次
  - 改成 `getattr(self, "tree_terminal_row_indices", None)` 后复跑稳定
  - 这次崩溃不是 root cause，只是 instrumentation 缺保护

## Active Hypotheses

- 新主假设（待验证）：
  - 当前 remaining divergence 更像是
    tree verify batch 内的非 bonus speculative row
    在相同已接受前缀下仍有 residual drift
  - 直接依据是：
    - `bonus` 行还能与 baseline `step 73` 基本对齐
    - 但 `row [0]` 已在同一 step 内把
      `104963` / `105355` 的排序翻转
  - 下一步验证应优先回答：
    - baseline `step 74`
      与 tree `step 66 row [0]`
      是否真的消费了同一 logical prefix / slot / cache 视图
    - 如果 prefix 一致，那问题就进一步收缩到
      non-bonus speculative row 的 attention/block/logits 路径
- 新主假设（待验证）：
  - 在修掉 `slot 206` continuity bug 之后，
    当前第 `74` 个 token 的 residual divergence
    更像是“长程累积下的轻微语义/数值漂移”
    而不是早前那种明显的 prefix materialization 硬错误
  - 当前依据只有：
    - `first_diff_index` 已从 `20` 后移到 `74`
    - baseline/tree 在分歧点之前共享很长前缀
    - 分歧位置对应的自然语言片段是：
      - baseline: `从古代的四大发明到现代的高铁、航天、信息技术...`
      - tree: `从古代的四大发明到现代的科技创新，中国在历史长河中...`
    - 两边文本都仍然连贯、没有回到旧版那种明显异常输出
  - 因而下一轮验证优先级应改为：
    - 先抓第 `74` 个 token 附近的 sample/logits margin
    - 再看这是不是“top-1 / top-2 很接近时被小 drift 推翻”的情况
    - 如果 margin 很大，再回头怀疑新的 state/cache 错位
- 当前最强怀疑点进一步调整为：
  - tree 与 baseline 在“逻辑前缀看似一致”时，某些历史 token 的高层 K/V 其实已经不同
  - 这种不同可能来自：
    - 更早期某一轮 token 在 `layer0 -> layer1` 之间就已经轻微分叉
    - 随后被写入 cache，之后各轮只是稳定读回这份已分叉的历史状态
    - 或 `o_proj / residual / mlp / block output` 路径在 baseline/tree 间存在细小但持续的数值差异
- 最新降权项：
  - recent generated prefix `positions 60..67`
  - 直接原因是 `v5` 已确认这整段 prefix 以及其直接前驱 `slot 195`
    在已对齐 shard 的 `layer0/layer1` 上都还是 exact
  - 因而如果还要继续怀疑“历史 prefix 已先漂移”，
    下一步也应该优先去查更老的 prompt prefix，
    而不是继续重复检查最近生成出的这几步
- 新增 block-level 对账结果：
  - 使用：
    - baseline: `/tmp/baseline_block_trace_v1.jsonl`
    - tree: `/tmp/tree_block_trace_v1.jsonl`
  - 对齐方式：
    - `baseline seq68` 对 `tree seq73 row0(slot195)`
    - `baseline seq69` 对 `tree seq73 row1(slot196, token 3837)`
  - 结果分成两部分：
    - `slot195`
      - `model.layers.0/1` 上
        `layer_input_hidden`
        `pre_attn_hidden`
        `attn_output`
        `post_attn_hidden`
        `post_attn_residual`
        `mlp_output`
        全部 `max_diff = 0.0`
      - 这与此前“row0 decode override 正常”的判断一致
    - `slot196`
      - `layer0 layer_input_hidden = 0.0`
      - `layer0 pre_attn_hidden = 0.0`
      - `layer0 attn_output` 首次非零：
        - `max_diff = 0.001953125`
      - 随后继续放大：
        - `layer0 post_attn_hidden max_diff = 0.000732421875`
        - `layer0 post_attn_residual max_diff = 0.001953125`
        - `layer0 mlp_output max_diff = 0.00390625`
        - `layer1 layer_input_hidden max_diff = 0.00390625`
        - `layer1 pre_attn_hidden max_diff = 0.000732421875`
        - `layer1 attn_output max_diff = 0.005859375`
        - `layer1 mlp_output max_diff = 0.015625`
- 这条 block-level 证据与 earlier attention trace 的组合结论是：
  - `attention_v1` 里的 `current_output_heads` 记录的是 attention impl 的输出，尚未经过 `o_proj`
  - `patch_qwen3vl.py` 里的 `attn_output` 记录的是 `Qwen3Attention.forward()` 返回给 decoder block 的值，
    即已经过 `o_proj`
  - 现象是：
    - non-row0 `slot196` 在 attention impl 输出处就已有微小漂移
    - 经 `o_proj` 后被进一步放大
    - 再在 residual / MLP / 下一层 attention 中继续放大
- 当前最窄 root-cause 面因此收缩为：
  - 非 `row0` speculative row 的 exact tree attention 计算本身
  - 或它消费的“更老 prefix K/V”已经存在未观测到的轻微漂移
  - 而不是 `layer0` 之后的 block glue / async batch state 才首次出错
- 新增 full-prefix head 对账结果：
  - baseline:
    `/tmp/baseline_prefix_trace_fullpref_v1.jsonl`
  - tree:
    `/tmp/tree_exact_fullpref_v1.jsonl`
  - 对齐点：
    - baseline `seq69`
    - tree `seq73 row1(slot196)`
    - `layer0`
  - 结果：
    - baseline `seq69` 的 prefix `slots 128..195`
    - 可拆成：
      - older prefix `slots 128..194`
      - `slot195` prefix-last
    - tree `row1(slot196)` 的可见输入面则是：
      - prefix `slots 128..194`
      - current row0 `slot195`
      - current row1 `slot196`
    - 程序化对账确认：
      - older prefix `slots 128..194` 的 `K/V` 全 `0.0`
      - `slot195` 的 `K/V`（baseline prefix-last vs tree current row0）全 `0.0`
      - `slot196 current_k/v` 也全 `0.0`
      - 但 `slot196 current_output_heads` 仍有
        `max_diff = 6.103515625e-05`
- 这条 full-prefix 证据的含义非常直接：
  - 至少在当前观测 head 上，
    `slot196` 的可见输入已经完全对齐
  - remaining 差异已经不再需要借助“older prefix 可能漂了”来解释
  - 当前最强判断应更新为：
    - non-row0 speculative row 的 attention 执行数值路径
    - 与 baseline decode/fused kernel 并不严格同值
  - 也就是说，当前 Python dense softmax exact path 即便语义上看似正确，
    也足以在 greedy 场景中积累出可见分叉
- vendor rowwise fallback 接入后的新确认：
  - 长文本主复现里，前 20 个 token 已与 baseline 完全一致
  - 新的首个分叉点从旧版 `11` 后移到了 `20`
  - 这说明 non-row0 attention math 确实是已修复的一部分主问题，不应回退
- fallback 之后的最新对齐证据：
  - baseline sample trace:
    `/tmp/baseline_sample_after_vendor_v1.jsonl`
  - tree sample trace:
    `/tmp/tree_sample_after_vendor_v1.jsonl`
  - 程序化对齐后可得：
    - tree `step16 bonus` 与 baseline `step18 sample` 一致，为 `20412`
    - tree `step16 row_label=[1]` 与 baseline `step19 sample` 一致，为 `99489`
    - 但 tree `step17 bonus/row0` 已偏成 `31235`
    - baseline `step20 sample` 则应为 `97145`
- 这组新证据的直接含义：
  - 当前剩余问题已经不再像“later non-row0 attention 又开始漂移”
  - 更像是“一次多 token accept 之后，next-root replay / async input_batch state 没有正确 materialize”
  - 所以下一轮优先级应改成：
    - 先抓 `tree step16 -> step17` 的 first-pass/state 联合证据
    - 没有反证前，不再把注意力重新放回 attention math
- 接续验证后的新反证：
  - 当前代码的定向 `tree target trace` 已确认，
    在问题点对应的 target batch 里：
    - row0 `input_id = 99489`
    - row0 `position = 79`
    - row0 `slot_mapping = 207`
  - 因而“next-root 的 token / position / slot 本身喂错”并不是当前主问题
- 新的更窄结论：
  - tree `step16` 的：
    - `bonus` 行与 baseline `step18` 完全一致
    - `row_label=[1]` 与 baseline `step19` 也完全一致
  - 但进入下一轮后，
    以 `99489` 为 current token 的 row0 bonus 直接偏成 `31235`
  - 这说明 first-pass 至少把“表面输入”重建对了，
    剩余问题更可能是 promoted terminal row 的 canonical KV / slot 收束没有完成
- 当前最强新怀疑点：
  - 当 `terminal_row_indices = [2]`，也就是接受的不是第一个 depth-1 row 时，
    accepted row 的物理 speculative slot 需要压回连续前缀；
    若这一步缺失，就会出现：
    - 下一轮 root `input_id / position` 看起来正确
    - 但 row0 bonus 依旧与 baseline 分叉
  - 进一步的代码级支撑是：
    - `vllm_ascend/spec_decode/tree_first_pass.py::build_tree_first_pass_inputs()`
      当前对 promoted row 的
      `positions / hidden_states / slot_mapping`
      都只是 `index_select(terminal_row_indices)` 后直接复用
    - 代码里没有显式的 accepted-path canonical re-materialize / repack
    - 因而当前最值得验证的不是“token 有没有喂错”，
      而是“full-prefix 视图有没有在 promoted sibling row -> next root 之间偷偷断掉”
  - 这条怀疑点现已被进一步证实到更细粒度：
    - baseline:
      `/tmp/baseline_prefix_trace_late_v1.jsonl`
    - tree:
      `/tmp/tree_prefix_trace_late_v1.jsonl`
    - 对齐点：
      - baseline `step19`
      - tree `step16`
      - `current_input_id = 99489`
      - `seq_len = 80`
      - `current_slot = 207`
    - 结果：
      - `current_input_id / current_input_position / current_slot` 全一致
      - `current_q / current_k / current_v` 全一致
      - 但 full-prefix 中只有最后一个 prefix slot 出现分叉：
        - `prefix slot = 206`
        - `position = 78`
        - `prefix_tail_k_heads.max_diff = 26.0`
        - `prefix_tail_v_heads.max_diff = 0.322265625`
      - 同时当前轮 `current_output_heads.max_diff = 0.023468017578125`
    - 含义：
      - continuity 的断点不是当前 root slot `207`
      - 而是多 token accept 中第一个 accepted draft token
        没有被 materialize 到 canonical prefix slot `206`
      - 这与“slot/position 表面看起来都对”并不矛盾，
        因为坏的是 last-prefix cache entry，不是 current row 字段

## Pending Verification

- 接续会话后的最小待验证项：
  - 重跑长文本 32B eager tree-only case
  - 同时打开：
    - `VLLM_ASCEND_TREE_FIRST_PASS_TRACE_PATH`
    - `VLLM_ASCEND_TREE_STATE_TRACE_PATH`
    - `VLLM_ASCEND_SAMPLE_TRACE_PATH`
  - 目标不是重新证明 sample 分叉，而是核对 `tree step16 -> step17`：
    - `accepted_paths` 是否与 `terminal_row_indices` 一致
    - `rebuilt_token_ids / rebuilt_positions / rebuilt_seq_lens` 是否正确承接 `[20412, 99489]`
    - `prev_sampled_token_ids / spec_token_ids / token_ids_cpu_*tail` 是否把这次双 token 接受正确写回
  - 新增更窄的下一步：
    - 上面的 full-prefix 对账已经完成，结论是：
      continuity 确实断在 `prefix slot 206 / position 78`
    - 所以下一步从“继续找证据”切换为“修 accepted-path KV materialization”：
      - 把 accepted draft tokens 从 speculative physical slots
        拷到 canonical contiguous prefix slots
      - accepted token 的 source row 不能直接拿 `terminal_row_indices` 复用，
        必须逐个 accepted runtime node 回推到当前 target execution row
      - destination slot 不能继续沿用 speculative row 的物理 slot，
        必须由 token 自身 `position` 配合 block table 重算
      - bonus token 不参与 prefix materialize，
        只保留给 next-root current token
      - 然后再复跑 `current_input_id = 99489` 这一轮，
        检查 `slot206` 是否恢复与 baseline 一致
- accepted-path KV materialization `v1` 已落代码并做了真实 32B eager 复跑：
  - 结果文件：
    `/tmp/qwen32_tree_eager_quick_check_noasync.json`
  - 结果仍然是：
    - `first_diff_index = 20`
    - diff token 仍是 `baseline=97145` vs `tree=31235`
  - 这条反证说明：
    - 仅凭当前这版 copy 逻辑，还没有真正修到运行时生效点
    - 下一步需要先验证 helper 自身：
      - 是否在问题步真的执行
      - 是否真的生成了 `source_slot=207 -> destination_slot=206` 这种 copy plan
      - copy 之后是否被后续链路覆盖
- 新增修复尝试与效果：
  - 已把 non-row0 tree exact path 改成：
    - 对每个 speculative row 单独抽取可见的 compact KV
    - 直接调用 Ascend `npu_fused_infer_attention_score`
    - row0 继续沿用 decode override
    - 原 Python dense softmax exact path 只在 vendor 调用异常时兜底
  - 长文本主复现最新结果：
    - baseline token count = `108`
    - tree token count = `112`
    - `first_diff_index = 20`
  - 对比旧结果：
    - 旧版 `first_diff_index = 11`
    - 现在前 20 个 token 与 baseline 全部一致
  - 结论：
    - “non-row0 attention 数值路径不够贴近 baseline”这一判断被正向验证
    - 当前 vendor rowwise fallback 不是小修小补，而是显著后移了首个分叉点
    - 但它还没有完全消除 later-stage 偏差
- 新增 sample-trace 定位结果，表明 root cause 已转移：
  - baseline sample trace:
    `/tmp/baseline_sample_after_vendor_v1.jsonl`
  - tree sample trace:
    `/tmp/tree_sample_after_vendor_v1.jsonl`
  - 关键对齐点：
    - baseline `step18` sampled `20412`
    - baseline `step19` sampled `99489`
    - baseline `step20` sampled `97145`
    - tree `step16` sampled `[20412, 99489]`
    - tree `step17` sampled `[31235]`
  - 更重要的是 step 内 row 对账：
    - tree `step16 bonus`
      与 baseline `step18 sample`
      `hidden_head/top_logits` 完全一致
    - tree `step16 row_label=[1]`
      与 baseline `step19 sample`
      也完全一致
    - 但 tree `step17 bonus/row0`
      与 baseline `step20 sample`
      已明显分叉：
      - baseline top-1 = `97145`
      - tree top-1 = `31235`
  - 这意味着：
    - 当前 remaining 问题已经不再主要来自 non-row0 attention
    - non-row0 attention 修复后，新的首个问题点落在：
      “一次多 token 接受之后的 next-root / bonus root 重建”
    - 当前最强怀疑面重新收缩为：
      - first-pass replay
      - async input_batch state materialization
      - 或 multi-token accept 后的 CPU/GPU seq-len / token_ids 视图错位
- 已被显著降权的怀疑点：
  - branching tree target batch 在量化线性 / 通信路径上的 batch-dependent 数值漂移
  - 直接原因是 row0 isolated rerun 已经在 16 个 step 上与 batched row0 完全一致
- 新增一条更可执行的分流假设，状态为“待验证”：
- row0 isolated full-forward 分流本身已经完成验证：
  - isolated row0 与 batched row0 基本一致，而且当前是“逐元素完全一致”
  - 因而后续排查重点应转向：
    - baseline vs tree 的历史状态差异
    - row 选择/消费错位
    - state reconstruction
    - 或 sample trace 观测粒度之外的上下文差异
- `step 9` 是下一轮重点：
  - prefix 仍与 baseline 对齐到 `29490`
  - 但 tree target 在这一轮给出 `100210`
  - baseline 在同一前缀下给出 `115995`
- 为了验证上面的主怀疑点，下一条最直接的证据应该是：
  - baseline decode-only 在同一步对同一层真正读取到的 `prefix_tail_k/v`
  - 与 tree row0 在同一逻辑前缀下读取到的 `prefix_tail_k/v`
  - 如果两者已经不同，说明问题在 sample 之前的历史状态；
  - 如果两者一致，再继续看更深层 hidden/residual 漂移
  - 说明 target 侧已经在“相同已输出前缀”下算出了不同结果
- 新增执行策略，状态为“待验证”：
  - 不再分别扩展 `tree_exact_debug` 和临时脚本，
  - 而是在 `attention_v1.py` 内补统一的 `prefix-cache trace` 导出，
  - 让 baseline decode-only 与 tree row0 共用同一字段集合：
    - `current_input_ids`
    - `q_len / seq_len / prefix_len`
    - `current_slots`
    - `prefix_tail_slots`
    - `prefix_tail_k/v_heads`
    - `prefix_last_k/v_head`
    - `current_k/v_heads`
  - 这样可以直接按“相同可见前缀”做程序化对账，而不再依赖手工比对不同 trace
- 新增已确认事实：
  - baseline 32B eager 首轮 prefix-trace 实跑已经成功生成正常文本与 token 序列，
    但 `/tmp/baseline_prefix_trace.jsonl` 为空
  - 原因不是 trace 条件没对齐 prefix，而是代码路径判断：
    - `using_paged_attention(runtime_shape, vllm_config)` 在 `speculative_config is None` 且 `cudagraph_mode != FULL_DECODE_ONLY` 时返回 `False`
    - 当前 baseline eager 配置满足这一条件
    - 因而 baseline decode 走的是 `forward_fused_infer_attention`
  - 结论：
    - baseline prefix-cache trace 必须补到 fused decode 路径
    - 之前只挂 `forward_paged_attention` 的版本不会命中 baseline
- 当前最关键的新证据：
  - baseline/tree prefix trace 改为按 `seq_len` 对齐后，不再看到“历史 prefix 先漂移”
  - 真实结果是：
    - `seq_len=61..71`：
      - `prefix_last_k/v` 完全一致
      - `current_k/v` 也完全一致
    - `seq_len=72`：
      - `prefix_last_k/v` 仍完全一致
      - `current_k/v` 首次分叉
    - `seq_len=73` 之后：
      - 上一轮已分叉的 `current_k/v` 开始变成新的 `prefix_last_k/v`，所以 prefix 也随之分叉
  - 含义：
    - 当前主根因不在“更早的 prefix/cache 历史已错”
    - 至少在 `layer0` 上，问题第一次出现于当前 token 本轮前向本身
    - 更像是当前 token 的 `input_id / position / RoPE / row0 query 构造` 之一在 tree 侧已不同
  - 上述结论现在需要加一个限定：
    - 这条“历史 prefix 先漂移已被排除”只对 `layer0` 成立
    - 到了 `layer1`，`slot 196` 的历史 prefix K/V 已经在 `seq_len=70` 暴露出微小差异
    - 因而下一步不该再泛化地说“prefix 历史没问题”，而应写成：
      - `layer0` 历史 prefix 没问题
      - `layer1` 历史 prefix 已开始轻微漂移
- 新增 instrumentation 事实，状态为“待验证”：
  - 第二轮为了补 `current_input_ids / current_input_positions`，
    已把它们写入 `set_ascend_forward_context()`
  - 但 trace 结果仍然是 `null`
  - 代码核对发现：
    - `_EXTRA_CTX` 是一个带白名单的代理
    - `input_ids / positions` 当前不在 `extra_attrs` 中
    - `getattr(_EXTRA_CTX, "input_ids", None)` 会因为代理拒绝访问而返回默认值 `None`
  - 若该判断成立，则下一轮只需补代理白名单，不涉及功能逻辑
- 代理白名单修正后的新结果：
  - baseline/tree 在首次真正分叉前一轮的输入侧已经可以直接对账
  - `seq_len=70`：
    - `current_input_ids = [97145]`
    - `current_input_positions = [69]`
    - `current_slots = [197]`
    - baseline/tree 完全一致
  - `seq_len=71`：
    - `current_input_ids = [29490]`
    - `current_input_positions = [70]`
    - `current_slots = [198]`
    - baseline/tree 完全一致
  - `seq_len=72`：
    - baseline `current_input_ids = [115995]`
    - tree `current_input_ids = [100210]`
    - `current_input_positions = [71]`
    - `current_slots = [199]`
  - 含义：
    - `seq_len=72` 的 current K/V 分叉只是“已采样 token 不同”后的结果
    - 真正的根因仍应回到前一轮 `seq_len=71`
    - 由于 `seq_len=71` 时 token / position / slot / prefix K/V / current K/V 都一致，
      下一轮最值得看的就是 layer0 attention output
- v5 trace 的进一步精确对账已经完成：
  - 对齐点：
    - baseline `step 10`
    - tree `step 9`
    - 共同 `seq_len = 71`
  - `requests[0]` 下的真实观测值显示：
    - `current_input_ids = [29490]`
    - `current_input_positions = [70]`
    - `current_slots = [198]`
    - baseline/tree 完全一致
  - 并且 `layer0` 的下列观测全部逐元素一致：
    - `prefix_last_k_head`
    - `prefix_last_v_head`
    - `current_q_heads`
    - `current_k_heads`
    - `current_v_heads`
    - `current_output_heads`
  - 其中 `current_output_heads` 是整头 `128` 维对账，
    `max_diff = 0.0`
  - 结论：
    - `layer0 attention` 已可直接排除
    - 当前首次可见 token 分叉虽然发生在下一轮，
      但根因必须位于：
      - `layer0 attention` 之后的同层残差 / `o_proj` / MLP / block output
      - 或更深层 block
      - 或 block 之后到 sampler 之前的 hidden/logits 链路
- 针对上面这组剩余可能性，当前最经济的下一步不是立刻加新的 block hook，
  而是先把现有 `prefix-cache trace` 的 layer filter 切到更深层 attention：
  - 若 `layer1 self_attn.attn` 的 `current_q_heads` 已分叉，
    就能把问题直接钉在 `layer0 attention output` 之后、`layer1 attention` 之前
  - 若 `layer1` 仍一致，再继续看更深层
  - 这样能先用现有 instrumentation 缩范围，再决定是否需要对
    `o_proj / post_attention_layernorm / mlp / block output` 额外挂钩
- v5 人工复盘过程中还确认了一点 instrumentation 细节：
  - `prefix-cache trace` 的观测字段并不在顶层，而是在 `requests[0]` 下
  - 首轮把字段读成 `null` 只是日志解析层级拿错，
    不是 trace 本身没落值
- 为了支持后续 `layerN` trace，我把
  - `/root/tmp_baseline_prefix_trace.py`
  - `/root/tmp_tree_only_step11.py`
  都补成了自带 CANN/ATB bootstrap 的版本
- 但首次用它们重跑 `layer1` deeper-layer trace 时，baseline/tree 都没有进入模型对账阶段，
  而是在 worker 初始化时报：
  - `pool INTERNAL ASSERT FAILED ... ParallelOpenMP.cpp:64 Invalid thread pool`
- 这条错误当前只能说明：
  - 新脚本 bootstrap 还没有完全对齐之前稳定脚本的 worker 启动环境
  - 暂时不能拿这轮失败去推断任何模型语义结论
- 之后对齐旧稳定脚本的 worker 环境项后，
  `Invalid thread pool` 已不再复现，说明那一轮确实是启动环境缺项导致
- 但第二轮 deeper-layer trace 又暴露出新的非模型问题：
  - baseline 和 tree 两个 4 卡任务同时跑时，
    HCCL 初始化发生端口冲突
  - 典型报错：
    - `Failed to enable listening for the NPU network adapter socket`
    - `The IP address 192.46.2.199 and port 16666 have already been bound`
  - 结论：
    - deeper-layer trace 后续必须串行跑 baseline/tree
    - 当前这一轮仍然没有产出可用于 `layer1` 语义判断的有效模型证据
- 串行重跑后的 `layer1` 边界探针已经得到有效结论：
  - 文件：
    - `/tmp/baseline_prefix_trace_v6.jsonl`
    - `/tmp/tree_prefix_trace_v6.jsonl`
  - 对齐点：
    - baseline `step 10`
    - tree `step 9`
    - `seq_len = 71`
    - `layer = model.layers.1.self_attn.attn`
  - 结果：
    - `current_input_ids / current_input_positions / current_slots` 完全一致
    - `prefix_last_k/v` 完全一致
    - `current_q_heads / current_k_heads / current_v_heads` 完全一致
    - 但 `current_output_heads` 已出现极小非零差异：
      - `max_diff = 0.00018310546875`
      - `mean_diff = 5.607865750789642e-05`
  - 含义：
    - `layer1 attention` 不是像 `layer0` 那样逐元素完全一致
    - 漂移并不是在 `q/k/v` 构造阶段出现，而是在 attention output 上开始出现微小数值偏差
    - 当前更像是：
      - `layer0` 之后的链路先引入了细小扰动
      - 到 `layer1 attention output` 才被观测到
      - 后续再由更深层放大成最终 logits/token 分叉
- 基于同一组 `v6` 文件，又补出一条更强的“历史 prefix 已轻微漂移”证据：
  - 对齐点：
    - baseline `step 9`
    - tree `step 8`
    - `seq_len = 70`
    - `layer = model.layers.1.self_attn.attn`
  - 结果：
    - `current_input_ids / current_input_positions / current_slots` 仍完全一致
    - `current_q_heads / current_k_heads / current_v_heads` 仍完全一致
    - 但 `prefix_tail_slots = [193, 194, 195, 196]` 中，
      只有最后一个 `slot 196` 的历史 K/V 已出现差异：
      - `prefix_tail_k_heads.max_diff = 0.125`
      - `prefix_tail_v_heads.max_diff = 0.0018515586853027344`
    - 同一轮的 `current_output_heads.max_diff = 6.103515625e-05`
  - 含义：
    - `seq_len=71` 看到的 `layer1 attention output` 微小 drift，
      很可能不是“同一组当前 q/k/v 当场算歪了”
    - 更像是更早一轮生成出的历史 token，
      在 `layer1` 就已经留下了轻微 K/V 漂移
    - 该漂移在 `seq_len=71` 时只是随 prefix 窗口右移，从 tail index 3 平移到 tail index 2
- 结合已有 token/slot 对账，这个 `slot 196` 不是随机历史 token，而是：
  - 位置 `68`
  - token `3837`
  - 且正是先前从 tree accepted-path 进入前缀的那个 token
  - 这使得当前最具体的怀疑点变成：
    - row0 isolated exact 对账只证明了 row0 本身没被 sibling 污染
    - 但它没有证明“被 accept 的非 row0 token 写入 cache 的高层表示”和 baseline 完全一致
    - 下一步应直接比较：
      - baseline 中该 token `3837` 的正常 decode 表示
      - tree 中对应 accepted row 的表示
- 这条比较已经用现有日志直接做出来了：
  - 数据来源：
    - baseline `layer0`: `/tmp/baseline_prefix_trace_v5.jsonl`
    - baseline `layer1`: `/tmp/baseline_prefix_trace_v6.jsonl`
    - tree accepted-row exact debug:
      `/tmp/tree_only_exact_debug_step11.jsonl`
  - 先用 `seq_len=68` 的 baseline row0 去匹配 tree 中 bonus row `slot 195`，
    已确认 `pid = 725850` 对应的正是同一 head shard
  - 在这个已对齐 shard 上，对 accepted row `slot 196 / token 3837` 的比较结果是：
    - `layer0`：
      - tree row1 vs baseline `seq_len=69` 的 `current_k_heads.max_diff = 0.0`
      - `current_v_heads.max_diff = 0.0`
    - `layer1`：
      - tree row1 vs baseline `seq_len=69` 的 `current_k_heads.max_diff = 0.01953125`
      - `current_v_heads.max_diff = 0.0015869140625`
  - 结论：
    - 这个 accepted token `3837` 在进入高层 prefix 历史之前，
      `layer0` 仍与 baseline 完全一致
    - 它的最早可观测分叉点就发生在同一个 token 的 `layer0 -> layer1` 之间
    - 这进一步把 root cause 收缩到：
      - `layer0 attention output` 之后
      - `layer0 o_proj / residual / mlp / block output`
      - 或 `layer1` 入口前的状态变换
- 新增更强证据：
  - `seq_len=71` 时，baseline/tree 在 `layer0` 上的
    - `current_input_id`
    - `current_input_position`
    - `current_slot`
    完全一致
  - 另外在当前 trace 口径下，`current_q / current_k / current_v`
    的“第一头前若干维切片”也一致
  - 含义：
    - query 构造本身至少在当前已观测切片上还没分叉
    - 若这一轮最终还是导致下一 token 分叉，最优先要查的就是同输入下的 attention output 是否已经不同
- 新增 `v5` prefix-tail 对账结果，进一步排除了 recent generated prefix：
  - 数据来源：
    - tree exact debug:
      `/tmp/tree_only_exact_debug_step11_v5.jsonl`
    - baseline `layer0`:
      `/tmp/baseline_prefix_trace_v5.jsonl`
    - baseline `layer1`:
      `/tmp/baseline_prefix_trace_v6.jsonl`
  - 对齐点：
    - `tp_rank = 0`
    - tree `seq_len = 73`
    - `current_slots = [195, 196, 197, 198, 199, 200]`
    - accepted row `slot 196 / token 3837`
      会消费到的 recent generated prefix 为
      `slots 188..195`（位置 `60..67`）
  - 结果：
    - `layer0`：
      - `prefix slots 188..194` 的 `K/V max_diff` 全部为 `0.0`
      - `current slot 195` 的 `q/k/v/output max_diff` 全部为 `0.0`
      - `current slot 196` 只有 `output max_diff = 6.103515625e-05`
    - `layer1`：
      - `prefix slots 188..194` 的 `K/V max_diff` 也全部为 `0.0`
      - `current slot 195` 的 `q/k/v/output max_diff` 仍全部为 `0.0`
      - `current slot 196` 才出现：
        - `q.max_diff = 0.04296875`
        - `k.max_diff = 0.125`
        - `v.max_diff = 0.0018515586853027344`
        - `output.max_diff = 0.00048828125`
  - 结论：
    - recent generated prefix `positions 60..67` 目前没有证据支持已漂移
    - 即便把 `slot 195 / position 67` 作为 accepted row 的直接前驱一起检查，也仍然是 exact
    - 当前最小闭环已经收缩为：
      accepted token `3837` 自身在 `layer0 -> layer1` 之间开始偏离 baseline
    - 下一步优先级应转到：
      - `layer0 attention output` 之后的 block 内链路
      - 或一次性检查更老 prompt prefix 是否早已不同
- sampler / verifier 当前继续保持为次级怀疑点：
  - 因为 sample trace 已经直接看到 row0 hidden/logits 在采样前就不同
  - 除非后续出现“prefix/context 实际不同只是 output_prefix 日志没看出来”的反证，否则不应继续把主精力放在 rejection/sampling 规则
- `AscendRejectionSampler.forward()` 的 tree 分支与标准 spec decode 分支仍有行为差异，后面仍需核对：
  - tree 分支直接对 `target_logits.argmax(dim=-1)` 做 verifier
  - 没有走标准分支里的 `bonus_sampler_output = self.sampler(...)`
  - 也没有走 `apply_sampling_constraints(...)`
  但在当前证据下，这条线暂时落后于 non-attention batch-drift 排查。

## Resume 2026-06-29 Late

### Pending Validation

- 当前恢复后的唯一待验证分流已经收窄为：
  - 对齐样本：
    - baseline `index = 73`
    - tree `index = 65 row [0]`
    - 共享 `token / position / slot = 9370 / 133 / 261`
  - 现有 block trace 已知：
    - `layer0` 全链路 `0.0`
    - `layer1 attn_output.max_diff = 0.0068359375`
  - 但仍未最终回答：
    - `layer1` 的 `current_q/k/v` 是否已经先分叉
    - 还是 `current_q/k/v` 仍一致，而 `current_output` 才首次出现差异
- 本轮先复用现有日志，不新增功能判断：
  - baseline:
    - `/tmp/baseline_prefix_trace_l1_v2.jsonl`
  - tree:
    - `/tmp/tree_exact_layer1_late_v1.jsonl`
- 只有在上述日志无法唯一回答输入面 vs 输出面分流时，
  才允许继续起新的 4 卡 trace。

### Confirmed

- 恢复后的 `layer1` 分流已经用现有日志直接回答：
  - baseline:
    - `/tmp/baseline_prefix_trace_l1_v2.jsonl`
  - tree:
    - `/tmp/tree_exact_layer1_late_v1.jsonl`
  - 对齐点：
    - baseline `step = 73`
      `current_input_ids = [9370]`
      `current_input_positions = [133]`
      `current_slots = [261]`
    - tree `index = 65 row[0]`
      的 `query_pos = 1`
      对应同一逻辑行 `slot = 261`
- 结果：
  - `current_q.max_diff = 0.0`
  - `current_k.max_diff = 0.0`
  - `current_v.max_diff = 0.0`
  - `current_output.max_diff = 0.00013399124145507812`
- 但同一行真正可见的 `layer1` 历史 prefix 已经不是全等：
  - 对齐后的可见 prefix slots 为 `253..260`
  - 其中：
    - `slot 253..255` 的 `K/V` 仍为 `0.0`
    - `slot 256..260` 开始出现 drift
      - `K max diff` 约 `0.03125 .. 0.125`
      - `V max diff` 约 `0.00225830078125 .. 0.00341796875`
- 结论：
  - 当前 residual divergence 不是
    “`layer1 attention impl` 在完全同输入面下算错”
  - 而是：
    - 当前 token `9370` 的 `q/k/v` 仍一致
    - 但它读到的最近几格 `layer1 prefix K/V`
      已经在更早历史里轻微漂移
    - `current_output` 的 tiny diff 是这段已漂历史输入的直接结果
- 因而下一步不应优先回到 `layer1 attention impl`，
  而应先把 `slot 256..260` 追溯回具体历史 token / step。

### Confirmed

- 已用现有 baseline `layer1` trace 把最近漂移 prefix slots
  反查回具体历史 token：
  - `slot 256 -> step 68 -> token 9370 -> position 128`
  - `slot 257 -> step 69 -> token 104978 -> position 129`
  - `slot 258 -> step 70 -> token 102832 -> position 130`
  - `slot 259 -> step 71 -> token 26939 -> position 131`
  - `slot 260 -> step 72 -> token 100390 -> position 132`
- 这让下一步可以直接从“抽象 prefix 漂移”
  收敛到“具体哪一个历史 token 首先把 `layer1` 写偏”。

### Pending Validation

- 继续只用现有 `layer1` 日志逐 token 对齐后，
  当前最早的已观测非零样本已经前移到：
  - `slot 256 -> token 9370 -> position 128`
  - 在它作为 current row 的那一轮就已有：
    - `q_max = 0.04296875`
    - `k_max = 0.125`
    - `v_max = 0.00225830078125`
    - `out_max = 0.000732421875`
- 但这还不是最终结论：
  - 需要继续验证更早的 `slot 253..255`
    在它们自己作为 current row 的那一轮是否仍为 `0.0`
  - 若 `253..255` 仍为 `0.0`，则 `slot 256` 可升级为当前最早漂移 token

### Confirmed

- 对 `slot 253..255` 的 current-row 前验已经完成：
  - `slot 253 -> token 1773`
  - `slot 254 -> token 45181`
  - `slot 255 -> token 102640`
  - 它们在自己那一轮的 `layer1 current_q/k/v/output`
    全部仍为 `0.0`
- 因而当前最早的已证实漂移点正式收敛为：
  - `slot 256`
  - `token 9370`
  - `position 128`
- 新的下一步：
  - 不再继续横向扫更多 prefix slots
  - 直接围绕 `slot 256 / token 9370`
    去查它在 `layer0 -> layer1` block 边界上的首个非零位置

### Pending Validation

- 恢复会话后的唯一待验证判断已经收敛为：
  - `slot 256 / token 9370 / position 128`
    很可能不是在“更晚层的 attention impl”里凭空出错
  - 而是它自己作为 current row 的那一轮，
    在 `layer0` block 路径的某个字段上就先偏离 baseline
- 这条判断目前还不能升级为结论：
  - 现有证据只证明：
    - 它在 `layer1 current_q/k/v/output` 已首次非零
    - 更早的 `slot 253..255` 仍为 `0.0`
  - 还没有证明：
    - `layer0` 的首个非零字段到底是
      `attn_output`
      还是 `post_attn_hidden / residual / mlp_output`
- 下一步只允许复用现有 block trace 先回答这件事，
  不先补新的长跑任务。

### Confirmed

- 现有 block trace 已经直接回答了
  `slot 256 / token 9370 / position 128`
  的 `layer0 -> layer1` 边界问题：
  - baseline:
    - `/tmp/baseline_block_trace_late_v2.jsonl`
    - `index = 68`
  - tree:
    - `/tmp/tree_block_trace_late_v2.jsonl`
    - `index = 61`
    - `qpos = 0`
- 对齐结果：
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
- 结论：
  - `slot 256` 的首个已证实非零字段
    不是 `layer1 current_q`
  - 而是它自己那一轮的 `layer0 attn_output`
  - `layer1` 看到的差异只是这个 `layer0` 漂移的后续传导

### Pending Validation

- 由于此前已知：
  - `slot 256` 那一轮的 `layer0 current_q/k/v` 是 exact match
- 现在又确认：
  - `layer0 attn_output` 已首次非零
- 因而新的最窄待验证问题变成：
  - `layer0` 在这一步消费到的 prefix K/V 是否已经先漂
  - 还是 `layer0` attention exact path 在同输入面下仍有 residual 数值差异
- 现有 `/tmp` 证据还缺一块：
  - 没有覆盖 `slot 256` 的 `layer0 prefix/exact` 晚步数日志
- 因而下一步需要最小化重跑一轮定向 trace：
  - baseline:
    `model.layers.0.self_attn.attn` 的 `decode_fused` prefix trace
  - tree:
    `model.layers.0.self_attn.attn` 的 exact debug
  - 目的只是区分：
    - 更老 prefix K/V 已漂
    - 还是 exact path 本身仍有差异

### Confirmed

- 上述 `layer0` 定向 trace 已经补完并回答了这个分流：
  - baseline:
    - `/tmp/layer0_prefix_compare_v1.jsonl`
    - `step = 68`
  - tree exact:
    - `/tmp/tree_exact_layer0_256_v1.jsonl`
    - `index = 61 row0`
- 对齐点仍然是：
  - `slot 256`
  - `token 9370`
  - `position 128`
  - `prefix_len 128`
- 结果：
  - `prefix_tail_slots 128..255` 完全对齐
  - `prefix_tail_k/v` 全部 `0.0`
  - `current_q = 0.0`
  - `current_k = 0.0`
  - `current_v = 0.0`
  - `current_output = 6.103515625e-05`
  - tree 内部 `direct current_k/v` vs `gathered current_k/v` 也仍是 `0.0`
- 结论：
  - `slot 256` 的 remaining drift
    不是更老 `layer0 prefix K/V` 先漂
  - 它已经收缩成：
    - 可见输入完全一致
    - 但 `layer0 exact attention` 输出仍有 residual 差异

### Pending Validation

- `tree_exact_layer0_256_v1.jsonl` 同时还给出：
  - `row0_output_max_diff_vs_isolated = 0.0001220703125`
- 这让下一步收缩为一个更窄的问题：
  - multi-row exact attention 是否就是直接根因
  - 还是 isolated exact / vendor-rowwise 路径本身也已和 baseline 有差异

### Confirmed

- 这条分流也已经被现有新日志回答：
  - baseline:
    - `/tmp/layer0_prefix_compare_v1.jsonl`
    - `step = 68`
    - `trace_source = decode_fused`
  - tree root decode:
    - `/tmp/layer0_prefix_compare_v1.jsonl`
    - `step = 61`
    - `trace_source = tree_row0_decode`
  - tree exact:
    - `/tmp/tree_exact_layer0_256_v1.jsonl`
    - `index = 61 row0`
- 三向比较结果：
  - `baseline vs tree_row0_decode current_output = 6.103515625e-05`
  - `baseline vs tree exact row0 current_output = 6.103515625e-05`
  - `tree_row0_decode vs tree exact row0 current_output = 0.0`
  - 并且 `baseline vs tree_row0_decode` 的
    `prefix_tail_k/v` 与 `current_q/k/v` 全部仍为 `0.0`
- 结论：
  - 这不是 multi-row exact attention 特有问题
  - tree 侧单行 root decode 路径
    在同输入面下就已经与 baseline 有稳定微差

### Pending Validation

- 因而当前最窄待验证点已经改成：
  - `decode_fused` 与 `tree_row0_decode`
    的 metadata / kernel-call 参数是否仍有可观测差异
  - 尤其是：
    - `_build_tree_root_decode_metadata()`
    - `actual_seq_lengths_q/kv`
    - `attn_state`
    - `block_tables / slot_mapping` 子视图
- 继续读代码后，当前最强怀疑点已经从“metadata 可能不同”
  收缩到更直接的一条：
  - baseline 当前步走的是 `decode_fused`
  - tree row0 在
    `forward_tree_exact_attention()`
    里却被硬编码成 `forward_paged_attention()`
- 因而当前最值得先验证的改动是：
  - 让 tree row0 复用 baseline decode 的同一 dispatch，
    看最早那条 `6.103515625e-05` 是否随之消失

### Confirmed

- 上述 dispatch 改动的第一轮复跑没有跑通 tree 用例，但错误信息本身已经给出新的确证：
  - tree row0 已经真正进入 `forward_fused_infer_attention()`
  - 当前失败点不是分支没命中
  - 而是 `row0_metadata.attn_mask` 为空，导致 Ascend fused kernel 报错：
    - `When attnMask is not provided, sparseMode must be 0`
- 结论：
  - 当前方向是对的
  - 但 tree row0 fused 路径还缺 baseline decode 所需的 metadata 补齐

### Pending Validation

- 下一步先补最小必要 metadata：
  - `row0_metadata.attn_mask`
- 若补齐后 tree row0 fused 能成功运行，
  再回到 `slot 256` 的 `current_output` 对账

### Confirmed

- 2026-06-30 新一轮 `batched fast path` profile v2 已确认：
  - 上一轮新增的 `non-root visible-row template cache`
    已带来真实收益，不是噪声优化
  - 对比：
    - `/tmp/qwen32_tree_batched_profile.json`
      - `generate_seconds = 27.01`
    - `/tmp/qwen32_tree_batched_profile_v2.json`
      - `generate_seconds = 23.45`
  - 对应 profile：
    - `/tmp/tree_fast_profile.jsonl`
    - `/tmp/tree_fast_profile_v2.jsonl`
  - 关键均值变化：
    - `non_root_ms: 1.494 -> 0.809`
    - `nr_plan_ms: 0.660 -> 0.028`
    - `nr_pack_ms: 0.637 -> 0.612`
    - `nr_fia_ms: 0.075 -> 0.074`
    - `gather_ms: 0.712 -> 0.613`
    - `total_ms: 3.021 -> 2.234`
  - 结论：
    - 当前 `plan` 已基本退出主瓶颈
    - 剩余主要热点已经收敛到：
      - `gather_paged_kv_to_ragged()`
      - non-root packed KV 中的 prefix duplication / pack
      - row0 单独 decode override

### Pending Validation

- 基于上述 v2 profile，下一轮结构优化方向已升级：
  - 不再停留在：
    - `gather full prefix -> 复制 prefix 到每个 speculative row -> batched FIA`
  - 新主方案是：
    - `prefix-only FIA`：
      直接基于 paged KV cache + repeated block_table
      对每个 non-root row 计算 prefix attention 与 `softmax_lse`
    - `current-only FIA`：
      只对树上当前 token 的可见子集做小型 dense FIA
    - 最后按 `lse` 做精确合并
  - 预期收益：
    - 去掉 `gather_paged_kv_to_ragged()`
    - 去掉 per-row prefix materialization / duplication
    - 把 non-root pack 缩小到只处理当前树 token
  - 当前主要风险：
    - 要确认 Ascend `TND + block_table + softmax_lse_flag=True`
      在 decode-like 单 query batched 场景下稳定可用
    - 以及 `prefix/current` 两段 attention 的 `lse` 合并
      与现有 batched/reference 路径完全一致

### Confirmed

- `prefix-only FIA + current-only FIA + lse merge`
  这条 split 路径已经在工程上跑通，但目前还不能取代
  正确的 legacy batched 路径：
  - tree-only 32B smoke：
    - `generate_seconds ≈ 16.8s`
    - 明显快于当前正确路径的 `23.45s`
  - 但 baseline/tree 对比仍然在很早处开始分叉：
    - `first_diff_index = 11`
    - tree 最终 `104 token`
    - baseline `108 token`
  - 新增 probe：
    - `/tmp/tree_split_compare_probe.jsonl`
  - probe 结论：
    - `split_vs_legacy_max_diff`
      平均约 `0.0031`
      最大约 `0.0078`
    - `split_vs_torch_merge_max_diff = 0.0`
    - `output_indices_match = true`
  - 含义：
    - split 路径与正确 legacy 路径的 non-root attention 输出
      已经非常接近
    - 但仍存在稳定的非零数值漂移
    - 且该漂移足以在 speculative decode 的接受/拒绝链路中放大，
      最终导致 token 序列提前分叉
    - 同时可以排除：
      - `npu_attention_update` 合并公式错误
      - 或 `torch logsumexp` 合并实现错误
  - 当前最合理判断：
    - 问题更可能来自：
      - 分段 FIA 暴露出来的 kernel 内部数值路径差异
      - `softmax_lse` / partial attention output 的精度损失
      - 而不是 merge 数学本身

### Pending Validation

- 因此 split 路径当前定性改为：
  - `experimental fast path`
  - 可保留继续研究
  - 但暂时不能替代正确主路径
- 接下来的正确优化主线应回到：
  - 继续优化当前正确的 legacy batched 路径
  - 优先削：
    - packed KV 的 prefix copy / pack
    - row0 override 开销
    - gather 路径里不必要的搬运

- 2026-06-30 恢复上下文后重新过了一遍正确主路径代码，已进一步确认
  `legacy batched` 里还有一层很具体的重复搬运：
  - `attention_v1.py::_forward_tree_fast_attention_legacy()`
    先调用 `gather_paged_kv_to_ragged()`
    把 paged KV 变成 request-major ragged dense KV
  - 随后 `tree_attention.py::_run_vendor_tree_attention_batched()`
    又把这些 ragged dense KV
    重新 pack 成 speculative-row 可见的 compact KV
  - 这等价于：
    - `paged KV -> ragged dense KV -> packed visible KV`
  - 而真正送入 FIA 的只有最后这份 `packed visible KV`
- 这说明当前正确路径仍存在一个可直接切掉的中间态：
  - 对 non-root batched rows 来说，
    更合理的实现应当是：
    - 直接从 paged cache 按 slot 收集到
      `packed visible KV`
    - 跳过中间的 ragged dense KV materialization
- 当前判断：
  - 这条改动不会改变 attention 语义，
    因为它只是把“先 gather 再 pack”的两段索引
    合并成“一次按最终可见 token 次序 gather”
  - 若实现正确，
    最直接受益项应当是：
    - `gather_ms`
    - 部分 prefix 搬运
  - 同时它比继续碰 split-path 数值问题更适合作为
    当前正确主线的下一刀

- 上述 `paged KV -> packed visible KV` 改动已经完成真实 32B 复验，
  而且结果是正向的：
  - 结果文件：
    - `/tmp/qwen32_tree_batched_compare_v4.json`
  - profile：
    - `/tmp/tree_fast_profile_v4.jsonl`
  - 对比上一轮正确主路径：
    - 旧：
      - baseline `9.97s`
      - tree `20.04s`
      - `same_token_ids = true`
      - `same_text = true`
    - 新：
      - baseline `10.12s`
      - tree `16.50s`
      - `same_token_ids = true`
      - `same_text = true`
  - 含义：
    - 当前正确主路径在不破坏输出一致性的前提下，
      端到端又下降了约 `3.54s`
      （约 `17.7%`）
- profile 也证明这次收益来源非常直接：
  - 旧 `tree_fast_profile_v3`：
    - `gather_ms ≈ 0.662`
    - `row0_ms ≈ 0.649`
    - `non_root_ms ≈ 0.369`
    - `total_ms ≈ 1.837`
  - 新 `tree_fast_profile_v4`：
    - `gather_ms = 0.0`
    - `row0_ms ≈ 0.799`
    - `non_root_ms ≈ 0.450`
    - `total_ms ≈ 1.336`
  - 结论：
    - `gather` 中间态已被完全拿掉
    - 当前 tree attention 主路径的平均层耗时
      又降了约 `27.3%`
- 但新 profile 同时暴露了新的次级热点迁移：
  - 旧 `non_root_stats.plan_ms ≈ 0.029`
  - 新 `non_root_stats.plan_ms ≈ 0.157`
  - 这是因为：
    - 之前 `gather` 路径隐含提供了 request-major dense token 索引空间
    - 现在 direct paged path 每层都要重新从
      `block_tables + seq_lens + tree_mask`
      生成 visible `slot` 计划
- 当前最合理的下一步因此已更新为：
  - 不回退 direct paged path
  - 而是把这份 paged visible plan
    （`output_indices / packed_slots / actual_seq_lengths_kv`）
    缓存到本轮 `attn_metadata` / runtime 上，
    让同一 decode step 的所有层直接复用

- 上述 step 级 visible-plan 缓存也已经完成真实 32B 复验：
  - 结果文件：
    - `/tmp/qwen32_tree_batched_compare_v5.json`
  - profile：
    - `/tmp/tree_fast_profile_v5.jsonl`
  - 结果：
    - baseline `10.05s`
    - tree `14.96s`
    - `same_token_ids = true`
    - `same_text = true`
- 这说明第二轮缓存优化同样没有破坏正确性，
  并且继续带来了新的端到端收益：
  - 相比 `v4`：
    - tree `16.50s -> 14.96s`
    - 再降约 `1.54s`
  - 相比 `v3`：
    - tree `20.04s -> 14.96s`
    - 总降约 `5.08s`
    - 约 `25.4%`
- profile 也和预期一致：
  - `gather_ms` 继续保持 `0.0`
  - `non_root_stats.plan_ms`
    从 `v4` 的约 `0.157ms`
    降到 `v5` 的约 `0.066ms`
  - `non_root_stats.pack_ms`
    进一步降到约 `0.0195ms`
  - `non_root_ms`
    从 `0.450ms`
    进一步降到约 `0.136ms`
  - `total_ms`
    从 `1.336ms`
    进一步降到约 `1.010ms`
- 当前由此得到的新结论是：
  - target 侧 correct tree attention 主路径，
    经过这两轮结构优化后已经明显变轻
  - 现在 remaining 的端到端差距，
    不再主要来自 non-root tree attention
  - 因为：
    - non-root 平均耗时已压到 `0.136ms`
    - 同时 acceptance 仍只有约 `2.3% ~ 2.5%`
    - `drafted throughput` 远高于 `accepted throughput`
  - 因而若还要继续拿大收益，
    更可能需要：
    - 修通更激进但数值仍精确的 exact fast kernel
      （例如解决 split-path 漂移）
    - 或转向 proposer / draft / acceptance 路线

- 2026-06-30 又继续把 visible-plan owner 从 attention 层内
  上提到了 `model_runner_v1` 的 patch 阶段：
  - 每个 tree step 只构造一次 shared `TreePagedPlan`
  - 同一个 plan 对象直接挂到所有 layer metadata 上
  - attention 层只消费，不再默认做 owner
- 这轮的真实结果：
  - compare：
    - `/tmp/qwen32_tree_batched_compare_v6.json`
  - profile：
    - `/tmp/tree_fast_profile_v6.jsonl`
  - 正确性：
    - `same_token_ids = true`
    - `same_text = true`
  - 性能：
    - baseline `10.23s`
    - tree `15.02s`
- 这轮端到端没有继续明显下降，
  但 profile 已明确证明结构目标达成：
  - `non_root_stats.plan_ms = 0.0`
  - 说明 per-layer plan build 已被完全移出 attention 热路径
  - 同时：
    - `non_root_ms ≈ 0.137`
    - `total_ms ≈ 1.014`
    与上一轮基本持平
- 当前这给出的更强结论是：
  - target attention 侧现在最主要的 remaining 成本
    已经不是 non-root
  - 而是 row0 / root override：
    - `row0_ms ≈ 0.795`
    - 明显高于 `non_root_ms ≈ 0.137`
  - 所以下一轮若继续在 attention 路线上做“大结构”，
    最合理的主线应转为：
    - 统一 row0 与 non-root，
      做 single-pass all-row exact tree attention
    - 目标是去掉当前单独的 row0 baseline decode override
