# vLLM Speculative Decoding: Upstream v023 to Ascend

## 1. Scope and reading target

This note explains two things together:

1. What "speculative decoding" means in `/root/worktrees/vllm-spec-tree-v023`, especially the parts relevant to `speculative_token_tree`.
2. How `/root/worktrees/vllm-ascend-clean` takes that upstream contract and turns it into an actual Ascend-side implementation.

The focus is the V1 path first, because the real Ascend tree implementation is there. V2 is covered only as status and gap analysis.

This document is intentionally biased toward source-reading efficiency:

- what the real call chain is
- which files are architecture files
- which files are just tests / docs / side details
- where the current limits come from

## 2. First split: upstream contract vs Ascend implementation

The most important thing to understand is that the two trees are not at the same completion stage.

### `/root/worktrees/vllm-spec-tree-v023`

This tree already has a public `speculative_token_tree` contract in code:

- config field exists
- parsing and validation exist
- CLI / `speculative_config` plumbing exists
- proposer base class recognizes tree mode
- built-in upstream proposers still reject runtime tree execution

So upstream v023 is mainly "contract + plumbing + explicit runtime rejection".

### `/root/worktrees/vllm-ascend-clean`

This tree does more than recognize the field:

- it builds a local tree runtime
- it changes draft execution from linear chain semantics to tree semantics
- it patches target-side metadata, positions, and attention masks
- it adds a tree verifier in the rejection sampler path

So Ascend is where tree execution is actually implemented.

If you only remember one sentence, remember this:

> Upstream v023 defines the public interface for `speculative_token_tree`; Ascend adds the real execution model.

## 3. Upstream v023: what speculative decoding consists of

### 3.1 Public config entry

Start from:

- `/root/worktrees/vllm-spec-tree-v023/vllm/config/speculative.py`

This is the public entry point for all speculative decoding methods.

Relevant points:

- `SpeculativeConfig.method` enumerates supported methods.
- `SpeculativeConfig.speculative_token_tree` is now a real field.
- `SpeculativeConfig._parse_speculative_token_tree()` parses the string into a normalized tuple-of-tuples tree path form.
- `parsed_speculative_token_tree` exposes the normalized internal representation.
- `use_speculative_token_tree()` is the feature gate.

The config layer also defines the public semantic rules:

- input type is still `str | None`
- string must be `ast.literal_eval`-parseable
- parsed value must be an iterable of non-empty paths
- duplicate paths are rejected
- parent-closure is required
- if only tree is given, `num_speculative_tokens = len(tree)`
- if both are given, they must match

This logic is not just validation. It is the boundary between user-visible API and all downstream implementations.

### 3.2 Method gating for tree mode

Still in `vllm/config/speculative.py`, `_verify_args()` enforces that `speculative_token_tree` only opens for:

- `eagle`
- `eagle3`
- `mtp`
- `dflash`

The key method is:

- `use_eagle()`

Its actual meaning in this branch is broader than the name suggests:

- `use_eagle()` returns true for `("eagle", "eagle3", "mtp", "dflash")`

That naming is worth noticing because later code sometimes says "eagle path" when it really means the entire draft-model-like proposer family that shares the same proposer base.

### 3.3 Tree config participates in graph identity

Also in `vllm/config/speculative.py`, `compute_hash()` includes:

- `parsed_speculative_token_tree`

That means tree layout is treated as computation-graph-shaping config, not a loose runtime hint.

This matters because any backend that wants to support tree mode must treat the tree shape as part of execution specialization.

## 4. Upstream v023: speculative execution chain

### 4.1 Drafter creation

Main file:

- `/root/worktrees/vllm-spec-tree-v023/vllm/v1/worker/gpu_model_runner.py`

The important method is:

- `_create_speculative_drafter()`

This is the main V1 method dispatch point. It chooses the proposer class based on `speculative_config.method`:

- `ngram` -> `NgramProposer`
- `draft_model` -> `DraftModelProposer`
- `ngram_gpu` -> `NgramProposerGPU`
- `mtp` specializations -> `Gemma4Proposer` / `Step3p5MTPProposer`
- `dflash` -> `DFlashProposer`
- `suffix` -> `SuffixDecodingProposer`
- `eagle/eagle3/mtp/dflash family` -> `EagleProposer`-style base path
- `medusa` -> `MedusaProposer`
- `extract_hidden_states` -> `ExtractHiddenStatesProposer`

So for source reading, `gpu_model_runner.py` is where "which proposer am I actually using?" is answered.

### 4.2 Common proposer base

Main file:

- `/root/worktrees/vllm-spec-tree-v023/vllm/v1/spec_decode/llm_base_proposer.py`

This is the shared proposer substrate for draft-model-like speculative methods.

Important responsibilities:

- stores `speculative_config`
- stores `parsed_speculative_token_tree`
- normal draft-model buffer management
- padded drafter batch logic
- position / hidden-state / slot management
- parallel drafting support
- draft-model forward orchestration

For tree mode, the key upstream fact is:

- constructor reads `parsed_speculative_token_tree`
- then calls `_raise_if_speculative_token_tree_unimplemented()`

And that method unconditionally raises if tree mode is enabled.

So upstream v023 intentionally stops here:

- tree is recognized
- tree is carried into proposer state
- built-in execution is still blocked

That is the clean seam Ascend hooks into.

### 4.3 Public docs and tests already exist upstream

Useful files:

- `/root/worktrees/vllm-spec-tree-v023/docs/features/speculative_decoding/README.md`
- `/root/worktrees/vllm-spec-tree-v023/tests/test_config.py`
- `/root/worktrees/vllm-spec-tree-v023/tests/engine/test_arg_utils.py`
- `/root/worktrees/vllm-spec-tree-v023/tests/v1/spec_decode/test_speculative_token_tree.py`

These prove that upstream v023 is no longer just a design idea. The feature is present as API contract and tested as such.

But the runtime test explicitly tests rejection, not execution.

## 5. What upstream does not do

Upstream v023 does not provide a full generic tree runtime for built-in proposers.

Concretely, it does not provide:

- tree-native draft execution inside the built-in proposer stack
- tree-native target metadata construction
- tree-native target attention mask patching
- tree-native verifier / accepted-path reconstruction for built-in execution

That gap is exactly why Ascend adds local files instead of just reusing upstream behavior.

## 6. Ascend V1: where speculative decoding is wired in

### 6.1 Method dispatch

Start from:

- `/root/worktrees/vllm-ascend-clean/vllm_ascend/spec_decode/__init__.py`

This is the Ascend-side equivalent of upstream drafter dispatch.

It maps methods to Ascend-specific proposer classes:

- `ngram` -> `AscendNgramProposer`
- `ngram_gpu` -> `AscendNgramProposerNPU`
- `suffix` -> `AscendSuffixDecodingProposer`
- `medusa` -> `AscendMedusaProposer`
- `eagle/eagle3/mtp` -> `AscendEagleProposer`
- `dflash` -> `AscendDflashProposer`
- `draft_model` -> `AscendDraftModelProposer`
- `extract_hidden_states` -> `AscendExtractHiddenStatesProposer`

This file answers "which Ascend proposer class is actually instantiated?"

### 6.2 Main model runner integration

Main file:

- `/root/worktrees/vllm-ascend-clean/vllm_ascend/worker/model_runner_v1.py`

This is the real V1 center of gravity on Ascend.

Key responsibilities:

- constructs the drafter with `_get_drafter()`
- installs `AscendRejectionSampler`
- computes speculative metadata
- patches target-side positions and attention metadata for tree mode
- coordinates propose -> target forward -> verify -> next-round draft

Important setup block:

- if `self.speculative_config` exists, last pipeline rank creates `self.drafter`
- then `self.rejection_sampler = AscendRejectionSampler(self.sampler)`
- and `self.rejection_sampler.runner = self`

That last line is important. The rejection sampler is not purely functional in tree mode; it reports accepted tree path terminal rows back into the runner.

## 7. Ascend V1: how tree mode is turned on

### 7.1 The proposer base overrides upstream rejection

Main file:

- `/root/worktrees/vllm-ascend-clean/vllm_ascend/spec_decode/llm_base_proposer.py`

Main class:

- `AscendSpecDecodeBaseProposer`

This class subclasses upstream `SpecDecodeBaseProposer`, but its first critical move is:

- override `_raise_if_speculative_token_tree_unimplemented()` to no-op

This is the first real fork from upstream behavior.

Without this override, upstream base-class construction would reject tree mode before Ascend could do anything.

### 7.2 Reading the upstream tree contract safely

Still in `AscendSpecDecodeBaseProposer`, the method:

- `_get_upstream_parsed_speculative_token_tree()`

does a compatibility-aware read of:

- `use_speculative_token_tree`
- `parsed_speculative_token_tree`

from the upstream `SpeculativeConfig` type.

This matters because Ascend has lived across upstream versions where tree fields existed, changed, or were removed. The code does not blindly assume the field is always present.

### 7.3 Local tree runtime initialization

Key method:

- `_init_speculative_token_tree_runtime()`

What it does:

1. reads normalized upstream tree paths
2. enforces Ascend-specific runtime limits
3. builds a local `SpecTreeRuntime`

Current hard limits come from here:

- method must be `eagle` or `eagle3`
- graph mode is rejected
- only eager mode is allowed

So although upstream config allows tree for `eagle/eagle3/mtp/dflash`, the actual Ascend executor is narrower:

- implemented now: `eagle`, `eagle3`
- not implemented as tree execution now: `mtp`, `dflash`

That difference between "accepted config contract" and "implemented runtime" is one of the main things that confuses readers.

## 8. Ascend tree runtime: the three local files that matter

If you only want the tree-specific implementation, the most important local files are:

- `/root/worktrees/vllm-ascend-clean/vllm_ascend/spec_decode/tree_runtime.py`
- `/root/worktrees/vllm-ascend-clean/vllm_ascend/spec_decode/tree_execution.py`
- `/root/worktrees/vllm-ascend-clean/vllm_ascend/spec_decode/tree_verifier.py`

### 8.1 `tree_runtime.py`: canonical tree representation

`build_spec_tree_runtime()` turns normalized path tuples into:

- `parent_index`
- `depth`
- `children`
- `nodes_by_depth`
- `leaf_indices`

This file is the structural model of the tree.

It does not execute anything. It defines the canonical runtime view that both drafter and verifier must agree on.

### 8.2 `tree_execution.py`: execution order and masks

This file defines tree execution semantics:

- `AscendTreeSpecDecodeMetadata`
- `build_tree_query_position_offsets()`
- `build_tree_query_slot_offsets()`
- `build_tree_execution_runtime_node_indices()`
- `build_tree_attn_mask()`
- `build_tree_target_attn_mask()`
- `build_tree_spec_decode_metadata()`

Conceptually it answers:

- in what order tree nodes are materialized
- how tree depth maps to semantic positions
- how sibling nodes get distinct KV slots
- which tokens may attend to which ancestors
- how request-local tree nodes are flattened into batch tensors

This file is the bridge between abstract tree shape and actual batched tensor layout.

### 8.3 `tree_verifier.py`: accepted-path reconstruction

This file answers the verification question:

given draft token ids and target token ids laid out in tree form, which branch is accepted?

Important methods:

- `select_tree_accept_paths()`
- `pack_tree_output_tokens()`

The verifier walks the tree by depth and chooses the deepest valid matched branch per request.

That is different from normal linear speculative decoding, where verification is a prefix-length problem. Tree mode is a best-valid-branch problem.

## 9. Ascend proposer tree execution: where draft tokens are actually generated

### 9.1 Main tree draft entry

The critical method is in:

- `/root/worktrees/vllm-ascend-clean/vllm_ascend/spec_decode/llm_base_proposer.py`

Method:

- `_run_tree_draft()`

This is the core of draft-side tree execution.

At a high level it does:

1. run the first draft forward from the target frontier
2. select root candidates
3. iterate depth by depth
4. build per-depth draft attention metadata
5. patch draft attention mask to tree form
6. assign distinct slot mappings for sibling branches
7. compute next-level logits from current-level parent hidden states
8. place chosen child tokens into the runtime-node result tensor
9. finally reorder to execution order

This is where Ascend stops being "upstream-compatible plumbing" and becomes a real backend implementation.

### 9.2 Why positions and slots both need tree logic

Tree mode needs two different notions to be correct:

- semantic position: based on branch depth
- physical KV slot: unique per materialized node

That is why the code separately uses:

- `build_tree_query_position_offsets()`
- `build_tree_query_slot_offsets()`

Sibling branches should share the same semantic depth but must not collide in KV storage. If you only patch one and not the other, the implementation becomes subtly wrong.

### 9.3 Why attention mask patching is necessary even inside draft execution

Within `_run_tree_draft()`, Ascend calls:

- `_patch_tree_attn_mask()`

using `build_tree_attn_mask(...)`.

This exists because the ordinary speculative drafting path assumes linear causal growth. Tree drafting needs ancestor-only visibility, not plain left-to-right visibility.

So tree mode is not just "same batch expansion but wider". It changes the legality of attention edges.

## 10. Ascend target-side tree execution: why model runner also needs patching

One easy misunderstanding is thinking that once the drafter emits tree tokens, the target side can stay unchanged. That is false.

Ascend also patches the target-side speculative execution in `model_runner_v1.py`.

### 10.1 Tree metadata construction

Key method:

- `_calc_tree_spec_decode_metadata()`

This method builds `AscendTreeSpecDecodeMetadata` for the target-side verification pass.

It computes:

- per-request node offsets
- per-request row offsets
- `num_draft_tokens`
- `cu_num_draft_tokens`
- `target_logits_indices`
- `bonus_logits_indices`
- `draft_token_ids`

This is crucial because tree verification does not consume the same row semantics as linear chain speculative decode.

### 10.2 Position patch on target side

Key method:

- `_patch_tree_spec_decode_positions()`

Why it exists:

- target-side execution still constructs batched rows using the existing speculative decode path
- tree mode needs depth-based semantic positions for those rows

So this method rewrites the semantic positions for the tree rows while preserving the already prepared physical layout.

### 10.3 Attention mask patch on target side

Key method:

- `_patch_tree_spec_decode_attn_metadata()`

It overlays:

- `build_tree_target_attn_mask(runtime)`

onto the target-side attention metadata.

This is the target-side analog of draft-side tree mask patching. Without it, target verification would still interpret tree rows with chain-style visibility.

### 10.4 Where these patches are applied

In `model_runner_v1.py`, before building final attention metadata for target execution:

- if `spec_decode_metadata` has `runtime`, patch positions
- then build target attention metadata
- then patch target attention masks

This ordering is intentional:

- positions must be tree-semantic before RoPE / input semantics are consumed
- attention metadata must be built first before mask overlay can be applied

## 11. Ascend verifier path: how accepted branch feeds back into next-round drafting

### 11.1 Tree-aware rejection sampler

Main file:

- `/root/worktrees/vllm-ascend-clean/vllm_ascend/sample/rejection_sampler.py`

When `metadata` is `AscendTreeSpecDecodeMetadata`, the sampler takes a separate path.

What it does:

1. requires greedy mode
2. computes target token ids from target logits
3. calls `select_tree_accept_paths(...)`
4. records accepted terminal row indices back onto the runner
5. packs accepted-path tokens into the existing sampler output tensor shape

Current hard limit from this file:

- tree verifier supports greedy sampling only

That is not just a documentation statement. It is explicitly enforced in code.

### 11.2 Why `tree_terminal_row_indices` exists

The sampler writes:

- `runner.tree_terminal_row_indices`

That field is later consumed by the proposer path in:

- `vllm_ascend/spec_decode/llm_base_proposer.py`

inside the code that prepares the next draft inputs.

When tree mode is active and terminal row indices exist, the proposer uses those terminal rows as the next sampling frontier instead of using the default linear "query_start_loc - rejected_tokens" rule.

This is the state-propagation hook that makes multi-round tree execution possible.

Without this hook, accepted tree branches would be verified correctly for one round but the next round would continue from the wrong frontier.

## 12. Why there is also a monkey patch

Main file:

- `/root/worktrees/vllm-ascend-clean/vllm_ascend/patch/worker/patch_tree_spec_decode.py`

This file monkey-patches:

- `GPUModelRunner._create_speculative_drafter`

Behavior:

- if tree mode is enabled and method is `eagle` or `eagle3`, return `AscendEagleProposer`
- otherwise delegate to original upstream creation logic

Why this exists:

- parent/upstream model runner may create the drafter before full Ascend runner state is initialized
- tree-mode Ascend proposer needs Ascend runner assumptions and local overrides
- this patch forces tree-enabled upstream-style paths onto the Ascend proposer implementation

In practice, this patch is a compatibility shim that protects the tree path from accidentally falling back to upstream drafter creation.

## 13. Current support matrix

This is the practical status, not the broad marketing surface.

### 13.1 Upstream v023 contract layer

`speculative_token_tree` is publicly accepted for:

- `eagle`
- `eagle3`
- `mtp`
- `dflash`

But built-in runtime execution is still rejected.

### 13.2 Ascend V1 method dispatch surface

Ascend V1 has proposer classes for:

- `ngram`
- `ngram_gpu`
- `suffix`
- `medusa`
- `eagle`
- `eagle3`
- `mtp`
- `dflash`
- `draft_model`
- `extract_hidden_states`

### 13.3 Ascend V1 tree execution reality

Actual implemented tree execution is limited to:

- `eagle`
- `eagle3`

Current code-level limits:

- eager-only
- no graph / fullgraph tree execution
- greedy-only verification
- no LM-head TP in tree mode

### 13.4 Ascend V2 status

Relevant files:

- `/root/worktrees/vllm-ascend-clean/vllm_ascend/worker/v2/README.md`
- `/root/worktrees/vllm-ascend-clean/vllm_ascend/worker/v2/spec_decode/eagle/__init__.py`

V2 speculative decode is not feature-complete.

Current status:

- V2 `init_speculator()` only supports the Eagle path
- non-Eagle methods raise `NotImplementedError`
- the V2 README explicitly lists `speculator.graph_manager_wrapper` as an outstanding gap

So if your goal is "where is the real production tree implementation today?", the answer is:

- V1, not V2

## 14. Where the implementation is currently stuck

This section is the answer to "support到了什么程度 / 卡在哪里".

### 14.1 Upstream contract is broader than Ascend runtime

Upstream config allows tree for `mtp` and `dflash`, but Ascend runtime only executes tree for `eagle/eagle3`.

So one real gap is:

- contract-level openness > backend-level implementation

### 14.2 Tree mode is still eager-only

This limit is enforced in `AscendSpecDecodeBaseProposer._init_speculative_token_tree_runtime()`.

Why it matters:

- graph-mode draft execution on Ascend has its own ACL graph management
- tree mode changes input layout, per-step metadata, and mask semantics
- that graph path has not been made tree-safe

So "support tree mode" currently means "support tree mode in eager V1".

### 14.3 Greedy-only verification

Tree path in `AscendRejectionSampler.forward()` rejects non-greedy sampling.

This is a real algorithmic limitation, not just a test gap.

### 14.4 LM-head TP is unsupported in tree mode

`_compute_tree_logits()` raises when LM-head TP is enabled.

This means tree execution currently assumes single logical logits head behavior at the point where tree branching decisions are made.

### 14.5 V2 is not the place to extend first

Because V2 spec decode is still incomplete, especially around graph-manager integration, it is the wrong place to start if your goal is to understand or extend current tree support quickly.

## 15. Recommended source-reading order

If you want the shortest path to understanding, read in this order.

### 15.1 Must-read

1. `/root/worktrees/vllm-spec-tree-v023/vllm/config/speculative.py`
2. `/root/worktrees/vllm-spec-tree-v023/vllm/v1/spec_decode/llm_base_proposer.py`
3. `/root/worktrees/vllm-ascend-clean/vllm_ascend/spec_decode/llm_base_proposer.py`
4. `/root/worktrees/vllm-ascend-clean/vllm_ascend/worker/model_runner_v1.py`
5. `/root/worktrees/vllm-ascend-clean/vllm_ascend/sample/rejection_sampler.py`
6. `/root/worktrees/vllm-ascend-clean/vllm_ascend/spec_decode/tree_runtime.py`
7. `/root/worktrees/vllm-ascend-clean/vllm_ascend/spec_decode/tree_execution.py`
8. `/root/worktrees/vllm-ascend-clean/vllm_ascend/spec_decode/tree_verifier.py`

### 15.2 Read next if you are modifying behavior

- `/root/worktrees/vllm-ascend-clean/vllm_ascend/patch/worker/patch_tree_spec_decode.py`
- `/root/worktrees/vllm-ascend-clean/vllm_ascend/spec_decode/__init__.py`
- `/root/worktrees/vllm-ascend-clean/vllm_ascend/spec_decode/eagle_proposer.py`
- `/root/worktrees/vllm-spec-tree-v023/vllm/v1/worker/gpu_model_runner.py`

### 15.3 Read for validation and regression safety

- `/root/worktrees/vllm-ascend-clean/tests/ut/spec_decode/test_tree_runtime.py`
- `/root/worktrees/vllm-ascend-clean/tests/ut/spec_decode/test_tree_execution.py`
- `/root/worktrees/vllm-ascend-clean/tests/ut/spec_decode/test_tree_verifier.py`
- `/root/worktrees/vllm-ascend-clean/tests/ut/worker/a2/test_model_runner_tree_spec_decode.py`
- `/root/worktrees/vllm-ascend-clean/tests/ut/patch/worker/test_patch_tree_spec_decode.py`
- `/root/worktrees/vllm-spec-tree-v023/tests/test_config.py`
- `/root/worktrees/vllm-spec-tree-v023/tests/engine/test_arg_utils.py`
- `/root/worktrees/vllm-spec-tree-v023/tests/v1/spec_decode/test_speculative_token_tree.py`

## 16. What not to read first

If your immediate goal is to understand tree speculative decoding end to end, do not start from these.

### 16.1 Do not start from broad benchmark / examples directories

They show usage, not architecture.

### 16.2 Do not start from V2

`vllm_ascend/worker/v2/` is useful only after you already understand V1 and want to inspect future migration status.

### 16.3 Do not start from giant generic Eagle tests

Files like:

- `/root/worktrees/vllm-ascend-clean/tests/ut/spec_decode/a2/test_eagle_proposer.py`

are large and mix many concerns. They are poor entry points unless you already know the core runtime model.

### 16.4 Do not over-index on upstream docs alone

Upstream docs explain the public contract well, but they do not explain the Ascend runtime additions that make tree mode actually run.

## 17. Minimal end-to-end mental model

You can compress the whole system into this sequence:

1. User passes `speculative_config`, possibly with `speculative_token_tree`.
2. Upstream `SpeculativeConfig` parses and validates the tree.
3. Upstream built-in proposer base would normally reject tree execution.
4. Ascend proposer base overrides that rejection.
5. Ascend builds a local `SpecTreeRuntime`.
6. Ascend draft proposer executes the tree depth by depth and emits draft tokens in tree execution order.
7. Ascend model runner constructs tree-aware target metadata.
8. Ascend patches target positions and attention masks so target verification sees tree semantics instead of chain semantics.
9. Ascend rejection sampler verifies the tree, chooses the deepest accepted branch, and records the terminal row.
10. Next drafting round resumes from that accepted frontier.

If one of steps 5 through 10 is missing, you do not have real tree speculative decoding, only tree-shaped config.

## 18. Bottom line

The clean architectural split is:

- upstream v023: public config contract for `speculative_token_tree`, plus explicit runtime non-support in built-in proposers
- Ascend V1: local execution model that fills the runtime gap

The most important practical limits today are:

- real tree execution only on `eagle/eagle3`
- eager-only
- greedy-only verifier
- no LM-head TP in tree mode
- V2 not ready as the main extension surface

If you want to modify or debug this feature, spend most of your time in:

- `vllm_ascend/spec_decode/llm_base_proposer.py`
- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/sample/rejection_sampler.py`
- `vllm_ascend/spec_decode/tree_execution.py`
- `vllm_ascend/spec_decode/tree_verifier.py`

That is where the real backend semantics live.
