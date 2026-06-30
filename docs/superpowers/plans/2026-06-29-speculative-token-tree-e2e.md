# Speculative Token Tree End-to-End Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `speculative_token_tree` a correct end-to-end feature by defining a stable upstream contract and implementing a fully working Ascend execution path, with real-model validation as the primary acceptance gate.

**Architecture:** Split the work into two layers. Upstream vLLM owns the generic tree contract, metadata semantics, verifier semantics, and feature gating. vLLM-Ascend owns the actual runnable backend integration: drafter layout, target-model attention/position semantics, sampler integration, async scheduling compatibility, and optional Triton acceleration.

**Tech Stack:** Python, PyTorch, vLLM v1 scheduler/model-runner stack, vLLM-Ascend attention/runtime stack, optional Triton-Ascend kernels, 32B real-model E2E scripts.

## Global Constraints

- Upstream and downstream must use one shared tree contract; Ascend-specific behavior cannot invent a private token-tree semantic.
- The final feature target is branching trees, not chain-only speculative decode.
- Validation priority is real E2E behavior on 32B models; unit tests stay minimal and only protect key contracts.
- First working target is greedy tree speculative decode for eager mode; later sampling modes can reuse the same contract.
- If Python reference logic is correct but too slow, Triton is allowed as an optimization layer, not as the first source of truth.
- No commit or push during this phase.

---

### Task 1: Freeze The Cross-Repo Tree Contract

**Files:**
- Modify: `/root/worktrees/vllm-spec-tree-v023/vllm/config/speculative.py`
- Create: `/root/worktrees/vllm-spec-tree-v023/vllm/v1/spec_decode/tree_runtime.py`
- Create: `/root/worktrees/vllm-spec-tree-v023/vllm/v1/spec_decode/tree_contract.py`
- Modify: `/root/worktrees/vllm-spec-tree-v023/vllm/v1/spec_decode/llm_base_proposer.py`
- Modify: `/root/worktrees/vllm-ascend-clean/docs/source/user_guide/feature_guide/speculative_decoding.md`

**Interfaces:**
- Consumes: `SpeculativeConfig.speculative_token_tree`
- Produces: a hardware-agnostic tree runtime contract:
  - canonical node ordering
  - parent/child relation
  - node depth
  - accepted-path rule
  - bonus-token rule
  - backend capability gate

- [ ] Define one canonical node order for tree execution.
  The recommended order is breadth-first by depth, stable within each depth by parsed path order.

- [ ] Define the verifier contract in words before code:
  - target logit row `i` validates draft node `i`
  - a node is valid only if its parent path is fully valid
  - accepted output is the deepest valid path, not “all individually matched nodes”
  - bonus token is only legal after the accepted path terminates at the verifier-defined frontier

- [ ] Upstream should explicitly expose a capability gate rather than pretending all backends support tree mode today.
  Example outcome:
  - config parses `speculative_token_tree`
  - generic contract exists
  - proposer/backend can raise `NotImplementedError` unless they opt in

- [ ] Document the current supported scope clearly:
  - branching trees are supported by contract
  - first runnable backend target is Ascend eager greedy path
  - other backends may adopt the same contract later


### Task 2: Rebuild Upstream Tree Metadata Semantics

**Files:**
- Create: `/root/worktrees/vllm-spec-tree-v023/vllm/v1/spec_decode/tree_metadata.py`
- Modify: `/root/worktrees/vllm-spec-tree-v023/vllm/v1/spec_decode/metadata.py`
- Modify: `/root/worktrees/vllm-spec-tree-v023/vllm/v1/worker/gpu_model_runner.py`
- Modify: `/root/worktrees/vllm-spec-tree-v023/vllm/v1/sample/rejection_sampler.py`

**Interfaces:**
- Consumes: canonical tree runtime from Task 1
- Produces: generic tree metadata fields:
  - `draft_token_ids`
  - `target_logits_indices`
  - `bonus_logits_indices`
  - `node_parent_indices`
  - `node_depths`
  - optional `request_node_offsets`
  - optional `tree_query_positions`

- [ ] Stop modeling tree mode as “normal speculative decode with a different mask”.
  The metadata must encode parent/child semantics explicitly.

- [ ] Replace the current single-dimension assumption behind:
  - `target_logits_indices`
  - `bonus_logits_indices`
  - `draft_token_ids`
  with a tree-aware contract.

- [ ] Decide the bonus policy in metadata, not ad hoc in backend code.
  Recommended policy:
  - Python reference path first
  - if leaf-specific bonus semantics are not ready, temporarily disable bonus append for tree mode instead of appending the wrong token

- [ ] Add one generic verifier/reference implementation upstream.
  This implementation becomes the semantic source of truth for downstream backends.

- [ ] Keep this task backend-neutral.
  No Ascend-specific attention patching should leak into upstream files.


### Task 3: Rebuild Ascend Target-Side Tree Execution

**Files:**
- Modify: `/root/worktrees/vllm-ascend-clean/vllm_ascend/worker/model_runner_v1.py`
- Modify: `/root/worktrees/vllm-ascend-clean/vllm_ascend/spec_decode/llm_base_proposer.py`
- Modify: `/root/worktrees/vllm-ascend-clean/vllm_ascend/spec_decode/tree_execution.py`
- Modify: `/root/worktrees/vllm-ascend-clean/vllm_ascend/spec_decode/tree_runtime.py`
- Modify: `/root/worktrees/vllm-ascend-clean/vllm_ascend/spec_decode/tree_verifier.py`

**Interfaces:**
- Consumes: upstream tree contract and Ascend eager eagle/eagle3 draft outputs
- Produces: correct target-side tree execution state:
  - correct token order
  - correct target verification rows
  - correct tree query positions
  - correct accepted path packing

- [ ] Replace the current prototype `_calc_tree_spec_decode_metadata(...)` with a real tree metadata builder.
  It must not assume:
  - every tree node is validated like a chain token
  - one fixed bonus row per request is always correct
  - `draft_token_ids = self.input_ids.gpu[target_logits_indices]` is automatically semantically valid

- [ ] Build explicit tree query positions for target verification.
  The target model must see the same tree depth semantics as the drafter:
  - root children at base position + 1
  - grandchildren at base position + 2
  - branching siblings share depth, not linear token offset

- [ ] Patch target-side attention metadata using tree semantics, not only generic causal masks.
  This likely needs both:
  - tree attention mask
  - tree-aware positions

- [ ] Refactor the current verifier so that it consumes explicit tree metadata instead of inferring semantics from flat tensors.

- [ ] If bonus semantics remain ambiguous at this stage, ship the first correct version with tree bonus disabled rather than wrong.
  Correctness first, bonus second.


### Task 4: Fix Scheduler, Async, And Drafters As A Full Chain

**Files:**
- Modify: `/root/worktrees/vllm-ascend-clean/vllm_ascend/worker/model_runner_v1.py`
- Modify: `/root/worktrees/vllm-ascend-clean/vllm_ascend/core/recompute_scheduler.py`
- Modify: `/root/worktrees/vllm-ascend-clean/vllm_ascend/core/scheduler_dynamic_batch.py`
- Modify: `/root/worktrees/vllm-ascend-clean/vllm_ascend/core/scheduler_profiling_chunk.py`
- Modify: `/root/worktrees/vllm-ascend-clean/vllm_ascend/patch/worker/patch_tree_spec_decode.py`

**Interfaces:**
- Consumes: tree draft tokens from Ascend drafter
- Produces: request-visible `spec_token_ids` whose order matches verifier/runtime order exactly

- [ ] Verify that the drafter output shape `[batch, node_count]` is copied into scheduler state in the same canonical node order as the verifier consumes.

- [ ] Audit all places that assume:
  - `num_spec_tokens == chain length`
  - “last valid token” represents the next decode token
  - placeholders can be treated as linear rejected chain tokens

- [ ] For async scheduling, define what `prev_sampled_token_ids` means in tree mode.
  It cannot silently reuse chain semantics if the accepted output is a path rather than a prefix of a flat speculative list.

- [ ] Ensure `request.spec_token_ids`, `scheduled_spec_decode_tokens`, `valid_sampled_tokens_count`, and `num_rejected_tokens` all agree with the tree accepted-path model.

- [ ] Keep the first supported runtime narrow if needed:
  - eager only
  - no PCP/CP coupling initially
  - no lmhead TP initially
  But encode those limits as explicit gates, not silent misbehavior.


### Task 5: Add Triton Only Where Python Semantics Are Proven

**Files:**
- Modify or Create: `/root/worktrees/vllm-ascend-clean/vllm_ascend/ops/triton/reject_sample.py`
- Modify or Create: `/root/worktrees/vllm-ascend-clean/vllm_ascend/ops/triton/spec_decode/*.py`
- Modify: `/root/worktrees/vllm-ascend-clean/vllm_ascend/sample/rejection_sampler.py`

**Interfaces:**
- Consumes: proven Python reference tree semantics
- Produces: optional accelerated kernels for:
  - accepted-path scan
  - bonus placement
  - recovered/packed output layout

- [ ] Do not start with Triton.
  First make the Python/torch reference path correct on 32B real inference.

- [ ] Add Triton only after the reference path stabilizes and profiling shows a real hotspot.

- [ ] The best first Triton target is the tree verifier/output packing path, not the attention path.
  Rationale:
  - lower surface area
  - easier to validate against Python reference
  - less risk than inventing a tree-specific attention kernel first

- [ ] Do not write a custom tree attention Triton kernel unless profiling proves dense-mask attention is the bottleneck after correctness is solved.


### Task 6: Verification Strategy Centered On Real Models

**Files:**
- Modify: `/root/tmp_tree_eager_quick_check.py`
- Modify: `/root/worktrees/vllm-ascend-clean/run_qwen32_tree_compare.py`
- Create: `/root/worktrees/vllm-ascend-clean/scripts/spec_tree/run_tree_smoke.py`
- Create: `/root/worktrees/vllm-ascend-clean/scripts/spec_tree/run_tree_qwen32_compare.py`

**Interfaces:**
- Consumes: base model + draft model + tree config
- Produces: machine-readable JSON reports for baseline vs tree

- [ ] Use two real acceptance gates:
  - quick eager smoke: short Chinese prompt, compare baseline vs tree text and tokens
  - full 32B compare: larger prompt, compare answer quality and acceptance metrics

- [ ] Keep unit coverage minimal.
  Only preserve small tests for:
  - canonical tree runtime build
  - accepted deepest-path rule
  - metadata layout invariants

- [ ] Primary pass/fail criteria:
  - tree output is semantically normal English/Chinese text, not gibberish
  - baseline and tree do not diverge catastrophically on deterministic greedy prompts
  - acceptance metrics are plausible
  - no scheduler corruption in the next step

- [ ] Save result JSONs under `/tmp` and a short markdown report under `/root`.


### Task 7: Cleanup And Refactor After First Green E2E

**Files:**
- Modify: `/root/worktrees/vllm-ascend-clean/vllm_ascend/spec_decode/tree_execution.py`
- Modify: `/root/worktrees/vllm-ascend-clean/vllm_ascend/spec_decode/tree_runtime.py`
- Modify: `/root/worktrees/vllm-ascend-clean/vllm_ascend/spec_decode/tree_verifier.py`
- Modify: `/root/worktrees/vllm-ascend-clean/vllm_ascend/worker/model_runner_v1.py`
- Modify: `/root/worktrees/vllm-ascend-clean/vllm_ascend/patch/worker/__init__.py`

**Interfaces:**
- Consumes: first working end-to-end implementation
- Produces: cleaner module boundaries

- [ ] Remove prototype logic that was only useful for bring-up.

- [ ] Move tree-specific logic out of generic files where possible.
  Recommended boundary:
  - runtime/contract
  - metadata builder
  - verifier
  - backend patch hook
  - optional Triton acceleration

- [ ] Keep only the patches that are structurally necessary for upstream/downstream version skew.

- [ ] Update docs with the real supported matrix:
  - methods
  - eager vs graph
  - greedy vs sampled
  - TP/CP/PCP limits
  - draft-model requirements


## Recommended Execution Order

- [ ] Phase A: Upstream contract and metadata semantics
- [ ] Phase B: Ascend target execution correctness
- [ ] Phase C: Scheduler/async alignment
- [ ] Phase D: Real 32B eager validation
- [ ] Phase E: Triton optimization only if needed
- [ ] Phase F: cleanup and documentation


## Current Diagnosis Summary

- [ ] Confirmed: current Ascend tree path is not correct end-to-end.
- [ ] Confirmed: target-side tree attention mask was previously missing and has already been partially patched.
- [ ] Suspected remaining gaps:
  - target-side tree positions still use chain/linear semantics
  - tree metadata does not yet fully encode parent/child verification semantics
  - bonus token semantics are still modeled too simplistically
  - scheduler/async path still assumes chain-style speculative decode in several places


## First Concrete Milestone

- [ ] Produce one deterministic eager run where:
  - baseline output is normal
  - tree output is also normal
  - no gibberish repeated tokens like `222222...`
  - acceptance metrics remain non-zero
  - next-step scheduling still works

