#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from dataclasses import dataclass
from enum import Enum
import json
import os
import time

import torch
import torch_npu
import vllm.envs as envs_vllm
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (  # type: ignore
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.v1.attention.backends.registry import (  # type: ignore
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, CrossAttentionSpec

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.context_parallel.common_cp import AscendMetadataForDecode, AscendMetadataForPrefill
from vllm_ascend.attention.kvcomp_attn.attention_utils import (
    get_kvcomp_decode_params,
    is_enable_hamming_sparse,
    reshape_and_cache_kvcomp,
)
from vllm_ascend.attention.tree_attention import (
    build_tree_split_plan,
    build_tree_paged_plan,
    gather_paged_kv_to_ragged,
    merge_split_tree_attention_outputs,
    maybe_dump_tree_exact_debug,
    resolve_tree_fast_impl,
    run_fast_tree_attention,
    run_fast_tree_attention_from_paged_cache,
    run_exact_tree_attention,
    run_split_current_tree_attention,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    enable_cp,
    get_effective_decode_threshold,
    notify_kv_cache_written,
    split_decodes_and_prefills,
    using_paged_attention,
)
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_draft_graph_prefill_params,
    get_graph_params,
    update_draft_graph_params_workspaces,
    update_graph_params_workspaces,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.memcache_comm_fence import record_attention_compute_start
from vllm_ascend.ops.flashcomm2_oshard_manager import flashcomm2_oshard_manager
from vllm_ascend.utils import weak_ref_tensors
from vllm_ascend.worker.kvcomp_utils import KVCompMetaData

# default max value of sliding window size
SWA_INT_MAX = 2147483647
_ATTN_KEYS_BUFFER = None
_TREE_ATTN_TRACE_COUNT = 0
_PREFIX_CACHE_TRACE_COUNT = 0
_TREE_FAST_PROFILE_COUNT = 0
_TREE_SPLIT_COMPARE_COUNT = 0


def _maybe_dump_tree_attn_trace(
    *,
    layer_name: str | None,
    impl_name: str,
    path: str,
    attn_metadata,
    extra: dict | None = None,
) -> None:
    trace_path = os.getenv("VLLM_ASCEND_TREE_ATTN_TRACE_PATH")
    if not trace_path:
        return

    global _TREE_ATTN_TRACE_COUNT
    max_steps = int(os.getenv("VLLM_ASCEND_TREE_ATTN_TRACE_STEPS", "256"))
    if _TREE_ATTN_TRACE_COUNT >= max_steps:
        return

    payload = {
        "index": _TREE_ATTN_TRACE_COUNT,
        "pid": os.getpid(),
        "layer_name": layer_name,
        "impl_name": impl_name,
        "path": path,
        "tree_spec_decode_enabled": getattr(attn_metadata, "tree_spec_decode_enabled", None),
        "attn_state": getattr(getattr(attn_metadata, "attn_state", None), "name", None),
        "num_actual_tokens": getattr(attn_metadata, "num_actual_tokens", None),
        "actual_seq_lengths_q": getattr(attn_metadata, "actual_seq_lengths_q", None),
        "seq_lens_list": getattr(attn_metadata, "seq_lens_list", None),
        "block_tables_present": getattr(attn_metadata, "block_tables", None) is not None,
    }
    if extra:
        payload.update(extra)
    with open(trace_path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _TREE_ATTN_TRACE_COUNT += 1


def _maybe_dump_tree_fast_profile(payload: dict) -> None:
    trace_path = os.getenv("VLLM_ASCEND_TREE_FAST_PROFILE_PATH")
    if not trace_path or get_tensor_model_parallel_rank() != 0:
        return

    global _TREE_FAST_PROFILE_COUNT
    max_steps = int(os.getenv("VLLM_ASCEND_TREE_FAST_PROFILE_STEPS", "128"))
    if _TREE_FAST_PROFILE_COUNT >= max_steps:
        return

    data = dict(payload)
    data["index"] = _TREE_FAST_PROFILE_COUNT
    data["pid"] = os.getpid()
    with open(trace_path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(data, ensure_ascii=False) + "\n")
    _TREE_FAST_PROFILE_COUNT += 1


def _maybe_dump_tree_split_compare(payload: dict) -> None:
    trace_path = os.getenv("VLLM_ASCEND_TREE_SPLIT_COMPARE_PATH")
    if not trace_path or get_tensor_model_parallel_rank() != 0:
        return

    global _TREE_SPLIT_COMPARE_COUNT
    max_steps = int(os.getenv("VLLM_ASCEND_TREE_SPLIT_COMPARE_STEPS", "16"))
    if _TREE_SPLIT_COMPARE_COUNT >= max_steps:
        return

    data = dict(payload)
    data["index"] = _TREE_SPLIT_COMPARE_COUNT
    data["pid"] = os.getpid()
    with open(trace_path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(data, ensure_ascii=False) + "\n")
    _TREE_SPLIT_COMPARE_COUNT += 1


def _matches_trace_filters(layer_name: str | None, filter_spec: str | None) -> bool:
    if not filter_spec:
        return True
    if layer_name is None:
        return False
    filters = [item.strip() for item in filter_spec.split(",") if item.strip()]
    if not filters:
        return True
    return any(item in layer_name for item in filters)


@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # HACK(Ronald1995): vllm `initialize_kv_cache` method in model runner v2 make
        # attention name assertion, we just set name to FLASH_ATTN to avoid assertion error.
        # rectify this when vllm disable the assertion.
        return "CUSTOM" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AscendAttentionBackendImpl"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPImpl

            return AscendAttentionCPImpl
        return AscendAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPMetadataBuilder

            return AscendAttentionCPMetadataBuilder
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: list[torch.Tensor],
        dst_kv_cache: list[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(dst_key_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            key_caches = kv_cache[0]
            value_caches = kv_cache[1]
            key_caches[dst_indices] = key_caches[src_indices]
            value_caches[dst_indices] = value_caches[src_indices]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]


class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


@dataclass
class AscendMetadata:
    """
    Per-layer attention metadata for Ascend FlashAttention backend.

    Contains attention masks, token counts, sequence lengths and KV cache
    related properties for attention computation.
    """

    # **************************** Basic Properties ************************** #
    attn_mask: torch.Tensor | None = None
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    # Number of tokens excluding padding.
    num_actual_tokens_pcp_padded: int = 0
    num_actual_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0

    # The sequence length per sequence. Sequence length means the computed
    # tokens + new tokens (is None if it is a decoding).
    # (batch_size,)
    # TODO(Angazenn): The following parameters are quite redundant and
    # contains similar information (such as seq_lens seq_lens_list). We
    # should simplified these parameters once attention schema in vLLM-Ascend
    # is unified.
    seq_lens: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None
    seq_lens_list: list[int] = None  # type: ignore
    actual_seq_lengths_q: list[int] = None  # type: ignore

    query_start_loc: torch.Tensor = None
    # Maximum query length in the batch (None for decoding).
    max_query_len: int | None = None

    # ********************** KV Cache Related Properties ********************* #
    # Block addresses per sequence (Seq id -> list of physical block).
    # (batch_size, max_blocks_per_seq)
    block_tables: torch.Tensor = None

    # The indices of the token slots that input tokens will be stored into.
    # E.g., if `slot_mapping` is [35, 2, 17] and the block size is 16, the
    # three tokens are stored in the 3rd slot in block 2, 2nd slot in block 0,
    # and 1st slot in block 1, respectively.
    # (num_tokens,)
    slot_mapping: torch.Tensor = None
    # pcp
    prefill: AscendMetadataForPrefill | None = None
    # dcp
    decode_meta: AscendMetadataForDecode | None = None

    causal: bool = True
    # runner_type in model_config.
    model_runner_type: str = ""
    # prefill reshape_and_cache event
    reshape_cache_event: torch.npu.Event = None

    kvcomp_metadata: KVCompMetaData | None = None
    tree_spec_decode_enabled: bool = False
    tree_fast_non_root_plan: object | None = None
    tree_fast_non_root_plan_key: tuple[object, ...] | None = None
    tree_fast_all_rows_plan: object | None = None
    tree_fast_mixed_batch_fallback: bool = False


class AscendAttentionMetadataBuilder(AttentionMetadataBuilder[AscendMetadata]):
    """
    Builder for constructing AscendMetadata from CommonAttentionMetadata.

    Handles attention mask generation and metadata preparation for
    Ascend FlashAttention backend.
    """

    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.compilation_config = vllm_config.compilation_config
        self.device = device
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len, AscendAttentionBackend.get_supported_kernel_block_sizes()[0]
        )

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )

        self.reorder_batch_threshold = self.decode_threshold

        scheduler_config = vllm_config.scheduler_config
        self.chunked_prefill_enabled = scheduler_config.enable_chunked_prefill
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendAttentionMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.ALWAYS

    def reorder_batch(self, input_batch, scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=get_effective_decode_threshold(
                common_attn_metadata,
                self.decode_threshold,
            ),
        )

        block_table = common_attn_metadata.block_table_tensor
        # Prefer _seq_lens_cpu (always available, updated during draft
        # iterations) over seq_lens_cpu (None in async spec decode mode).
        if common_attn_metadata._seq_lens_cpu is not None:
            seq_lens = common_attn_metadata._seq_lens_cpu[:num_reqs]
        elif common_attn_metadata.seq_lens_cpu is not None:
            seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]
        else:
            seq_lens = common_attn_metadata.seq_lens[:num_reqs].to("cpu")

        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]
        # this slot_mapping override doesn't work since vllm will override it again. We should fix it vllm.
        # see: https://github.com/vllm-project/vllm/blob/ce88756b967c2c5006746a424c15dd59a284ed8c/vllm/model_executor/layers/attention/cross_attention.py#L117
        if isinstance(self.kv_cache_spec, CrossAttentionSpec):
            seq_lens = common_attn_metadata.seq_lens
            slot_mapping = common_attn_metadata.slot_mapping.to(torch.int32)
        elif self.speculative_config and self.speculative_config.parallel_drafting:
            seq_lens = common_attn_metadata.seq_lens

        attn_state = common_attn_metadata.attn_state

        # Get attn_mask from singleton AttentionMaskBuilder
        attn_mask = self.attn_mask_builder.get_attention_mask(common_attn_metadata.causal, self.model_config)

        # TODO: Yet another unnecessary H2D while we already have a query_start_loc on device
        query_start_loc = query_start_loc_cpu.pin_memory().to(self.device, non_blocking=True)

        actual_seq_lengths_q = query_start_loc_cpu[1:].tolist()
        seq_lens_list = seq_lens.tolist()
        # flashcomm1/SP (or cudagraph) padding makes the model runner insert a
        # dummy padding request into query_start_loc to satisfy the FIA TND-layout
        # constraint (sum of q lengths == hidden_states.shape[0]), bumping the
        # q-derived batchSize by one. The query_start_loc buffer is sized
        # `max_num_reqs + 2` to hold it, but the seq_lens and block_table buffers
        # are only `max_num_reqs`, so when the batch is full the padded request
        # overflows and `[:num_reqs_padded]` silently truncates them. FIA then
        # fails (error 561002) checking, in order, the `actualSeqLengthsKv` length
        # and then the block_table row count against batchSize. Pad them to match:
        # the dummy request points at block 0, and its output is harmless because:
        #   (1) read side: the attention output for padding tokens is trimmed by
        #       `hidden_states = hidden_states[:-pad_size, :]` downstream;
        #   (2) write side: reshape_and_cache slices key/value/slot_mapping to
        #       `[:num_actual_tokens]` (unpadded count), so the dummy request
        #       never writes to KV cache.
        # So any valid positive KV length / zero block row is fine. Pad both
        # seq_lens_list and the seq_lens tensor: full_graph_fia_v2 passes the
        # seq_lens tensor (not seq_lens_list) as actual_seq_kvlen during graph
        # capture, and _get_fia_params derives the PrefillCacheHit batch size from
        # seq_lens.shape[0], so the tensor has to carry the dummy request too.
        num_reqs_fia = len(actual_seq_lengths_q)
        if len(seq_lens_list) < num_reqs_fia:
            padding_len = num_reqs_fia - len(seq_lens_list)
            seq_lens_list = seq_lens_list + [1] * padding_len
            seq_lens = torch.cat([seq_lens, seq_lens.new_ones(padding_len)])
        if block_table is not None and block_table.shape[0] < num_reqs_fia:
            block_table = torch.cat(
                [
                    block_table,
                    block_table.new_zeros((num_reqs_fia - block_table.shape[0], block_table.shape[1])),
                ],
                dim=0,
            )

        attn_metadata = AscendMetadata(
            num_actual_tokens=num_actual_tokens,
            num_decode_tokens=num_decode_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            seq_lens_list=seq_lens_list,
            max_query_len=common_attn_metadata.max_query_len,
            actual_seq_lengths_q=actual_seq_lengths_q,
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            attn_state=attn_state,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            causal=common_attn_metadata.causal,
            model_runner_type=self.model_config.runner_type,
            kvcomp_metadata=common_attn_metadata.kvcomp_metadata,
            tree_spec_decode_enabled=common_attn_metadata.tree_spec_decode_enabled,
        )
        return attn_metadata

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ):
        if attn_state in (
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.ChunkedPrefill,
            AscendAttentionState.SpecDecoding,
        ):
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly and ChunkedPrefill state"
            )

        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendAttentionBackendImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        sinks: torch.Tensor = None,
        **kwargs,
    ) -> None:
        self.vllm_config = get_current_vllm_config()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32, device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None
        self.is_kv_producer = (
            self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.is_kv_producer
        )
        self.enable_c8_quant = self.vllm_config.quant_config is not None and getattr(
            self.vllm_config.quant_config, "enable_c8_quant", False
        )
        self.sinks = sinks
        self.layerIndex = 0
        self.enable_hamming_sparse = is_enable_hamming_sparse()

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        num_dcp_pcp_tokens=None,
        draft_attn_metadatas=None,
    ):
        if using_paged_attention(num_tokens, vllm_config):
            # Paged Attention update logic
            if _EXTRA_CTX.is_draft_model:
                if _EXTRA_CTX.is_draft_model_prefill:
                    graph_params = get_draft_graph_prefill_params()
                else:
                    graph_params = get_draft_graph_params()
            else:
                graph_params = get_graph_params()
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    forward_context.attn_metadata,
                    graph_params.attn_params[num_tokens],
                    graph_params.handles[num_tokens],
                    graph_params.events[num_tokens],
                ):
                    (
                        query,
                        key_cache,
                        value_cache,
                        num_kv_heads,
                        num_heads,
                        scale,
                        block_table,
                        seq_lens,
                        output,
                    ) = param
                    seq_lens = forward_context.attn_metadata[key].seq_lens

                    workspace = torch_npu._npu_paged_attention_get_workspace(
                        query=query,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        num_kv_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale_value=scale,
                        block_table=block_table,
                        context_lens=seq_lens,
                        out=output,
                    )
                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu._npu_paged_attention(
                        query=query,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        num_kv_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale_value=scale,
                        block_table=block_table,
                        context_lens=seq_lens,
                        out=output,
                        workspace=workspace,
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    event.record(update_stream)
        elif _EXTRA_CTX.sinks:
            # FIA update logic
            if _EXTRA_CTX.is_draft_model:
                graph_params = get_draft_graph_params()
                attn_metadata = draft_attn_metadatas
                attn_keys = list(attn_metadata[0].keys())
            else:
                graph_params = get_graph_params()
                attn_metadata = forward_context.attn_metadata
                attn_keys = list(attn_metadata.keys())
            # For Qwen3-next, since the kv_cache_config has already categorized
            # linear_attn and self_attn, the attn_metadata is first arranged with
            # self_attn followed by linear_attn. Therefore, using zip directly
            # filters out the update operations for linear_attn.
            # TODO: We use a new variable `attn_keys` to ensure the loop count is
            # correct after get by `zip` because of the new structure of the attn_metadata
            # when running with the merged full eagle-graph. Should check it with Qwen3-next.
            num_layers = len(attn_keys)
            if num_layers == 0:
                return
            if _EXTRA_CTX.is_draft_model:
                attn_keys = attn_keys * (len(graph_params.attn_params[num_tokens]) // num_layers)
            attn_count = 0
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    attn_keys,
                    graph_params.attn_params[num_tokens],
                    graph_params.handles[num_tokens],
                    graph_params.events[num_tokens],
                ):
                    (
                        query,
                        key_cache,
                        value,
                        block_tables,
                        attn_mask,
                        block_size,
                        seq_lens,
                        num_kv_heads,
                        num_heads,
                        scale,
                        sliding_window,
                        sinks,
                        attn_output,
                        softmax_lse,
                    ) = param

                    if _EXTRA_CTX.is_draft_model:
                        draft_step = attn_count // num_layers
                        seq_lens = attn_metadata[draft_step][key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[draft_step][key].actual_seq_lengths_q
                        attn_count = attn_count + 1
                    else:
                        seq_lens = attn_metadata[key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[key].actual_seq_lengths_q

                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu.npu_fused_infer_attention_score_v2.out(
                        query=query,
                        key=key_cache,
                        value=value,
                        block_table=block_tables,
                        atten_mask=attn_mask,
                        input_layout="TND",
                        block_size=block_size,
                        actual_seq_qlen=actual_seq_lengths_q,
                        actual_seq_kvlen=seq_lens,
                        num_key_value_heads=num_kv_heads,
                        num_query_heads=num_heads,
                        sparse_mode=4 if sliding_window is not None else 3,
                        pre_tokens=sliding_window if sliding_window is not None else SWA_INT_MAX,
                        next_tokens=0,
                        softmax_scale=scale,
                        learnable_sink=sinks,
                        workspace=graph_params.workspaces.get(num_tokens),
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    event.record(update_stream)
        else:
            # FIA update logic
            if _EXTRA_CTX.is_draft_model:
                if _EXTRA_CTX.is_draft_model_prefill:
                    graph_params = get_draft_graph_prefill_params()
                else:
                    graph_params = get_draft_graph_params()
                attn_metadata = draft_attn_metadatas
                attn_keys = list(attn_metadata[0].keys())
            else:
                graph_params = get_graph_params()
                attn_metadata = forward_context.attn_metadata
                attn_keys = list(attn_metadata.keys())
                # In some speculative methods (such as DFlash), the order of attn_keys in the Target model
                # will be disrupted instead of increasing by layer index, so need regular expressions to
                # reorder the attn_keys and stor the results in _ATTN_KEYS_BUFFER.
                attn_keys_length = len(graph_params.attn_params[num_tokens])
                global _ATTN_KEYS_BUFFER
                if _ATTN_KEYS_BUFFER is None:
                    import regex as re

                    def extract_layer_index(key: str) -> int:
                        match = re.search(r"(\d+)", key)
                        return int(match.group(1)) if match else 0

                    attn_keys_tmp = attn_keys[:attn_keys_length]
                    attn_keys_tmp.sort(key=extract_layer_index)
                    _ATTN_KEYS_BUFFER = attn_keys_tmp
                attn_keys[:attn_keys_length] = _ATTN_KEYS_BUFFER
            # For Qwen3-next, since the kv_cache_config has already categorized
            # linear_attn and self_attn, the attn_metadata is first arranged with
            # self_attn followed by linear_attn. Therefore, using zip directly
            # filters out the update operations for linear_attn.
            # TODO: We use a new variable `attn_keys` to ensure the loop count is
            # correct after get by `zip` because of the new structure of the attn_metadata
            # when running with the merged full eagle-graph. Should check it with Qwen3-next.
            num_layers = len(attn_keys)
            if num_layers == 0:
                return
            if _EXTRA_CTX.is_draft_model:
                attn_keys = attn_keys * (len(graph_params.attn_params[num_tokens]) // num_layers)
            attn_count = 0
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    attn_keys,
                    graph_params.attn_params[num_tokens],
                    graph_params.handles[num_tokens],
                    graph_params.events[num_tokens],
                ):
                    (
                        query,
                        key_cache,
                        value,
                        block_tables,
                        attn_mask,
                        block_size,
                        seq_lens,
                        query_start_loc,
                        num_kv_heads,
                        num_heads,
                        scale,
                        attn_output,
                        softmax_lse,
                        sparse_mode,
                        pre_tokens,
                        next_tokens,
                        c8_k_aq_scale,
                        c8_k_aq_offset,
                        c8_v_aq_scale,
                        c8_v_aq_offset,
                    ) = param

                    if _EXTRA_CTX.is_draft_model:
                        draft_step = attn_count // num_layers
                        seq_lens = attn_metadata[draft_step][key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[draft_step][key].actual_seq_lengths_q
                        block_tables = attn_metadata[draft_step][key].block_tables
                        attn_count = attn_count + 1
                        if not attn_metadata[draft_step][key].causal:
                            sparse_mode = 0
                    else:
                        seq_lens = attn_metadata[key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[key].actual_seq_lengths_q
                        # NOTE:
                        # For models with sliding-window attention on the FIA full-graph replay path,
                        # rebinding `block_tables` to the latest metadata tensor causes corrupted /
                        # repeated outputs in our repro on Ascend NPU.
                        #
                        # Keep the captured block_tables tensor on this affected path.
                        # Non-SWA models preserve the original behavior and continue to refresh
                        # block_tables from attn_metadata.
                        if not hasattr(vllm_config.model_config.hf_text_config, "sliding_window"):
                            block_tables = attn_metadata[key].block_tables

                    torch.npu.graph_task_update_begin(update_stream, handle)
                    input_layout = "TND"
                    extra_args = {}
                    if c8_k_aq_scale is not None:
                        extra_args = {
                            "key_antiquant_scale": c8_k_aq_scale,
                            "value_antiquant_scale": c8_v_aq_scale,
                            "key_antiquant_mode": 0,
                            "value_antiquant_mode": 0,
                            "inner_precise": 1,
                        }
                        input_layout = "BNSD"
                        sparse_mode = 0
                    torch_npu.npu_fused_infer_attention_score.out(
                        query=query,
                        key=key_cache,
                        value=value,
                        block_table=block_tables,
                        atten_mask=attn_mask,
                        input_layout=input_layout,
                        block_size=block_size,
                        actual_seq_lengths=actual_seq_lengths_q,
                        actual_seq_lengths_kv=seq_lens,
                        num_key_value_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale=scale,
                        sparse_mode=sparse_mode,
                        pre_tokens=pre_tokens,
                        next_tokens=next_tokens,
                        **extra_args,
                        workspace=graph_params.workspaces.get(num_tokens),
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)

                    event.record(update_stream)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)
        if flashcomm2_oshard_manager.flashcomm2_oshard_enable():
            flashcomm2_oshard_manager.post_process_after_loading()

    def full_graph_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer=None,
    ) -> torch.Tensor:
        passed_key = key
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)
        if self.enable_hamming_sparse and attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
            reshape_and_cache_kvcomp(attn_metadata.kvcomp_metadata, self.layerIndex, passed_key)
        elif self.enable_hamming_sparse:
            block_table, actual_seq_lengths_kv = get_kvcomp_decode_params(
                self.layerIndex, attn_metadata.kvcomp_metadata, query, passed_key, block_table, actual_seq_lengths_kv
            )

        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        if _EXTRA_CTX.is_draft_model:
            if _EXTRA_CTX.is_draft_model_prefill:
                graph_params = get_draft_graph_prefill_params()
            else:
                graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        # Prepare tensors for attention output
        # TODO: Refactor this to step-level instead of layer-level

        # Get workspace from cache or calculate it if not present.
        workspace = graph_params.workspaces.get(num_tokens)
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        input_layout = "TND"
        attn_mask = attn_metadata.attn_mask
        sparse_mode = 4 if self.sliding_window else 3 if attn_metadata.causal else 0
        pre_tokens = self.sliding_window or SWA_INT_MAX
        next_tokens = 0 if self.sliding_window else SWA_INT_MAX

        extra_args = {}
        if self.enable_c8_quant and layer is not None:
            extra_args = {
                "key_antiquant_scale": layer._c8_k_aq_scale_nz_bnsd,
                "value_antiquant_scale": layer._c8_v_aq_scale_nz_bnsd,
                "key_antiquant_mode": 0,
                "value_antiquant_mode": 0,
                "inner_precise": 1,
            }

            # change key/value shape
            _, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self._nz_5d_view(self.key_cache, block_size)
            value = self._nz_5d_view(self.value_cache, block_size)

            # TODO: change layerout from BNSD to TND.
            input_layout = "BNSD"
            query = query.unsqueeze(2)
            output = output.unsqueeze(2)
            attn_mask = None
            sparse_mode = 0
        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_mask,
                block_table=block_table,
                input_layout=input_layout,
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                sparse_mode=sparse_mode,
                pre_tokens=pre_tokens,
                next_tokens=next_tokens,
                scale=self.scale,
                **extra_args,
            )
            if _EXTRA_CTX.is_draft_model:
                update_draft_graph_params_workspaces(num_tokens, workspace)
            else:
                update_graph_params_workspaces(num_tokens, workspace)

        # Handle graph capturing mode
        stream = torch_npu.npu.current_stream()

        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        attn_params = (
            weak_ref_tensors(query),
            weak_ref_tensors(key),
            weak_ref_tensors(value),
            weak_ref_tensors(block_table),
            weak_ref_tensors(attn_mask) if attn_mask is not None else None,
            block_size,
            actual_seq_lengths_kv,
            actual_seq_lengths_q,
            self.num_kv_heads,
            self.num_heads,
            self.scale,
            weak_ref_tensors(output),
            weak_ref_tensors(softmax_lse),
            sparse_mode,
            pre_tokens,
            next_tokens,
        )
        if self.enable_c8_quant and layer is not None:
            attn_params = attn_params + (
                weak_ref_tensors(layer._c8_k_aq_scale_nz_bnsd),
                None,
                weak_ref_tensors(layer._c8_v_aq_scale_nz_bnsd),
                None,
            )  # type: ignore
        else:
            attn_params = attn_params + (None, None, None, None)  # type: ignore
        graph_params.attn_params[num_tokens].append(attn_params)

        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_mask,
            block_table=block_table,
            input_layout=input_layout,
            block_size=block_size,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=sparse_mode,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            workspace=workspace,
            out=[output, softmax_lse],
            **extra_args,
        )

        output = output.view(num_tokens, self.num_heads, self.head_size)

        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[num_tokens].append(handle)
        return output, num_tokens

    def full_graph_fia_v2(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)
        actual_seq_lengths_kv = attn_metadata.seq_lens
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        if _EXTRA_CTX.is_draft_model:
            graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()

        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        workspace = graph_params.workspaces.get(num_tokens)
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_qlen=actual_seq_lengths_q,
                actual_seq_kvlen=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                softmax_scale=self.scale,
                num_query_heads=self.num_heads,
                sparse_mode=4 if self.sliding_window is not None else 3,
                pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
                next_tokens=0,
                learnable_sink=self.sinks,
            )

            if _EXTRA_CTX.is_draft_model:
                update_draft_graph_params_workspaces(num_tokens, workspace)
            else:
                update_graph_params_workspaces(num_tokens, workspace)

        # Handle graph capturing mode
        stream = torch_npu.npu.current_stream()

        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        graph_params.attn_params[num_tokens].append(
            (
                weak_ref_tensors(query),
                weak_ref_tensors(key),
                weak_ref_tensors(value),
                weak_ref_tensors(block_table),
                weak_ref_tensors(attn_metadata.attn_mask),
                block_size,
                actual_seq_lengths_kv,
                self.num_kv_heads,
                self.num_heads,
                self.scale,
                self.sliding_window,
                self.sinks,
                weak_ref_tensors(output),
                weak_ref_tensors(softmax_lse),
            )
        )
        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score_v2.out(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_qlen=actual_seq_lengths_q,
            actual_seq_kvlen=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_query_heads=self.num_heads,
            sparse_mode=4 if self.sliding_window is not None else 3,
            pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
            next_tokens=0,
            softmax_scale=self.scale,
            learnable_sink=self.sinks,
            workspace=workspace,
            out=[output, softmax_lse],
        )
        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[num_tokens].append(handle)
        return output, num_tokens

    def full_graph_pa(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ):
        graph_params = get_graph_params()
        num_tokens = query.shape[0]
        if _EXTRA_CTX.capturing:
            # Get workspace from cache or calculate it if not present.
            workspace = graph_params.workspaces.get(num_tokens)
            if workspace is None:
                workspace = torch_npu._npu_paged_attention_get_workspace(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=attn_metadata.block_tables,
                    context_lens=attn_metadata.seq_lens,
                    out=output,
                )
                update_graph_params_workspaces(num_tokens, workspace)

            # Handle graph capturing mode
            stream = torch_npu.npu.current_stream()

            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            graph_params.events[num_tokens].append(event)
            graph_params.attn_params[num_tokens].append(
                (
                    weak_ref_tensors(query),
                    weak_ref_tensors(self.key_cache),
                    weak_ref_tensors(self.value_cache),
                    self.num_kv_heads,
                    self.num_heads,
                    self.scale,
                    attn_metadata.block_tables,
                    attn_metadata.seq_lens,
                    weak_ref_tensors(output),
                )
            )

            torch.npu.graph_task_group_begin(stream)
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                block_table=attn_metadata.block_tables,
                context_lens=attn_metadata.seq_lens,
                out=output,
                workspace=workspace,
            )
            handle = torch.npu.graph_task_group_end(stream)
            graph_params.handles[num_tokens].append(handle)
            return output

    def _get_fia_params(self, key: torch.Tensor, value: torch.Tensor, attn_metadata: AscendMetadata, kv_cache=None):
        # PrefillNoCache doesn't need key_cache, but other modes do
        # Only initialize/require cache for modes that actually use it
        if attn_metadata.attn_state != AscendAttentionState.PrefillNoCache:
            # Initialize cache from kv_cache if not already set (for DecodeOnly mode)
            if self.key_cache is None and kv_cache is not None:
                if (
                    isinstance(kv_cache, torch.Tensor)
                    and kv_cache.dim() > 0
                    and kv_cache.shape[0] == 2
                    or isinstance(kv_cache, (list, tuple))
                    and len(kv_cache) >= 2
                ):
                    self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

            if self.key_cache is None:
                raise RuntimeError(
                    f"key_cache is None in _get_fia_params for mode {attn_metadata.attn_state}. kv_cache={kv_cache}"
                )

        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            block_size = 128
            block_table = None
            actual_seq_lengths_kv = attn_metadata.actual_seq_lengths_q
            if self.attn_type == AttentionType.ENCODER_DECODER:
                actual_seq_lengths_kv = torch.cumsum(attn_metadata.seq_lens, dim=0).tolist()
        elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
            batch_size = attn_metadata.seq_lens.shape[0]
            block_table = attn_metadata.block_tables[:batch_size, :]
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        # chunked prefill.
        else:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        return key, value, block_size, block_table, actual_seq_lengths_kv

    def forward_fused_infer_attention(
        self,
        layer: AttentionLayer | None,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        kv_cache=None,
        trace_source: str = "decode_fused",
    ):
        # we inherit ForwardContext in model runner v2, when enable model
        # runner v2, there is not capturing attribute in forward_context,
        # just use getattr to avoid attribute error.
        if _EXTRA_CTX.capturing:
            if self.sinks is not None:
                attn_output, num_tokens = self.full_graph_fia_v2(query, key, value, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
            else:
                attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
        passed_key = key
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(
            key, value, attn_metadata, kv_cache
        )
        if self.enable_hamming_sparse and attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
            reshape_and_cache_kvcomp(attn_metadata.kvcomp_metadata, self.layerIndex, passed_key)
        elif self.enable_hamming_sparse:
            block_table, actual_seq_lengths_kv = get_kvcomp_decode_params(
                self.layerIndex, attn_metadata.kvcomp_metadata, query, passed_key, block_table, actual_seq_lengths_kv
            )
        if (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            and attn_metadata.block_tables is not None
            and attn_metadata.seq_lens_list is not None
        ):
            decode_trace_metadata = self._build_decode_cache_trace_metadata(attn_metadata)
        else:
            decode_trace_metadata = None
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]
        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]
        # Get workspace from cache or calculate it if not present.
        if self.sinks is not None:
            actual_seq_qlen = attn_metadata.actual_seq_lengths_q
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                actual_seq_qlen = torch.tensor([1] * len(attn_metadata.seq_lens_list), dtype=torch.int32).cumsum(dim=0)
            if self.sliding_window is not None:
                sparse_mode = 4
            else:
                sparse_mode = 3
            attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
                query,
                key,
                value,
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
                next_tokens=0,
                atten_mask=attn_metadata.attn_mask,
                sparse_mode=sparse_mode,
                softmax_scale=self.scale,
                block_table=block_table,
                block_size=block_size,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_lengths_kv,
                learnable_sink=self.sinks,
            )
        else:
            if not attn_metadata.causal:
                attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                    query=query,
                    key=key,
                    value=value,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale=self.scale,
                    sparse_mode=0,
                )
            elif self.sliding_window is not None:
                attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                    query=query,
                    key=key,
                    value=value,
                    atten_mask=attn_metadata.attn_mask,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale=self.scale,
                    pre_tokens=self.sliding_window,
                    next_tokens=0,
                    sparse_mode=4,
                )
            else:
                attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                    query=query,
                    key=key,
                    value=value,
                    atten_mask=attn_metadata.attn_mask,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale=self.scale,
                    sparse_mode=3,
                )

            attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        if decode_trace_metadata is not None:
            self._maybe_dump_prefix_cache_trace(
                layer=layer,
                query=query,
                attn_metadata=decode_trace_metadata,
                trace_source=trace_source,
                attn_output=attn_output,
            )
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    def forward_paged_attention(
        self,
        layer: AttentionLayer | None,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        trace_source: str = "decode_paged",
    ) -> torch.Tensor:
        if _EXTRA_CTX.capturing:
            return self.full_graph_pa(query, attn_metadata, output)
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=attn_metadata.block_tables,
            context_lens=attn_metadata.seq_lens,
            out=output,
        )
        self._maybe_dump_prefix_cache_trace(
            layer=layer,
            query=query,
            attn_metadata=attn_metadata,
            trace_source=trace_source,
            attn_output=output,
        )
        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        _: torch.Tensor,
    ) -> torch.Tensor:
        # use default sparse_mode 0 in normal scenario, which means no mask works on it
        # Pad actual_seq_len with 0 when num_tokens > actual_seq_len in TND layout
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        if query.shape[0] > actual_seq_qlen[-1]:
            actual_seq_qlen = actual_seq_qlen + [0]
        return torch_npu.npu_fusion_attention(
            query=query,
            key=key,
            value=value,
            head_num=self.num_heads,
            input_layout="TND",
            scale=self.scale,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_qlen,
        )[0]

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: list[torch.Tensor],
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY):
            return

        if self.key_cache is None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

        DeviceOperator.reshape_and_cache(
            key=key,
            value=value,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            slot_mapping=slot_mapping,
        )

    def reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        if len(kv_cache) > 1:
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            slots = attn_metadata.slot_mapping
            encoder_decoder = self.attn_type == AttentionType.ENCODER_DECODER
            DeviceOperator.reshape_and_cache(
                key=key[: attn_metadata.num_actual_tokens] if not encoder_decoder else key,
                value=value[: attn_metadata.num_actual_tokens] if not encoder_decoder else value,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                # quick fix to make sure slots is int32 for cross attention case.
                # see: https://github.com/vllm-project/vllm/blob/ce88756b967c2c5006746a424c15dd59a284ed8c/vllm/model_executor/layers/attention/cross_attention.py#L117
                slot_mapping=slots[: attn_metadata.num_actual_tokens] if not encoder_decoder else slots.to(torch.int32),
            )
            notify_kv_cache_written()
        return query, key, value, output

    def forward_impl(
        self,
        layer: AttentionLayer | None,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        num_tokens = query.shape[0]
        record_attention_compute_start()
        if (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            and using_paged_attention(num_tokens, self.vllm_config)
            and self.sliding_window is None
        ):
            output = self.forward_paged_attention(layer, query, attn_metadata, output)
        else:
            output = self.forward_fused_infer_attention(
                layer,
                query,
                key,
                value,
                attn_metadata,
                output,
                kv_cache,
            )

        return output

    def _should_use_tree_exact_attention(self, attn_metadata: AscendMetadata) -> bool:
        return (
            attn_metadata.tree_spec_decode_enabled
            and attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill
            and attn_metadata.block_tables is not None
            and attn_metadata.causal
            and not enable_cp()
            and self.sliding_window is None
            and self.sinks is None
            and not getattr(self, "enable_hamming_sparse", False)
            and self.attn_type != AttentionType.ENCODER_DECODER
        )

    def _gather_paged_kv_to_ragged(
        self,
        block_tables: torch.Tensor,
        seq_lens: list[int],
        target_dtype: torch.dtype,
        layer: AttentionLayer | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del target_dtype, layer
        if self.key_cache is None or self.value_cache is None:
            raise RuntimeError("tree exact attention requires initialized KV cache")
        return gather_paged_kv_to_ragged(self.key_cache, self.value_cache, block_tables, seq_lens)

    def _maybe_dump_prefix_cache_trace(
        self,
        *,
        layer: AttentionLayer | None,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        trace_source: str,
        attn_output: torch.Tensor | None = None,
    ) -> None:
        trace_path = os.getenv("VLLM_ASCEND_PREFIX_CACHE_TRACE_PATH")
        if not trace_path or get_tensor_model_parallel_rank() != 0:
            return

        layer_name = getattr(layer, "layer_name", None)
        if not _matches_trace_filters(layer_name, os.getenv("VLLM_ASCEND_PREFIX_CACHE_TRACE_LAYERS")):
            return

        source_filters = os.getenv("VLLM_ASCEND_PREFIX_CACHE_TRACE_SOURCES")
        if source_filters:
            allowed_sources = {item.strip() for item in source_filters.split(",") if item.strip()}
            if allowed_sources and trace_source not in allowed_sources:
                return

        global _PREFIX_CACHE_TRACE_COUNT
        max_steps = int(os.getenv("VLLM_ASCEND_PREFIX_CACHE_TRACE_STEPS", "32"))
        if _PREFIX_CACHE_TRACE_COUNT >= max_steps:
            return

        if attn_metadata.block_tables is None or not attn_metadata.actual_seq_lengths_q:
            return

        batch_size = len(attn_metadata.actual_seq_lengths_q)
        seq_lens = attn_metadata.seq_lens_list[:batch_size]
        gathered_key, gathered_value = self._gather_paged_kv_to_ragged(
            attn_metadata.block_tables,
            seq_lens,
            query.dtype,
            layer,
        )

        block_size = self.vllm_config.cache_config.block_size
        head_dim = max(1, int(os.getenv("VLLM_ASCEND_PREFIX_CACHE_TRACE_HEAD", "8")))
        prefix_tail = max(1, int(os.getenv("VLLM_ASCEND_PREFIX_CACHE_TRACE_TAIL", "4")))
        current_input_ids = getattr(_EXTRA_CTX, "input_ids", None)
        current_input_positions = getattr(_EXTRA_CTX, "positions", None)

        def _vector_head(tensor: torch.Tensor) -> list[float]:
            return tensor[:head_dim].detach().to(torch.float32).cpu().tolist()

        def _slot_id_for_position(req_index: int, position: int) -> int:
            block_index = int(attn_metadata.block_tables[req_index, position // block_size].item())
            return block_index * block_size + (position % block_size)

        def _serialize_row_slice(value, start: int, end: int):
            if value is None:
                return None
            try:
                sliced = value[start:end]
            except Exception:
                return None
            if isinstance(sliced, torch.Tensor):
                return sliced.detach().cpu().tolist()
            try:
                return torch.as_tensor(sliced).detach().cpu().tolist()
            except Exception:
                try:
                    return list(sliced)
                except Exception:
                    return None

        requests = []
        q_start = 0
        kv_start = 0
        for req_idx, q_end in enumerate(attn_metadata.actual_seq_lengths_q):
            q_len = q_end - q_start
            seq_len = seq_lens[req_idx]
            prefix_len = seq_len - q_len
            query_req = query[q_start:q_end]
            output_req = attn_output[q_start:q_end] if attn_output is not None else None
            gathered_k_req = gathered_key[kv_start : kv_start + seq_len]
            gathered_v_req = gathered_value[kv_start : kv_start + seq_len]
            current_k_req = gathered_k_req[prefix_len:seq_len]
            current_v_req = gathered_v_req[prefix_len:seq_len]
            prefix_tail_positions = list(range(max(0, prefix_len - prefix_tail), prefix_len))

            req_payload = {
                "request_index": req_idx,
                "q_len": q_len,
                "seq_len": seq_len,
                "prefix_len": prefix_len,
                "current_positions": list(range(prefix_len, seq_len)),
                "current_input_ids": _serialize_row_slice(current_input_ids, q_start, q_end),
                "current_input_positions": _serialize_row_slice(current_input_positions, q_start, q_end),
                "current_slots": (
                    attn_metadata.slot_mapping[q_start:q_end].detach().cpu().tolist()
                    if attn_metadata.slot_mapping is not None
                    else None
                ),
                "prefix_tail_positions": prefix_tail_positions,
                "prefix_tail_slots": [_slot_id_for_position(req_idx, position) for position in prefix_tail_positions],
                "prefix_tail_k_heads": [_vector_head(gathered_k_req[position, 0]) for position in prefix_tail_positions],
                "prefix_tail_v_heads": [_vector_head(gathered_v_req[position, 0]) for position in prefix_tail_positions],
                "prefix_last_k_head": (_vector_head(gathered_k_req[prefix_len - 1, 0]) if prefix_len > 0 else None),
                "prefix_last_v_head": (_vector_head(gathered_v_req[prefix_len - 1, 0]) if prefix_len > 0 else None),
                "current_q_heads": [_vector_head(query_req[row_idx, 0]) for row_idx in range(q_len)],
                "current_k_heads": [_vector_head(current_k_req[row_idx, 0]) for row_idx in range(q_len)],
                "current_v_heads": [_vector_head(current_v_req[row_idx, 0]) for row_idx in range(q_len)],
                "current_output_heads": (
                    [_vector_head(output_req[row_idx, 0]) for row_idx in range(q_len)] if output_req is not None else None
                ),
            }
            requests.append(req_payload)
            q_start = q_end
            kv_start += seq_len

        payload = {
            "step": _PREFIX_CACHE_TRACE_COUNT,
            "pid": os.getpid(),
            "tp_rank": get_tensor_model_parallel_rank(),
            "label": os.getenv("VLLM_ASCEND_PREFIX_CACHE_TRACE_LABEL"),
            "trace_source": trace_source,
            "layer_name": layer_name,
            "attn_state": getattr(getattr(attn_metadata, "attn_state", None), "name", None),
            "tree_spec_decode_enabled": getattr(attn_metadata, "tree_spec_decode_enabled", None),
            "num_actual_tokens": getattr(attn_metadata, "num_actual_tokens", None),
            "requests": requests,
        }
        with open(trace_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        _PREFIX_CACHE_TRACE_COUNT += 1

    def _build_tree_root_decode_metadata(
        self,
        attn_metadata: AscendMetadata,
        active_req_indices: list[int],
        root_seq_lens: list[int],
        row0_slot_mapping: torch.Tensor | None = None,
    ) -> AscendMetadata:
        decode_metadata = AscendMetadata()
        device = attn_metadata.seq_lens.device
        mask_builder = AttentionMaskBuilder(device)
        decode_metadata.attn_state = AscendAttentionState.DecodeOnly
        decode_metadata.block_tables = attn_metadata.block_tables[active_req_indices]
        decode_metadata.seq_lens = torch.tensor(
            root_seq_lens,
            dtype=attn_metadata.seq_lens.dtype,
            device=device,
        )
        decode_metadata.seq_lens_cpu = decode_metadata.seq_lens
        decode_metadata.seq_lens_list = root_seq_lens
        decode_metadata.actual_seq_lengths_q = list(range(1, len(active_req_indices) + 1))
        decode_metadata.query_start_loc = torch.arange(
            len(active_req_indices) + 1,
            dtype=torch.int32,
            device=device,
        )
        decode_metadata.max_query_len = 1
        decode_metadata.slot_mapping = row0_slot_mapping
        decode_metadata.attn_mask = mask_builder.get_attention_mask(
            attn_metadata.causal,
            self.vllm_config.model_config,
        )
        decode_metadata.causal = attn_metadata.causal
        decode_metadata.tree_spec_decode_enabled = False
        decode_metadata.num_actual_tokens = len(root_seq_lens)
        decode_metadata.num_decode_tokens = len(root_seq_lens)
        decode_metadata.num_decodes = len(root_seq_lens)
        decode_metadata.num_prefills = 0
        return decode_metadata

    def _build_decode_cache_trace_metadata(self, attn_metadata: AscendMetadata) -> AscendMetadata:
        trace_metadata = AscendMetadata()
        trace_metadata.attn_state = AscendAttentionState.DecodeOnly
        trace_metadata.block_tables = attn_metadata.block_tables
        trace_metadata.seq_lens = attn_metadata.seq_lens
        trace_metadata.seq_lens_cpu = attn_metadata.seq_lens_cpu
        trace_metadata.seq_lens_list = attn_metadata.seq_lens_list
        trace_metadata.actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        trace_metadata.slot_mapping = attn_metadata.slot_mapping
        trace_metadata.causal = attn_metadata.causal
        trace_metadata.tree_spec_decode_enabled = attn_metadata.tree_spec_decode_enabled
        trace_metadata.num_actual_tokens = attn_metadata.num_actual_tokens
        return trace_metadata

    def forward_tree_exact_attention(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]
        current_key = key[:num_tokens]
        current_value = value[:num_tokens]
        batch_size = len(attn_metadata.actual_seq_lengths_q)
        seq_lens = attn_metadata.seq_lens_list[:batch_size]
        tree_mask = attn_metadata.attn_mask
        max_q_len = max(
            q_end - (attn_metadata.actual_seq_lengths_q[idx - 1] if idx > 0 else 0)
            for idx, q_end in enumerate(attn_metadata.actual_seq_lengths_q)
        )
        if tree_mask is None or tree_mask.shape[0] < max_q_len or tree_mask.shape[1] < max_q_len:
            raise RuntimeError("tree exact attention requires a tree-specific attention mask")

        gathered_key, gathered_value = self._gather_paged_kv_to_ragged(
            attn_metadata.block_tables,
            seq_lens,
            query.dtype,
            layer,
        )
        attn_output = run_exact_tree_attention(
            query=query,
            key=gathered_key,
            value=gathered_value,
            seq_lens=seq_lens,
            actual_seq_lengths_q=attn_metadata.actual_seq_lengths_q,
            tree_mask=tree_mask,
            scale=self.scale,
            num_queries_per_kv=self.num_queries_per_kv,
        )

        row0_token_indices: list[int] = []
        active_req_indices: list[int] = []
        root_seq_lens: list[int] = []
        q_start = 0
        for req_idx, q_end in enumerate(attn_metadata.actual_seq_lengths_q):
            q_len = q_end - q_start
            if q_len > 0:
                active_req_indices.append(req_idx)
                row0_token_indices.append(q_start)
                root_seq_lens.append(seq_lens[req_idx] - q_len + 1)
            q_start = q_end

        if active_req_indices:
            row0_query = query[row0_token_indices]
            row0_key = current_key[row0_token_indices]
            row0_value = current_value[row0_token_indices]
            row0_output = torch.empty_like(row0_query)
            row0_slot_mapping = (
                attn_metadata.slot_mapping[row0_token_indices]
                if attn_metadata.slot_mapping is not None
                else None
            )
            row0_metadata = self._build_tree_root_decode_metadata(
                attn_metadata,
                active_req_indices,
                root_seq_lens,
                row0_slot_mapping,
            )
            if (
                row0_metadata.attn_state == AscendAttentionState.DecodeOnly
                and using_paged_attention(row0_query.shape[0], self.vllm_config)
                and self.sliding_window is None
            ):
                self.forward_paged_attention(
                    layer,
                    row0_query,
                    row0_metadata,
                    row0_output,
                    trace_source="tree_row0_paged",
                )
            else:
                self.forward_fused_infer_attention(
                    layer,
                    row0_query,
                    row0_key,
                    row0_value,
                    row0_metadata,
                    row0_output,
                    trace_source="tree_row0_fused",
                )
            attn_output[row0_token_indices] = row0_output

        debug_path = os.getenv("VLLM_ASCEND_TREE_EXACT_DEBUG_PATH")
        if debug_path:
            q_start = 0
            kv_start = 0
            current_start = 0
            debug_rows = []
            block_size = self.vllm_config.cache_config.block_size
            debug_tail = max(1, int(os.getenv("VLLM_ASCEND_TREE_EXACT_DEBUG_TAIL", "3")))
            debug_head = max(
                1,
                int(
                    os.getenv(
                        "VLLM_ASCEND_TREE_EXACT_DEBUG_HEAD",
                        os.getenv("VLLM_ASCEND_PREFIX_CACHE_TRACE_HEAD", "8"),
                    )
                ),
            )

            def _vector_head(tensor: torch.Tensor, *, limit: int = debug_head) -> list[float]:
                return tensor[:limit].detach().to(torch.float32).cpu().tolist()

            def _slot_id_for_position(req_index: int, position: int) -> int:
                block_index = int(attn_metadata.block_tables[req_index, position // block_size].item())
                return block_index * block_size + (position % block_size)

            for req_idx, q_end in enumerate(attn_metadata.actual_seq_lengths_q):
                q_len = q_end - q_start
                seq_len = seq_lens[req_idx]
                prefix_len = seq_len - q_len
                gathered_k_req = gathered_key[kv_start : kv_start + seq_len]
                gathered_v_req = gathered_value[kv_start : kv_start + seq_len]
                direct_k_req = current_key[current_start : current_start + q_len]
                direct_v_req = current_value[current_start : current_start + q_len]
                gathered_cur_k = gathered_k_req[prefix_len:seq_len]
                gathered_cur_v = gathered_v_req[prefix_len:seq_len]

                req_payload = {
                    "request_index": req_idx,
                    "q_len": q_len,
                    "seq_len": seq_len,
                    "prefix_len": prefix_len,
                    "current_slots": attn_metadata.slot_mapping[q_start:q_end].detach().cpu().tolist(),
                    "current_k_max_diff": float((gathered_cur_k - direct_k_req).abs().max().item()),
                    "current_v_max_diff": float((gathered_cur_v - direct_v_req).abs().max().item()),
                }

                if prefix_len > 0:
                    prefix_tail_positions = list(range(max(0, prefix_len - debug_tail), prefix_len))
                    req_payload["prefix_tail_slots"] = [
                        _slot_id_for_position(req_idx, position) for position in prefix_tail_positions
                    ]
                    req_payload["prefix_tail_k_heads"] = [
                        _vector_head(gathered_k_req[position, 0]) for position in prefix_tail_positions
                    ]
                    req_payload["prefix_tail_v_heads"] = [
                        _vector_head(gathered_v_req[position, 0]) for position in prefix_tail_positions
                    ]
                    req_payload["prefix_last_k_head"] = _vector_head(gathered_k_req[prefix_len - 1, 0])
                    req_payload["prefix_last_v_head"] = _vector_head(gathered_v_req[prefix_len - 1, 0])
                else:
                    req_payload["prefix_tail_slots"] = []
                    req_payload["prefix_tail_k_heads"] = []
                    req_payload["prefix_tail_v_heads"] = []
                    req_payload["prefix_last_k_head"] = None
                    req_payload["prefix_last_v_head"] = None

                req_payload["current_k_heads"] = [
                    _vector_head(direct_k_req[row_idx, 0]) for row_idx in range(q_len)
                ]
                req_payload["current_v_heads"] = [
                    _vector_head(direct_v_req[row_idx, 0]) for row_idx in range(q_len)
                ]
                req_payload["current_q_heads"] = [
                    _vector_head(query[q_start + row_idx, 0]) for row_idx in range(q_len)
                ]
                req_payload["current_output_heads"] = [
                    _vector_head(attn_output[q_start + row_idx, 0]) for row_idx in range(q_len)
                ]

                if q_len > 0:
                    row0_key = torch.cat((gathered_k_req[:prefix_len], direct_k_req[:1]), dim=0)
                    row0_value = torch.cat((gathered_v_req[:prefix_len], direct_v_req[:1]), dim=0)
                    row0_only_output = run_exact_tree_attention(
                        query=query[q_start : q_start + 1],
                        key=row0_key,
                        value=row0_value,
                        seq_lens=[prefix_len + 1],
                        actual_seq_lengths_q=[1],
                        tree_mask=torch.zeros((1, 1), dtype=tree_mask.dtype, device=tree_mask.device),
                        scale=self.scale,
                        num_queries_per_kv=self.num_queries_per_kv,
                    )
                    req_payload["row0_output_max_diff_vs_isolated"] = float(
                        (attn_output[q_start : q_start + 1] - row0_only_output).abs().max().item()
                    )
                debug_rows.append(req_payload)

                q_start = q_end
                kv_start += seq_len
                current_start += q_len

            maybe_dump_tree_exact_debug(
                {
                    "pid": os.getpid(),
                    "tp_rank": get_tensor_model_parallel_rank(),
                    "layer_name": getattr(layer, "layer_name", None),
                    "rows": debug_rows,
                }
            )
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    def _build_tree_non_root_indices(
        self,
        actual_seq_lengths_q: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        indices: list[int] = []
        q_start = 0
        for q_end in actual_seq_lengths_q:
            q_len = q_end - q_start
            if q_len > 1:
                indices.extend(range(q_start + 1, q_end))
            q_start = q_end
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _run_tree_row0_override(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        attn_metadata: AscendMetadata,
        seq_lens: list[int],
        output: torch.Tensor,
    ) -> tuple[list[int], float]:
        row0_token_indices: list[int] = []
        active_req_indices: list[int] = []
        root_seq_lens: list[int] = []
        q_start = 0
        for req_idx, q_end in enumerate(attn_metadata.actual_seq_lengths_q):
            q_len = q_end - q_start
            if q_len > 0:
                active_req_indices.append(req_idx)
                row0_token_indices.append(q_start)
                root_seq_lens.append(seq_lens[req_idx] - q_len + 1)
            q_start = q_end

        if not active_req_indices:
            return row0_token_indices, 0.0

        row0_start = time.perf_counter()
        row0_query = query[row0_token_indices]
        row0_key = current_key[row0_token_indices]
        row0_value = current_value[row0_token_indices]
        row0_output = torch.empty_like(row0_query)
        row0_slot_mapping = (
            attn_metadata.slot_mapping[row0_token_indices] if attn_metadata.slot_mapping is not None else None
        )
        row0_metadata = self._build_tree_root_decode_metadata(
            attn_metadata,
            active_req_indices,
            root_seq_lens,
            row0_slot_mapping,
        )
        if (
            row0_metadata.attn_state == AscendAttentionState.DecodeOnly
            and using_paged_attention(row0_query.shape[0], self.vllm_config)
            and self.sliding_window is None
        ):
            self.forward_paged_attention(
                layer,
                row0_query,
                row0_metadata,
                row0_output,
                trace_source="tree_row0_paged",
            )
        else:
            self.forward_fused_infer_attention(
                layer,
                row0_query,
                row0_key,
                row0_value,
                row0_metadata,
                row0_output,
                trace_source="tree_row0_fused",
            )
        output[row0_token_indices] = row0_output
        return row0_token_indices, (time.perf_counter() - row0_start) * 1000.0

    def _get_or_build_tree_non_root_plan(
        self,
        attn_metadata: AscendMetadata,
        seq_lens: list[int],
        tree_mask: torch.Tensor,
        device: torch.device,
    ) -> tuple[object, float]:
        if attn_metadata.tree_fast_non_root_plan is not None:
            return attn_metadata.tree_fast_non_root_plan, 0.0

        if self.key_cache is None or attn_metadata.block_tables is None:
            raise RuntimeError("tree fast attention requires initialized KV cache and block tables")

        plan_start = time.perf_counter()
        plan_key = (
            int(attn_metadata.block_tables.data_ptr()),
            tuple(seq_lens),
            tuple(attn_metadata.actual_seq_lengths_q),
            int(tree_mask.data_ptr()),
            int(self.key_cache.shape[1]),
            str(device),
        )
        cached_plan = attn_metadata.tree_fast_non_root_plan
        if attn_metadata.tree_fast_non_root_plan_key != plan_key or cached_plan is None:
            cached_plan = build_tree_paged_plan(
                block_tables=attn_metadata.block_tables,
                seq_lens=seq_lens,
                actual_seq_lengths_q=attn_metadata.actual_seq_lengths_q,
                tree_mask=tree_mask,
                device=device,
                block_size=self.key_cache.shape[1],
            )
            attn_metadata.tree_fast_non_root_plan = cached_plan
            attn_metadata.tree_fast_non_root_plan_key = plan_key
        return cached_plan, (time.perf_counter() - plan_start) * 1000.0

    def _get_or_build_tree_all_rows_plan(
        self,
        attn_metadata: AscendMetadata,
        seq_lens: list[int],
        tree_mask: torch.Tensor,
        device: torch.device,
    ) -> tuple[object, float]:
        if attn_metadata.tree_fast_all_rows_plan is not None:
            return attn_metadata.tree_fast_all_rows_plan, 0.0

        if self.key_cache is None or attn_metadata.block_tables is None:
            raise RuntimeError("tree fast attention requires initialized KV cache and block tables")

        plan_start = time.perf_counter()
        cached_plan = build_tree_paged_plan(
            block_tables=attn_metadata.block_tables,
            seq_lens=seq_lens,
            actual_seq_lengths_q=attn_metadata.actual_seq_lengths_q,
            tree_mask=tree_mask,
            device=device,
            block_size=self.key_cache.shape[1],
            include_row0=True,
        )
        attn_metadata.tree_fast_all_rows_plan = cached_plan
        return cached_plan, (time.perf_counter() - plan_start) * 1000.0

    def _get_tree_cache_fia_tensors(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self.key_cache is None or self.value_cache is None:
            raise RuntimeError("tree fast attention requires initialized KV cache")
        num_block, block_size, _, _ = self.key_cache.shape
        return self.key_cache.view(num_block, block_size, -1), self.value_cache.view(num_block, block_size, -1), block_size

    def _forward_tree_fast_attention_legacy(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        seq_lens: list[int],
        tree_mask: torch.Tensor,
        total_start: float,
    ) -> torch.Tensor:
        row0_token_indices, row0_ms = self._run_tree_row0_override(
            layer,
            query,
            current_key,
            current_value,
            attn_metadata,
            seq_lens,
            output,
        )

        non_root_plan, non_root_plan_ms = self._get_or_build_tree_non_root_plan(
            attn_metadata,
            seq_lens,
            tree_mask,
            query.device,
        )

        non_root_start = time.perf_counter()
        non_root_indices, non_root_output, non_root_stats = run_fast_tree_attention_from_paged_cache(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            plan=non_root_plan,
            scale=self.scale,
            num_queries_per_kv=self.num_queries_per_kv,
        )
        non_root_ms = (time.perf_counter() - non_root_start) * 1000.0
        if non_root_indices.numel() > 0:
            output[non_root_indices] = non_root_output[: non_root_indices.numel()]
        _maybe_dump_tree_fast_profile(
            {
                "layer_name": getattr(layer, "layer_name", None),
                "num_tokens": query.shape[0],
                "seq_lens_list": seq_lens,
                "actual_seq_lengths_q": attn_metadata.actual_seq_lengths_q,
                "row0_count": len(row0_token_indices),
                "non_root_count": int(non_root_indices.numel()),
                "gather_ms": 0.0,
                "row0_ms": row0_ms,
                "non_root_ms": non_root_ms,
                "total_ms": (time.perf_counter() - total_start) * 1000.0,
                "tree_impl": "batched_legacy",
                "non_root_stats": {
                    **non_root_stats,
                    "plan_ms": non_root_plan_ms,
                },
            }
        )
        return output

    def _forward_tree_fast_attention_all_rows(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        seq_lens: list[int],
        tree_mask: torch.Tensor,
        total_start: float,
    ) -> torch.Tensor:
        del current_key, current_value
        all_rows_plan, plan_ms = self._get_or_build_tree_all_rows_plan(
            attn_metadata,
            seq_lens,
            tree_mask,
            query.device,
        )

        all_rows_start = time.perf_counter()
        output_indices, attn_output, attn_stats = run_fast_tree_attention_from_paged_cache(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            plan=all_rows_plan,
            scale=self.scale,
            num_queries_per_kv=self.num_queries_per_kv,
        )
        all_rows_ms = (time.perf_counter() - all_rows_start) * 1000.0
        if output_indices.numel() > 0:
            output[output_indices] = attn_output[: output_indices.numel()]
        _maybe_dump_tree_fast_profile(
            {
                "layer_name": getattr(layer, "layer_name", None),
                "num_tokens": query.shape[0],
                "seq_lens_list": seq_lens,
                "actual_seq_lengths_q": attn_metadata.actual_seq_lengths_q,
                "row0_count": len(attn_metadata.actual_seq_lengths_q),
                "non_root_count": int(output_indices.numel()),
                "gather_ms": 0.0,
                "row0_ms": 0.0,
                "non_root_ms": all_rows_ms,
                "total_ms": (time.perf_counter() - total_start) * 1000.0,
                "tree_impl": "batched",
                "all_rows_unified": True,
                "non_root_stats": {
                    **attn_stats,
                    "plan_ms": plan_ms,
                },
            }
        )
        return output

    def _forward_tree_fast_attention_split(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        seq_lens: list[int],
        tree_mask: torch.Tensor,
        total_start: float,
    ) -> torch.Tensor:
        row0_token_indices, row0_ms = self._run_tree_row0_override(
            layer,
            query,
            current_key,
            current_value,
            attn_metadata,
            seq_lens,
            output,
        )

        plan, plan_stats = build_tree_split_plan(
            actual_seq_lengths_q=attn_metadata.actual_seq_lengths_q,
            seq_lens=seq_lens,
            tree_mask=tree_mask,
            block_tables=attn_metadata.block_tables,
            device=query.device,
        )
        if plan.num_rows == 0:
            _maybe_dump_tree_fast_profile(
                {
                    "layer_name": getattr(layer, "layer_name", None),
                    "num_tokens": query.shape[0],
                    "seq_lens_list": seq_lens,
                    "actual_seq_lengths_q": attn_metadata.actual_seq_lengths_q,
                    "row0_count": len(row0_token_indices),
                    "non_root_count": 0,
                    "gather_ms": 0.0,
                    "row0_ms": row0_ms,
                    "non_root_ms": 0.0,
                    "total_ms": (time.perf_counter() - total_start) * 1000.0,
                    "tree_impl": "batched_split",
                }
            )
            return output

        packed_query = query.index_select(0, plan.output_indices).contiguous()

        prefix_start = time.perf_counter()
        cache_key, cache_value, block_size = self._get_tree_cache_fia_tensors()
        prefix_output, prefix_lse = torch_npu.npu_fused_infer_attention_score(
            query=packed_query,
            key=cache_key,
            value=cache_value,
            block_table=plan.prefix_block_tables,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=plan.actual_seq_lengths_q,
            actual_seq_lengths_kv=plan.prefix_actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=0,
            softmax_lse_flag=True,
        )
        prefix_ms = (time.perf_counter() - prefix_start) * 1000.0
        prefix_output = prefix_output.view(plan.num_rows, self.num_heads, self.head_size)
        prefix_lse = prefix_lse.view(plan.num_rows, self.num_heads, 1)

        current_output, current_lse, current_stats = run_split_current_tree_attention(
            packed_query=packed_query,
            current_key=current_key,
            current_value=current_value,
            plan=plan,
            scale=self.scale,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )

        merge_start = time.perf_counter()
        merged_output = merge_split_tree_attention_outputs(
            prefix_output=prefix_output,
            prefix_lse=prefix_lse,
            current_output=current_output,
            current_lse=current_lse,
        )
        merge_ms = (time.perf_counter() - merge_start) * 1000.0

        if os.getenv("VLLM_ASCEND_TREE_SPLIT_COMPARE_PATH"):
            torch_lse = torch.stack([prefix_lse.to(torch.float32), current_lse.to(torch.float32)], dim=0)
            torch_out = torch.stack([prefix_output.to(torch.float32), current_output.to(torch.float32)], dim=0)
            torch_merged = torch.sum(
                torch.exp(torch_lse - torch.logsumexp(torch_lse, dim=0, keepdim=True)) * torch_out,
                dim=0,
            ).to(merged_output.dtype)

            gathered_key, gathered_value = self._gather_paged_kv_to_ragged(
                attn_metadata.block_tables,
                seq_lens,
                query.dtype,
                layer,
            )
            legacy_indices, legacy_output, _ = run_fast_tree_attention(
                query=query,
                key=gathered_key,
                value=gathered_value,
                seq_lens=seq_lens,
                actual_seq_lengths_q=attn_metadata.actual_seq_lengths_q,
                tree_mask=tree_mask,
                scale=self.scale,
                num_queries_per_kv=self.num_queries_per_kv,
            )
            legacy_trimmed = legacy_output[: plan.num_rows]
            _maybe_dump_tree_split_compare(
                {
                    "layer_name": getattr(layer, "layer_name", None),
                    "num_tokens": query.shape[0],
                    "seq_lens_list": seq_lens,
                    "actual_seq_lengths_q": attn_metadata.actual_seq_lengths_q,
                    "output_indices_match": bool(torch.equal(legacy_indices, plan.output_indices)),
                    "split_vs_legacy_max_diff": float((merged_output - legacy_trimmed).abs().max().item()),
                    "split_vs_legacy_mean_diff": float((merged_output - legacy_trimmed).abs().mean().item()),
                    "split_vs_torch_merge_max_diff": float((merged_output - torch_merged).abs().max().item()),
                    "split_vs_torch_merge_mean_diff": float((merged_output - torch_merged).abs().mean().item()),
                    "prefix_only_vs_legacy_max_diff": float((prefix_output - legacy_trimmed).abs().max().item()),
                    "current_only_vs_legacy_max_diff": float((current_output - legacy_trimmed).abs().max().item()),
                    "prefix_lse_min": float(prefix_lse.min().item()),
                    "prefix_lse_max": float(prefix_lse.max().item()),
                    "current_lse_min": float(current_lse.min().item()),
                    "current_lse_max": float(current_lse.max().item()),
                }
            )

        output[plan.output_indices] = merged_output[: plan.num_rows]
        non_root_ms = prefix_ms + current_stats["total_ms"] + merge_ms
        _maybe_dump_tree_fast_profile(
            {
                "layer_name": getattr(layer, "layer_name", None),
                "num_tokens": query.shape[0],
                "seq_lens_list": seq_lens,
                "actual_seq_lengths_q": attn_metadata.actual_seq_lengths_q,
                "row0_count": len(row0_token_indices),
                "non_root_count": plan.num_rows,
                "gather_ms": 0.0,
                "row0_ms": row0_ms,
                "non_root_ms": non_root_ms,
                "total_ms": (time.perf_counter() - total_start) * 1000.0,
                "tree_impl": "batched_split",
                "non_root_stats": {
                    **plan_stats,
                    "pack_ms": current_stats["pack_ms"],
                    "fia_ms": prefix_ms + current_stats["fia_ms"],
                    "prefix_fia_ms": prefix_ms,
                    "current_fia_ms": current_stats["fia_ms"],
                    "merge_ms": merge_ms,
                    "total_ms": non_root_ms,
                },
            }
        )
        return output

    def forward_tree_fast_attention(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        tree_impl: str = "batched",
    ) -> torch.Tensor:
        total_start = time.perf_counter()
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]
        current_key = key[:num_tokens]
        current_value = value[:num_tokens]
        batch_size = len(attn_metadata.actual_seq_lengths_q)
        seq_lens = attn_metadata.seq_lens_list[:batch_size]
        tree_mask = attn_metadata.attn_mask
        max_q_len = max(
            q_end - (attn_metadata.actual_seq_lengths_q[idx - 1] if idx > 0 else 0)
            for idx, q_end in enumerate(attn_metadata.actual_seq_lengths_q)
        )
        if tree_mask is None or tree_mask.shape[0] < max_q_len or tree_mask.shape[1] < max_q_len:
            raise RuntimeError("tree fast attention requires a tree-specific attention mask")
        tree_impl = resolve_tree_fast_impl(tree_impl)
        if tree_impl == "batched":
            return self._forward_tree_fast_attention_all_rows(
                layer,
                query,
                current_key,
                current_value,
                attn_metadata,
                output,
                seq_lens,
                tree_mask,
                total_start,
            )
        if tree_impl == "batched_legacy":
            return self._forward_tree_fast_attention_legacy(
                layer,
                query,
                current_key,
                current_value,
                attn_metadata,
                output,
                seq_lens,
                tree_mask,
                total_start,
            )
        if tree_impl == "batched_split":
            return self._forward_tree_fast_attention_split(
                layer,
                query,
                current_key,
                current_value,
                attn_metadata,
                output,
                seq_lens,
                tree_mask,
                total_start,
            )
        raise ValueError(f"Unsupported tree fast attention impl '{tree_impl}'")

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        if self.enable_hamming_sparse:
            self.layerIndex = int(layer.layer_name.split(".")[2])

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendAttentionBackendImpl")

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        # Initialize key_cache and value_cache from kv_cache if not already set.
        # This is needed for DecodeOnly mode where key/value are None but we still
        # need access to the cache for attention computation.
        if self.key_cache is None and kv_cache is not None:
            if (
                isinstance(kv_cache, torch.Tensor)
                and kv_cache.dim() > 0
                and kv_cache.shape[0] == 2
                or isinstance(kv_cache, (list, tuple))
                and len(kv_cache) >= 2
            ):
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

        output_padded = None
        if key is not None and value is not None:
            output_padded = output
            query, key, value, output_padded = self.reshape_and_cache(
                query, key, value, kv_cache, attn_metadata, output
            )
        # pooling model branch
        if attn_metadata.model_runner_type == "pooling" and not attn_metadata.causal:
            attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output
        use_tree_exact_attention = self._should_use_tree_exact_attention(attn_metadata)
        if use_tree_exact_attention and attn_metadata.tree_fast_mixed_batch_fallback:
            use_tree_exact_attention = False
        tree_impl = resolve_tree_fast_impl(os.getenv("VLLM_ASCEND_TREE_FAST_IMPL", "batched")) if use_tree_exact_attention else None
        if attn_metadata.tree_spec_decode_enabled:
            _maybe_dump_tree_attn_trace(
                layer_name=getattr(layer, "layer_name", None),
                impl_name=type(self).__name__,
                path="exact" if use_tree_exact_attention else "fallback",
                attn_metadata=attn_metadata,
                extra={
                    "causal": attn_metadata.causal,
                    "sliding_window": self.sliding_window,
                    "has_sinks": self.sinks is not None,
                    "enable_hamming_sparse": getattr(self, "enable_hamming_sparse", False),
                    "attn_type": getattr(self.attn_type, "name", str(self.attn_type)),
                    "selected_tree_impl": tree_impl,
                },
            )
        if use_tree_exact_attention:
            if tree_impl == "reference":
                return self.forward_tree_exact_attention(layer, query, key, value, attn_metadata, output)
            return self.forward_tree_fast_attention(
                layer,
                query,
                key,
                value,
                attn_metadata,
                output,
                tree_impl=tree_impl,
            )
        if output_padded is not None:
            attn_output = self.forward_impl(layer, query, key, value, kv_cache, attn_metadata, output_padded)
        else:
            attn_output = self.forward_impl(layer, query, key, value, kv_cache, attn_metadata, output)
        output[:num_tokens] = attn_output[:num_tokens]
        return output


class AscendC8AttentionBackendImpl(AscendAttentionBackendImpl):
    """Attention backend implementation for INT8 KV cache (C8/QuaRot) models.

    This subclass handles static per-channel INT8 KV cache quantization.
    It is activated via class surgery in AscendC8KVCacheAttentionMethod.create_weights
    (vllm_ascend/quantization/methods/kv_c8.py)
    so that C8 attention layers automatically use this forward path.
    """

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendC8AttentionBackendImpl")

        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        self._prepare_c8_scales(layer, query.device)
        float_key, float_value = None, None
        if self.vllm_config.kv_transfer_config is None:
            if key is not None and value is not None:
                if attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
                    float_key, float_value = key, value
                key, value = self._quantize_kv_to_int8(key, value, layer, attn_metadata.num_actual_tokens)
                query, key, value, _ = self._reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)
            # pooling model branch
            if attn_metadata.model_runner_type == "pooling":
                attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
            if self._should_use_tree_exact_attention(attn_metadata):
                return self.forward_tree_exact_attention(layer, query, attn_metadata, output)
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                if _EXTRA_CTX.capturing:
                    attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output, layer)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                return self._forward_c8_decode(query, attn_metadata, output, layer)
            elif attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
                return self._forward_c8_chunked_prefill(query, float_key, float_value, attn_metadata, output, layer)
            else:
                return self._forward_c8_fused_infer_attention(
                    query,
                    float_key if float_key is not None else key,
                    float_value if float_value is not None else value,
                    attn_metadata,
                    output,
                    layer,
                )
        else:
            if attn_metadata.attn_state != AscendAttentionState.DecodeOnly and self.is_kv_producer:
                output_padded = None
                if key is not None and value is not None:
                    output_padded = output
                    query, key, value, output_padded = self.reshape_and_cache(
                        query, key, value, kv_cache, attn_metadata, output
                    )
                # pooling model branch
                if attn_metadata.model_runner_type == "pooling":
                    attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                if self._should_use_tree_exact_attention(attn_metadata):
                    return self.forward_tree_exact_attention(layer, query, attn_metadata, output)
                if output_padded is not None:
                    attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output_padded)
                else:
                    attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
            elif not self.is_kv_producer:
                if key is not None and value is not None:
                    key, value = self._quantize_kv_to_int8(key, value, layer, attn_metadata.num_actual_tokens)
                    query, key, value, _ = self._reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)
                # pooling model branch
                if attn_metadata.model_runner_type == "pooling":
                    attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                if self._should_use_tree_exact_attention(attn_metadata):
                    return self.forward_tree_exact_attention(layer, query, attn_metadata, output)
                if _EXTRA_CTX.capturing:
                    attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output, layer)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                    return self._forward_c8_decode(query, attn_metadata, output, layer)

    def _nz_5d_view(self, cache: torch.Tensor, block_size: int) -> torch.Tensor:
        """View a KV cache tensor in NZ 5D layout: (num_blocks, num_kv_heads, head_size//nz, block_size, nz)."""
        NZ_FMT_LAST_DIM = 32
        return cache.view(-1, self.num_kv_heads, self.head_size // NZ_FMT_LAST_DIM, block_size, NZ_FMT_LAST_DIM)

    def _prepare_c8_scales(self, layer: AttentionLayer, device: torch.device) -> None:
        """Shard per-channel C8 scales/offsets to this TP rank and pre-compute
        BF16 BNSD antiquant tensors for FIA V1 decode fast path.
        """
        if hasattr(layer, "_c8_scales_prepared"):
            return

        def _shard_and_reshape(raw: torch.Tensor) -> torch.Tensor:
            if raw.numel() == 1:
                return raw.to(device=device)
            expected = self.num_kv_heads * self.head_size
            if raw.numel() != expected:
                total_kv_heads = raw.numel() // self.head_size
                tp_rank = get_tensor_model_parallel_rank()
                tp_size = get_tensor_model_parallel_world_size()
                kv_head_start = tp_rank * total_kv_heads // tp_size
                raw = raw.view(total_kv_heads, self.head_size)[
                    kv_head_start : kv_head_start + self.num_kv_heads
                ].contiguous()
            return raw.view(1, self.num_kv_heads, self.head_size).to(device=device)

        layer._c8_k_scale = _shard_and_reshape(layer.k_cache_scale.data)
        layer._c8_k_offset = _shard_and_reshape(layer.k_cache_offset.data)
        layer._c8_v_scale = _shard_and_reshape(layer.v_cache_scale.data)
        layer._c8_v_offset = _shard_and_reshape(layer.v_cache_offset.data)

        layer._c8_k_inv_scale = 1.0 / layer._c8_k_scale
        layer._c8_v_inv_scale = 1.0 / layer._c8_v_scale

        nz_bnsd = (self.num_kv_heads, 1, self.head_size)
        layer._c8_k_aq_scale_nz_bnsd = layer._c8_k_scale.view(nz_bnsd).contiguous()
        layer._c8_v_aq_scale_nz_bnsd = layer._c8_v_scale.view(nz_bnsd).contiguous()

        layer._c8_scales_prepared = True

    def _dequant_paged_kv_to_dense(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: list,
        target_dtype: torch.dtype,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather paged INT8 KV blocks and dequantize."""
        batch_size = block_table.shape[0]
        max_blocks_per_seq = block_table.shape[1]

        # NZ 5D view: (num_blocks, num_kv_heads, head_size//nz, block_size, nz)
        block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
        max_tokens_padded = max_blocks_per_seq * block_size

        flat_ids = block_table.reshape(-1)
        key_nz = self._nz_5d_view(key, block_size)
        value_nz = self._nz_5d_view(value, block_size)

        # Gather: (batch*max_blocks, H, D//nz, S, nz)
        gathered_k = key_nz[flat_ids]
        gathered_v = value_nz[flat_ids]
        # NZ→ND conversion: permute (S, H, D//nz, nz) → reshape (S, H, D)
        gathered_k = (
            gathered_k.permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(batch_size, max_tokens_padded, self.num_kv_heads, self.head_size)
        )
        gathered_v = (
            gathered_v.permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(batch_size, max_tokens_padded, self.num_kv_heads, self.head_size)
        )

        seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=key.device)
        positions = torch.arange(max_tokens_padded, dtype=torch.long, device=key.device)
        valid_mask = (positions.unsqueeze(0) < seq_lens_t.unsqueeze(1)).view(-1)

        dense_k = gathered_k.view(-1, self.num_kv_heads, self.head_size)[valid_mask]
        dense_v = gathered_v.view(-1, self.num_kv_heads, self.head_size)[valid_mask]

        # Scale-only dequant for NZ (symmetric)
        dense_k = dense_k.to(target_dtype) * layer._c8_k_scale
        dense_v = dense_v.to(target_dtype) * layer._c8_v_scale
        return dense_k, dense_v

    def _gather_paged_kv_to_ragged(
        self,
        block_tables: torch.Tensor,
        seq_lens: list[int],
        target_dtype: torch.dtype,
        layer: AttentionLayer | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer is None:
            raise RuntimeError("C8 tree exact attention requires layer context for dequantization")
        if self.key_cache is None or self.value_cache is None:
            raise RuntimeError("tree exact attention requires initialized KV cache")
        return self._dequant_paged_kv_to_dense(
            self.key_cache,
            self.value_cache,
            block_tables[: len(seq_lens)],
            seq_lens,
            target_dtype,
            layer,
        )

    def _quantize_kv_to_int8(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: AttentionLayer,
        num_actual_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize K/V from float to INT8 using static per-channel C8 scales."""
        actual_key = key[:num_actual_tokens]
        actual_value = value[:num_actual_tokens]

        k_int8 = torch.clamp(
            torch.round(actual_key * layer._c8_k_inv_scale + layer._c8_k_offset),
            -128,
            127,
        ).to(torch.int8)
        v_int8 = torch.clamp(
            torch.round(actual_value * layer._c8_v_inv_scale + layer._c8_v_offset),
            -128,
            127,
        ).to(torch.int8)
        return k_int8, v_int8

    def _forward_c8_decode(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        """C8 decode via FIA V1 BNSD with native paged INT8 KV + perchannel antiquant."""
        num_block, block_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
        assert block_size % 32 == 0, f"C8 INT8 KV cache requires block_size to be a multiple of 32, got {block_size}"
        batch_size = len(attn_metadata.seq_lens_list)

        key = self._nz_5d_view(self.key_cache, block_size)
        value = self._nz_5d_view(self.value_cache, block_size)

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query[:batch_size].unsqueeze(2),
            key,
            value,
            key_antiquant_scale=layer._c8_k_aq_scale_nz_bnsd,
            value_antiquant_scale=layer._c8_v_aq_scale_nz_bnsd,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BNSD",
            scale=self.scale,
            block_size=block_size,
            antiquant_mode=0,
            key_antiquant_mode=0,
            value_antiquant_mode=0,
            inner_precise=1,
            sparse_mode=0,
        )
        attn_output = attn_output.squeeze(2)
        output[:batch_size] = attn_output
        return output

    def _forward_c8_chunked_prefill(
        self,
        query: torch.Tensor,
        float_key: torch.Tensor | None,
        float_value: torch.Tensor | None,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        """C8 ChunkedPrefill: decode via FIA V1 BNSD paged INT8 (zero gather),
        prefill via FIA V1 TND with float KV (new) or gather+dequant (continuing).
        """
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_decodes = attn_metadata.num_decodes
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])  # type: ignore[index]

        if num_decode_tokens > 0:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
            assert block_size % 32 == 0, (
                f"C8 INT8 KV cache requires block_size to be a multiple of 32, got {block_size}"
            )
            kv_k = self._nz_5d_view(self.key_cache, block_size)
            kv_v = self._nz_5d_view(self.value_cache, block_size)

            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query[:num_decode_tokens].unsqueeze(2),
                kv_k,
                kv_v,
                key_antiquant_scale=layer._c8_k_aq_scale_nz_bnsd,
                value_antiquant_scale=layer._c8_v_aq_scale_nz_bnsd,
                block_table=attn_metadata.block_tables[:num_decodes],
                actual_seq_lengths_kv=attn_metadata.seq_lens_list[:num_decodes],
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BNSD",
                scale=self.scale,
                block_size=block_size,
                antiquant_mode=0,
                key_antiquant_mode=0,
                value_antiquant_mode=0,
                inner_precise=1,
                sparse_mode=0,
            )
            output[:num_decode_tokens] = attn_out.squeeze(2)

        if attn_metadata.num_prefills > 0:
            prefill_q = query[num_decode_tokens:num_tokens]

            prefill_seq_qlen = [
                actual_seq_qlen[i] - num_decode_tokens for i in range(num_decodes, len(actual_seq_qlen))
            ]

            all_new_prefill = True
            for i in range(num_decodes, len(attn_metadata.seq_lens_list)):
                q_start = actual_seq_qlen[i - 1] if i > 0 else 0
                qlen_i = actual_seq_qlen[i] - q_start
                if attn_metadata.seq_lens_list[i] > qlen_i:
                    all_new_prefill = False
                    break

            if all_new_prefill and float_key is not None and float_value is not None:
                prefill_k = float_key[num_decode_tokens:num_tokens]
                prefill_v = float_value[num_decode_tokens:num_tokens]
                prefill_seq_kvlen = prefill_seq_qlen
            else:
                num_block, blk_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
                paged_k = self._nz_5d_view(self.key_cache, blk_size)
                paged_v = self._nz_5d_view(self.value_cache, blk_size)

                prefill_bt = attn_metadata.block_tables[num_decodes:]
                prefill_sl = attn_metadata.seq_lens_list[num_decodes:]
                prefill_k, prefill_v = self._dequant_paged_kv_to_dense(
                    paged_k, paged_v, prefill_bt, prefill_sl, query.dtype, layer
                )
                prefill_seq_kvlen = torch.tensor(prefill_sl, dtype=torch.int32).cumsum(dim=0)

            # block_table is None for prefill; FIA ignores block_size in this case.
            # Use cache block_size for consistency rather than a magic number.
            cache_block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query=prefill_q,
                key=prefill_k,
                value=prefill_v,
                atten_mask=attn_metadata.attn_mask,
                block_table=None,
                input_layout="TND",
                block_size=cache_block_size,
                actual_seq_lengths=prefill_seq_qlen,
                actual_seq_lengths_kv=prefill_seq_kvlen,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
            )
            n_prefill = num_tokens - num_decode_tokens
            attn_out = attn_out.view(n_prefill, self.num_heads, self.head_size)
            output[num_decode_tokens:num_tokens] = attn_out[:n_prefill]

        return output

    def _forward_c8_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ):
        """C8 FIA V1 TND for prefill states (PrefillNoCache uses float KV directly,
        PrefillCacheHit gathers + dequants paged INT8 KV).
        """
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)

        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])  # type: ignore[index]
        query = query[:num_tokens]

        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]

        if key.dtype == torch.int8:
            if block_table is not None:
                seq_lens = (
                    actual_seq_lengths_kv if isinstance(actual_seq_lengths_kv, list) else actual_seq_lengths_kv.tolist()
                )
                key, value = self._dequant_paged_kv_to_dense(key, value, block_table, seq_lens, query.dtype, layer)
                block_table = None
                # block_table is None after dequant; FIA ignores block_size.
                # Use cache block_size for consistency rather than a magic number.
                block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
                actual_seq_lengths_kv = torch.tensor(seq_lens, dtype=torch.int32).cumsum(dim=0)
            else:
                key = (key.to(query.dtype) - layer._c8_k_offset) * layer._c8_k_scale
                value = (value.to(query.dtype) - layer._c8_v_offset) * layer._c8_v_scale

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output
        return output

    def _reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        if len(kv_cache) > 1:
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            slots = attn_metadata.slot_mapping

            encoder_decoder = self.attn_type == AttentionType.ENCODER_DECODER

            # NZ write path: 5D view + npu_scatter_pa_kv_cache
            block_size = self.vllm_config.cache_config.block_size
            k_cache_layer = self._nz_5d_view(self.key_cache, block_size)
            v_cache_layer = self._nz_5d_view(self.value_cache, block_size)

            torch_npu.npu_scatter_pa_kv_cache(
                key=key[: attn_metadata.num_actual_tokens] if not encoder_decoder else key,
                value=value[: attn_metadata.num_actual_tokens] if not encoder_decoder else value,
                key_cache=k_cache_layer,
                value_cache=v_cache_layer,
                slot_mapping=slots[: attn_metadata.num_actual_tokens] if not encoder_decoder else slots,
            )

            notify_kv_cache_written()
        return query, key, value, output
