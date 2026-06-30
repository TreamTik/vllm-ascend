#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_ascend.spec_decode.tree_runtime import SpecTreeRuntime


@dataclass(frozen=True)
class TreeBatchItem:
    request_index: int
    runtime_node_index: int
    node_index: int


@dataclass(frozen=True)
class TreeStepLayout:
    depth: int
    items: tuple[TreeBatchItem, ...]


@dataclass(frozen=True)
class AscendTreeSpecDecodeMetadata:
    batch_size: int
    runtime: SpecTreeRuntime
    request_node_offsets: tuple[int, ...]
    request_row_offsets: tuple[int, ...]
    request_frontier_offsets: tuple[tuple[int, ...], ...]
    execution_runtime_node_indices: tuple[int, ...]
    runtime_node_to_execution_index: tuple[int, ...]
    node_parent_indices: torch.Tensor
    node_depths: torch.Tensor
    draft_token_ids: torch.Tensor | None = None
    num_draft_tokens: list[int] | None = None
    cu_num_draft_tokens: torch.Tensor | None = None
    target_logits_indices: torch.Tensor | None = None
    bonus_logits_indices: torch.Tensor | None = None
    logits_indices: torch.Tensor | None = None

    @property
    def max_path_len(self) -> int:
        return max(self.runtime.depth, default=0)

    @property
    def max_spec_len(self) -> int:
        if self.num_draft_tokens:
            return max(self.num_draft_tokens)
        return self.runtime.node_count

    @property
    def rows_per_request(self) -> int:
        return self.runtime.node_count + 1


def build_tree_query_position_offsets(runtime: SpecTreeRuntime) -> tuple[int, ...]:
    execution_runtime_node_indices = build_tree_execution_runtime_node_indices(runtime)
    return tuple(runtime.depth[runtime_node_index] for runtime_node_index in execution_runtime_node_indices)


def build_tree_query_slot_offsets(runtime: SpecTreeRuntime) -> tuple[int, ...]:
    slot_offsets = [0] * runtime.node_count
    next_offset = 1
    for runtime_nodes in runtime.nodes_by_depth:
        for runtime_node_index in runtime_nodes:
            slot_offsets[runtime_node_index] = next_offset
            next_offset += 1
    return tuple(slot_offsets)


def build_tree_query_depth_offsets(runtime: SpecTreeRuntime) -> tuple[int, ...]:
    return build_tree_query_position_offsets(runtime)


def build_tree_execution_runtime_node_indices(runtime: SpecTreeRuntime) -> tuple[int, ...]:
    return tuple(
        runtime_node_index
        for runtime_nodes in runtime.nodes_by_depth
        for runtime_node_index in runtime_nodes
    )


def build_tree_attn_mask(runtime: SpecTreeRuntime, depth: int) -> torch.Tensor:
    if depth < 1 or depth > len(runtime.nodes_by_depth):
        raise ValueError(f"depth must be in [1, {len(runtime.nodes_by_depth)}], got {depth}.")

    execution_nodes = [
        runtime_node_index
        for runtime_nodes in runtime.nodes_by_depth[:depth]
        for runtime_node_index in runtime_nodes
    ]
    node_to_col = {runtime_node_index: col for col, runtime_node_index in enumerate(execution_nodes)}
    size = len(execution_nodes)
    attn_mask = torch.ones((size, size), dtype=torch.int8)

    for row, runtime_node_index in enumerate(execution_nodes):
        attn_mask[row, row] = 0
        parent_runtime_node_index = runtime.parent_index[runtime_node_index]
        while parent_runtime_node_index != -1:
            attn_mask[row, node_to_col[parent_runtime_node_index]] = 0
            parent_runtime_node_index = runtime.parent_index[parent_runtime_node_index]

    return attn_mask


def build_tree_target_attn_mask(runtime: SpecTreeRuntime) -> torch.Tensor:
    size = runtime.node_count + 1
    attn_mask = torch.ones((size, size), dtype=torch.int8)
    execution_runtime_node_indices = build_tree_execution_runtime_node_indices(runtime)
    runtime_node_to_execution_index = {
        runtime_node_index: execution_index
        for execution_index, runtime_node_index in enumerate(execution_runtime_node_indices)
    }

    for row in range(size):
        attn_mask[row, row] = 0

    for execution_index, runtime_node_index in enumerate(execution_runtime_node_indices):
        row = execution_index + 1
        attn_mask[row, 0] = 0
        parent_runtime_node_index = runtime.parent_index[runtime_node_index]
        while parent_runtime_node_index != -1:
            attn_mask[row, runtime_node_to_execution_index[parent_runtime_node_index] + 1] = 0
            parent_runtime_node_index = runtime.parent_index[parent_runtime_node_index]

    return attn_mask


def _build_request_node_offsets(runtime: SpecTreeRuntime, batch_size: int) -> tuple[int, ...]:
    return tuple(request_idx * runtime.node_count for request_idx in range(batch_size + 1))


def _build_request_row_offsets(runtime: SpecTreeRuntime, batch_size: int) -> tuple[int, ...]:
    rows_per_request = runtime.node_count + 1
    return tuple(request_idx * rows_per_request for request_idx in range(batch_size + 1))


def build_tree_step_layout(
    runtime: SpecTreeRuntime,
    batch_size: int,
) -> tuple[TreeStepLayout, ...]:
    request_node_offsets = _build_request_node_offsets(runtime, batch_size)
    layouts: list[TreeStepLayout] = []

    for depth, runtime_nodes in enumerate(runtime.nodes_by_depth, start=1):
        items: list[TreeBatchItem] = []
        for request_index in range(batch_size):
            request_offset = request_node_offsets[request_index]
            for runtime_node_index in runtime_nodes:
                items.append(
                    TreeBatchItem(
                        request_index=request_index,
                        runtime_node_index=runtime_node_index,
                        node_index=request_offset + runtime_node_index,
                    )
                )
        layouts.append(TreeStepLayout(depth=depth, items=tuple(items)))

    return tuple(layouts)


def build_tree_spec_decode_metadata(
    runtime: SpecTreeRuntime,
    batch_size: int,
) -> AscendTreeSpecDecodeMetadata:
    request_node_offsets = _build_request_node_offsets(runtime, batch_size)
    request_row_offsets = _build_request_row_offsets(runtime, batch_size)
    execution_runtime_node_indices = build_tree_execution_runtime_node_indices(runtime)
    runtime_node_to_execution_index_list = [0] * runtime.node_count
    for execution_index, runtime_node_index in enumerate(execution_runtime_node_indices):
        runtime_node_to_execution_index_list[runtime_node_index] = execution_index
    request_frontier_offsets: list[tuple[int, ...]] = []
    request_offset = 0
    depth_counts = [len(nodes) for nodes in runtime.nodes_by_depth]

    for _ in range(batch_size):
        frontiers = [request_offset]
        for depth_count in depth_counts:
            frontiers.append(frontiers[-1] + depth_count)
        request_frontier_offsets.append(tuple(frontiers))
        request_offset += runtime.node_count

    node_parent_indices = torch.tensor(runtime.parent_index * batch_size, dtype=torch.int32)
    node_depths = torch.tensor(runtime.depth * batch_size, dtype=torch.int32)

    return AscendTreeSpecDecodeMetadata(
        batch_size=batch_size,
        runtime=runtime,
        request_node_offsets=request_node_offsets,
        request_row_offsets=request_row_offsets,
        request_frontier_offsets=tuple(request_frontier_offsets),
        execution_runtime_node_indices=execution_runtime_node_indices,
        runtime_node_to_execution_index=tuple(runtime_node_to_execution_index_list),
        node_parent_indices=node_parent_indices,
        node_depths=node_depths,
    )
