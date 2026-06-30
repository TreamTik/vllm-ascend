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
import json
import os

import torch
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID

from vllm_ascend.spec_decode.tree_execution import AscendTreeSpecDecodeMetadata


@dataclass(frozen=True)
class AcceptedTreePath:
    node_indices: tuple[int, ...]
    token_ids: tuple[int, ...]
    bonus_token_id: int | None
    terminal_row_index: int


_TREE_TRACE_COUNTER = 0


def _maybe_dump_tree_trace(
    metadata: AscendTreeSpecDecodeMetadata,
    draft_token_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    accepted_paths: tuple[AcceptedTreePath, ...],
) -> None:
    trace_path = os.getenv("VLLM_ASCEND_TREE_TRACE_PATH")
    if not trace_path:
        return

    global _TREE_TRACE_COUNTER
    max_steps = int(os.getenv("VLLM_ASCEND_TREE_TRACE_STEPS", "16"))
    if _TREE_TRACE_COUNTER >= max_steps:
        return

    payload = {
        "step": _TREE_TRACE_COUNTER,
        "pid": os.getpid(),
        "runtime_paths": [list(path) for path in metadata.runtime.paths],
        "runtime_parent_index": list(metadata.runtime.parent_index),
        "runtime_depth": list(metadata.runtime.depth),
        "request_row_offsets": list(metadata.request_row_offsets),
        "draft_token_ids": draft_token_ids.tolist(),
        "target_token_ids": target_token_ids.tolist(),
        "accepted_paths": [
            {
                "node_indices": list(path.node_indices),
                "token_ids": list(path.token_ids),
                "bonus_token_id": path.bonus_token_id,
                "terminal_row_index": path.terminal_row_index,
            }
            for path in accepted_paths
        ],
    }
    with open(trace_path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _TREE_TRACE_COUNTER += 1


def select_tree_accept_paths(
    metadata: AscendTreeSpecDecodeMetadata,
    draft_token_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    *,
    include_bonus_token: bool = True,
) -> tuple[AcceptedTreePath, ...]:
    runtime = metadata.runtime
    accepted_paths: list[AcceptedTreePath] = []
    num_requests = len(metadata.request_node_offsets) - 1

    for request_index in range(num_requests):
        request_node_offset = metadata.request_node_offsets[request_index]
        request_row_offset = metadata.request_row_offsets[request_index]
        valid_runtime_nodes: dict[int, bool] = {}
        best_runtime_node: int | None = None
        best_depth = 0

        for runtime_nodes in runtime.nodes_by_depth:
            for runtime_node_index in runtime_nodes:
                parent_runtime_node_index = runtime.parent_index[runtime_node_index]
                parent_valid = (
                    parent_runtime_node_index == -1 or valid_runtime_nodes.get(parent_runtime_node_index, False)
                )
                if not parent_valid:
                    valid_runtime_nodes[runtime_node_index] = False
                    continue

                token_index = request_node_offset + runtime_node_index
                if parent_runtime_node_index == -1:
                    target_row_index = request_row_offset
                else:
                    target_row_index = request_row_offset + metadata.runtime_node_to_execution_index[parent_runtime_node_index] + 1
                matched = int(draft_token_ids[token_index].item()) == int(target_token_ids[target_row_index].item())
                valid_runtime_nodes[runtime_node_index] = matched
                if matched and runtime.depth[runtime_node_index] > best_depth:
                    best_runtime_node = runtime_node_index
                    best_depth = runtime.depth[runtime_node_index]

        node_indices: list[int] = []
        token_ids: list[int] = []
        if best_runtime_node is not None:
            current_runtime_node = best_runtime_node
            while current_runtime_node != -1:
                node_indices.append(current_runtime_node)
                token_index = request_node_offset + current_runtime_node
                token_ids.append(int(draft_token_ids[token_index].item()))
                current_runtime_node = runtime.parent_index[current_runtime_node]
            node_indices.reverse()
            token_ids.reverse()

        terminal_row_index = request_row_offset
        if best_runtime_node is not None:
            terminal_row_index = request_row_offset + metadata.runtime_node_to_execution_index[best_runtime_node] + 1

        bonus_token_id: int | None = None
        if include_bonus_token:
            bonus_token_id = int(target_token_ids[terminal_row_index].item())

        accepted_paths.append(
            AcceptedTreePath(
                node_indices=tuple(node_indices),
                token_ids=tuple(token_ids),
                bonus_token_id=bonus_token_id,
                terminal_row_index=terminal_row_index,
            )
        )

    _maybe_dump_tree_trace(
        metadata=metadata,
        draft_token_ids=draft_token_ids,
        target_token_ids=target_token_ids,
        accepted_paths=tuple(accepted_paths),
    )
    return tuple(accepted_paths)


def pack_tree_output_tokens(
    accepted_paths: tuple[AcceptedTreePath, ...],
    max_path_len: int,
    placeholder_token_id: int = PLACEHOLDER_TOKEN_ID,
    device: torch.device | None = None,
) -> torch.Tensor:
    output = torch.full(
        (len(accepted_paths), max_path_len + 1),
        placeholder_token_id,
        dtype=torch.int32,
        device=device,
    )
    for row, accepted_path in enumerate(accepted_paths):
        for col, token_id in enumerate(accepted_path.token_ids):
            output[row, col] = token_id
        if accepted_path.bonus_token_id is not None:
            output[row, len(accepted_path.token_ids)] = accepted_path.bonus_token_id
    return output
