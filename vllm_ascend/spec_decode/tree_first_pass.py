# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import os

import torch

from vllm.v1.attention.backends.utils import CommonAttentionMetadata


@dataclass(frozen=True)
class TreeFirstPassInputs:
    token_ids: torch.Tensor
    positions: torch.Tensor
    hidden_states: torch.Tensor
    token_indices_to_sample: torch.Tensor
    num_tokens: int
    common_attn_metadata: CommonAttentionMetadata


_TREE_FIRST_PASS_TRACE_COUNTER = 0


def _serialize(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _maybe_dump_tree_first_pass_trace(
    *,
    target_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    next_token_ids: torch.Tensor,
    terminal_row_indices: torch.Tensor,
    tree_inputs: TreeFirstPassInputs,
) -> None:
    trace_path = os.getenv("VLLM_ASCEND_TREE_FIRST_PASS_TRACE_PATH")
    if not trace_path:
        return

    global _TREE_FIRST_PASS_TRACE_COUNTER
    max_steps = int(os.getenv("VLLM_ASCEND_TREE_FIRST_PASS_TRACE_STEPS", "24"))
    if _TREE_FIRST_PASS_TRACE_COUNTER >= max_steps:
        return

    row_indices = terminal_row_indices.to(device=target_token_ids.device, dtype=torch.long)
    payload = {
        "step": _TREE_FIRST_PASS_TRACE_COUNTER,
        "pid": os.getpid(),
        "terminal_row_indices": _serialize(terminal_row_indices),
        "selected_target_token_ids": _serialize(target_token_ids.index_select(0, row_indices)),
        "selected_target_positions": _serialize(_select_positions(target_positions, row_indices)),
        "next_token_ids": _serialize(next_token_ids),
        "rebuilt_token_ids": _serialize(tree_inputs.token_ids),
        "rebuilt_positions": _serialize(tree_inputs.positions),
        "rebuilt_query_start_loc": _serialize(getattr(tree_inputs.common_attn_metadata, "query_start_loc", None)),
        "rebuilt_seq_lens": _serialize(getattr(tree_inputs.common_attn_metadata, "seq_lens", None)),
        "rebuilt_num_computed_tokens_cpu": _serialize(
            getattr(tree_inputs.common_attn_metadata, "num_computed_tokens_cpu", None)
        ),
        "rebuilt_slot_mapping": _serialize(getattr(tree_inputs.common_attn_metadata, "slot_mapping", None)),
    }
    with open(trace_path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _TREE_FIRST_PASS_TRACE_COUNTER += 1


def _select_positions(positions: torch.Tensor, row_indices: torch.Tensor) -> torch.Tensor:
    if positions.ndim == 1:
        return positions.index_select(0, row_indices)
    return positions.index_select(-1, row_indices)


def _seq_lens_from_positions(positions: torch.Tensor) -> torch.Tensor:
    base_positions = positions[0] if positions.ndim > 1 else positions
    return base_positions.to(dtype=torch.int32) + 1


def build_tree_first_pass_inputs(
    *,
    target_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    target_hidden_states: torch.Tensor,
    next_token_ids: torch.Tensor,
    terminal_row_indices: torch.Tensor,
    common_attn_metadata: CommonAttentionMetadata,
) -> TreeFirstPassInputs:
    """Build the next tree-draft root batch from verifier terminal rows.

    Tree speculative decoding should advance from the accepted frontier only.
    The next root proposal therefore needs exactly one row per request:
    - token id: the verifier bonus token for the request
    - position / hidden state / slot mapping: the terminal target row chosen by
      the verifier

    Re-feeding the whole previous target tree batch keeps stale sibling rows
    alive and gradually drifts the draft context. Compressing to terminal rows
    avoids that linear-chain leakage.
    """
    device = target_token_ids.device
    row_indices = terminal_row_indices.to(device=device, dtype=torch.long)
    batch_size = row_indices.shape[0]

    rebuilt_positions = _select_positions(target_positions, row_indices)
    rebuilt_hidden_states = target_hidden_states.index_select(0, row_indices)
    rebuilt_token_ids = next_token_ids.to(device=device, dtype=target_token_ids.dtype)
    rebuilt_token_indices = torch.arange(batch_size, dtype=torch.int32, device=device)

    rebuilt_metadata = copy.copy(common_attn_metadata)
    query_start_loc = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    seq_lens = _seq_lens_from_positions(rebuilt_positions)
    num_computed_tokens = seq_lens - 1

    rebuilt_metadata.num_reqs = batch_size
    rebuilt_metadata.num_actual_tokens = batch_size
    if hasattr(rebuilt_metadata, "num_input_tokens"):
        rebuilt_metadata.num_input_tokens = batch_size
    rebuilt_metadata.max_query_len = 1
    if hasattr(rebuilt_metadata, "decode_token_per_req"):
        rebuilt_metadata.decode_token_per_req = 1
    if hasattr(rebuilt_metadata, "graph_pad_size"):
        rebuilt_metadata.graph_pad_size = -1
    rebuilt_metadata.query_start_loc = query_start_loc
    rebuilt_metadata.query_start_loc_cpu = query_start_loc.cpu()
    rebuilt_metadata.positions = rebuilt_positions
    if getattr(rebuilt_metadata, "positions_cpu", None) is not None:
        rebuilt_metadata.positions_cpu = rebuilt_positions.cpu()

    if getattr(rebuilt_metadata, "slot_mapping", None) is not None:
        rebuilt_metadata.slot_mapping = rebuilt_metadata.slot_mapping.index_select(0, row_indices).to(dtype=torch.int32)
    if getattr(rebuilt_metadata, "slot_mapping_cpu", None) is not None:
        rebuilt_metadata.slot_mapping_cpu = rebuilt_metadata.slot_mapping.cpu()

    rebuilt_metadata.seq_lens = seq_lens
    if getattr(rebuilt_metadata, "seq_lens_cpu", None) is not None:
        rebuilt_metadata.seq_lens_cpu = seq_lens.cpu()
    if getattr(rebuilt_metadata, "_seq_lens_cpu", None) is not None:
        rebuilt_metadata._seq_lens_cpu = seq_lens.cpu().clone()
    if getattr(rebuilt_metadata, "seq_lens_cpu_upper_bound", None) is not None:
        rebuilt_metadata.seq_lens_cpu_upper_bound = seq_lens.cpu().clone()

    if getattr(rebuilt_metadata, "num_computed_tokens_cpu", None) is not None:
        rebuilt_metadata.num_computed_tokens_cpu = num_computed_tokens.cpu()
    if getattr(rebuilt_metadata, "_num_computed_tokens_cpu", None) is not None:
        rebuilt_metadata._num_computed_tokens_cpu = num_computed_tokens.cpu().clone()
    if hasattr(rebuilt_metadata, "max_seq_len"):
        rebuilt_metadata.max_seq_len = int(seq_lens.max().item())

    tree_inputs = TreeFirstPassInputs(
        token_ids=rebuilt_token_ids,
        positions=rebuilt_positions,
        hidden_states=rebuilt_hidden_states,
        token_indices_to_sample=rebuilt_token_indices,
        num_tokens=batch_size,
        common_attn_metadata=rebuilt_metadata,
    )
    _maybe_dump_tree_first_pass_trace(
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        next_token_ids=next_token_ids,
        terminal_row_indices=terminal_row_indices,
        tree_inputs=tree_inputs,
    )
    return tree_inputs
