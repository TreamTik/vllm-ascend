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

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time

import torch

_NON_ROOT_VISIBLE_ROW_CACHE: dict[tuple[int, int, str], tuple[tuple[torch.Tensor, ...], tuple[int, ...], int]] = {}
_ALL_ROW_VISIBLE_ROW_CACHE: dict[tuple[int, int, str], tuple[tuple[torch.Tensor, ...], tuple[int, ...], int]] = {}


def resolve_tree_fast_impl(tree_impl: str | None) -> str:
    raw_impl = (tree_impl or "batched").strip().lower()
    if raw_impl in ("batched", "batched_all_rows"):
        return "batched"
    if raw_impl in ("batched_legacy", "legacy"):
        return "batched_legacy"
    if raw_impl in ("batched_split", "split"):
        return "batched_split"
    if raw_impl == "reference":
        return "reference"
    raise ValueError(
        "Unsupported VLLM_ASCEND_TREE_FAST_IMPL "
        f"'{tree_impl}'. Expected one of: batched, batched_legacy, "
        "batched_split, reference."
    )


def _get_cached_non_root_visible_rows(
    tree_mask: torch.Tensor,
    q_len: int,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...], int]:
    cache_key = (int(tree_mask.data_ptr()), q_len, str(device))
    cached = _NON_ROOT_VISIBLE_ROW_CACHE.get(cache_key)
    if cached is not None:
        return cached

    req_tree_mask = tree_mask[1:q_len, :q_len].to(device=device)
    visible_rows: list[torch.Tensor] = []
    visible_counts: list[int] = []
    current_copy_tokens = 0
    for local_row in range(max(0, q_len - 1)):
        visible_current = torch.nonzero(req_tree_mask[local_row].eq(0), as_tuple=False).flatten().to(torch.long)
        visible_rows.append(visible_current)
        visible_count = int(visible_current.numel())
        visible_counts.append(visible_count)
        current_copy_tokens += visible_count

    result = (tuple(visible_rows), tuple(visible_counts), current_copy_tokens)
    _NON_ROOT_VISIBLE_ROW_CACHE[cache_key] = result
    return result


def _get_cached_all_row_visible_rows(
    tree_mask: torch.Tensor,
    q_len: int,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...], int]:
    cache_key = (int(tree_mask.data_ptr()), q_len, str(device))
    cached = _ALL_ROW_VISIBLE_ROW_CACHE.get(cache_key)
    if cached is not None:
        return cached

    req_tree_mask = tree_mask[:q_len, :q_len].to(device=device)
    visible_rows: list[torch.Tensor] = []
    visible_counts: list[int] = []
    current_copy_tokens = 0
    for local_row in range(q_len):
        visible_current = torch.nonzero(req_tree_mask[local_row].eq(0), as_tuple=False).flatten().to(torch.long)
        visible_rows.append(visible_current)
        visible_count = int(visible_current.numel())
        visible_counts.append(visible_count)
        current_copy_tokens += visible_count

    result = (tuple(visible_rows), tuple(visible_counts), current_copy_tokens)
    _ALL_ROW_VISIBLE_ROW_CACHE[cache_key] = result
    return result


def _build_request_slot_ids(
    block_table_row: torch.Tensor,
    seq_len: int,
    block_size: int,
) -> torch.Tensor:
    if seq_len <= 0:
        return torch.empty((0,), dtype=torch.long, device=block_table_row.device)

    num_blocks = (seq_len + block_size - 1) // block_size
    used_blocks = block_table_row[:num_blocks].to(dtype=torch.long)
    block_offsets = torch.arange(block_size, dtype=torch.long, device=block_table_row.device)
    return (used_blocks.unsqueeze(1) * block_size + block_offsets.unsqueeze(0)).reshape(-1)[:seq_len]


@dataclass
class TreeSplitPlan:
    output_indices: torch.Tensor
    prefix_block_tables: torch.Tensor
    actual_seq_lengths_q: list[int]
    prefix_actual_seq_lengths_kv: list[int]
    current_actual_seq_lengths_kv: list[int]
    current_token_indices: torch.Tensor
    num_rows: int
    total_visible_tokens: int
    max_visible_tokens: int
    prefix_copy_tokens: int
    current_copy_tokens: int


@dataclass
class TreePagedPlan:
    output_indices: torch.Tensor
    packed_slots: torch.Tensor
    actual_seq_lengths_q: list[int]
    actual_seq_lengths_kv: list[int]
    num_rows: int
    total_visible_tokens: int
    max_visible_tokens: int
    prefix_copy_tokens: int
    current_copy_tokens: int


def build_tree_split_plan(
    actual_seq_lengths_q: list[int],
    seq_lens: list[int],
    tree_mask: torch.Tensor,
    block_tables: torch.Tensor,
    device: torch.device,
) -> tuple[TreeSplitPlan, dict[str, float | int]]:
    total_start = time.perf_counter()
    output_indices: list[int] = []
    prefix_req_indices: list[int] = []
    prefix_lens: list[int] = []
    current_visible_lens: list[int] = []
    current_token_parts: list[torch.Tensor] = []
    prefix_copy_tokens = 0
    current_copy_tokens = 0
    max_visible_tokens = 0

    plan_start = time.perf_counter()
    q_start = 0
    for req_idx, q_end in enumerate(actual_seq_lengths_q):
        q_len = q_end - q_start
        if q_len <= 1:
            q_start = q_end
            continue

        seq_len = seq_lens[req_idx]
        if seq_len < q_len:
            raise ValueError(f"tree split attention requires seq_len >= q_len, got {seq_len} < {q_len}")

        prefix_len = seq_len - q_len
        if prefix_len <= 0:
            raise ValueError(f"tree split attention requires prefix_len > 0, got {prefix_len}")

        visible_rows, visible_counts, cached_current_copy_tokens = _get_cached_non_root_visible_rows(
            tree_mask,
            q_len,
            device,
        )
        for local_row, query_index in enumerate(range(q_start + 1, q_end)):
            visible_count = visible_counts[local_row]
            if visible_count <= 0:
                raise ValueError("tree split attention requires at least one visible current token per row")
            output_indices.append(query_index)
            prefix_req_indices.append(req_idx)
            prefix_lens.append(prefix_len)
            current_visible_lens.append(visible_count)
            current_token_parts.append(visible_rows[local_row].to(dtype=torch.long) + q_start)
            prefix_copy_tokens += prefix_len
            max_visible_tokens = max(max_visible_tokens, prefix_len + visible_count)
        current_copy_tokens += cached_current_copy_tokens
        q_start = q_end
    plan_ms = (time.perf_counter() - plan_start) * 1000.0

    if not output_indices:
        empty_long = torch.empty((0,), dtype=torch.long, device=device)
        empty_block_tables = block_tables[:0]
        return TreeSplitPlan(
            output_indices=empty_long,
            prefix_block_tables=empty_block_tables,
            actual_seq_lengths_q=[],
            prefix_actual_seq_lengths_kv=[],
            current_actual_seq_lengths_kv=[],
            current_token_indices=empty_long,
            num_rows=0,
            total_visible_tokens=0,
            max_visible_tokens=0,
            prefix_copy_tokens=0,
            current_copy_tokens=0,
        ), {
            "num_rows": 0,
            "total_visible_tokens": 0,
            "max_visible_tokens": 0,
            "prefix_copy_tokens": 0,
            "current_copy_tokens": 0,
            "plan_ms": plan_ms,
            "pack_ms": 0.0,
            "fia_ms": 0.0,
            "total_ms": (time.perf_counter() - total_start) * 1000.0,
        }

    output_indices_t = torch.tensor(output_indices, dtype=torch.long, device=device)
    prefix_req_indices_t = torch.tensor(prefix_req_indices, dtype=torch.long, device=block_tables.device)
    current_token_indices = torch.cat(current_token_parts, dim=0)

    running_current_kv_len = 0
    current_actual_seq_lengths_kv: list[int] = []
    for visible_len in current_visible_lens:
        running_current_kv_len += visible_len
        current_actual_seq_lengths_kv.append(running_current_kv_len)

    plan = TreeSplitPlan(
        output_indices=output_indices_t,
        prefix_block_tables=block_tables.index_select(0, prefix_req_indices_t),
        actual_seq_lengths_q=list(range(1, len(output_indices) + 1)),
        prefix_actual_seq_lengths_kv=prefix_lens,
        current_actual_seq_lengths_kv=current_actual_seq_lengths_kv,
        current_token_indices=current_token_indices,
        num_rows=len(output_indices),
        total_visible_tokens=prefix_copy_tokens + current_copy_tokens,
        max_visible_tokens=max_visible_tokens,
        prefix_copy_tokens=prefix_copy_tokens,
        current_copy_tokens=current_copy_tokens,
    )
    return plan, {
        "num_rows": plan.num_rows,
        "total_visible_tokens": plan.total_visible_tokens,
        "max_visible_tokens": plan.max_visible_tokens,
        "prefix_copy_tokens": plan.prefix_copy_tokens,
        "current_copy_tokens": plan.current_copy_tokens,
        "plan_ms": plan_ms,
        "pack_ms": 0.0,
        "fia_ms": 0.0,
        "total_ms": (time.perf_counter() - total_start) * 1000.0,
    }


def build_tree_paged_plan(
    block_tables: torch.Tensor,
    seq_lens: list[int],
    actual_seq_lengths_q: list[int],
    tree_mask: torch.Tensor,
    device: torch.device,
    block_size: int,
    include_row0: bool = False,
) -> TreePagedPlan:
    output_indices: list[int] = []
    visible_lens: list[int] = []
    packed_slot_parts: list[torch.Tensor] = []
    q_start = 0
    prefix_copy_tokens = 0
    current_copy_tokens = 0

    for req_idx, q_end in enumerate(actual_seq_lengths_q):
        q_len = q_end - q_start
        seq_len = seq_lens[req_idx]
        if q_len <= 0:
            q_start = q_end
            continue
        if not include_row0 and q_len == 1:
            q_start = q_end
            continue
        if seq_len < q_len:
            raise ValueError(f"tree exact attention requires seq_len >= q_len, got {seq_len} < {q_len}")

        prefix_len = seq_len - q_len
        req_slots = _build_request_slot_ids(block_tables[req_idx], seq_len, block_size)
        prefix_slots = req_slots[:prefix_len]
        current_slots = req_slots[prefix_len:seq_len]
        if include_row0:
            visible_rows, visible_counts, cached_current_copy_tokens = _get_cached_all_row_visible_rows(
                tree_mask,
                q_len,
                device,
            )
            query_indices = range(q_start, q_end)
        else:
            visible_rows, visible_counts, cached_current_copy_tokens = _get_cached_non_root_visible_rows(
                tree_mask,
                q_len,
                device,
            )
            query_indices = range(q_start + 1, q_end)

        for local_row, query_index in enumerate(query_indices):
            visible_current = visible_rows[local_row]
            visible_count = visible_counts[local_row]
            output_indices.append(query_index)
            visible_lens.append(prefix_len + visible_count)
            if prefix_slots.numel() > 0:
                packed_slot_parts.append(prefix_slots)
            if visible_current.numel() > 0:
                packed_slot_parts.append(current_slots.index_select(0, visible_current.to(dtype=torch.long)))
            prefix_copy_tokens += prefix_len
        current_copy_tokens += cached_current_copy_tokens
        q_start = q_end

    if not output_indices:
        empty_indices = torch.empty((0,), dtype=torch.long, device=device)
        empty_slots = torch.empty((0,), dtype=torch.long, device=block_tables.device)
        return TreePagedPlan(
            output_indices=empty_indices,
            packed_slots=empty_slots,
            actual_seq_lengths_q=[],
            actual_seq_lengths_kv=[],
            num_rows=0,
            total_visible_tokens=0,
            max_visible_tokens=0,
            prefix_copy_tokens=0,
            current_copy_tokens=0,
        )

    actual_seq_lengths_kv: list[int] = []
    running_kv_len = 0
    for visible_len in visible_lens:
        running_kv_len += visible_len
        actual_seq_lengths_kv.append(running_kv_len)

    return TreePagedPlan(
        output_indices=torch.tensor(output_indices, dtype=torch.long, device=device),
        packed_slots=torch.cat(packed_slot_parts, dim=0),
        actual_seq_lengths_q=list(range(1, len(output_indices) + 1)),
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        num_rows=len(output_indices),
        total_visible_tokens=sum(visible_lens),
        max_visible_tokens=max(visible_lens),
        prefix_copy_tokens=prefix_copy_tokens,
        current_copy_tokens=current_copy_tokens,
    )


def _run_python_tree_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    keep_mask: torch.Tensor,
    scale: float,
    num_queries_per_kv: int,
) -> torch.Tensor:
    """Dense reference implementation used as a fallback/debug path."""
    finfo_min = torch.finfo(torch.float32).min
    if num_queries_per_kv > 1:
        key = key.repeat_interleave(num_queries_per_kv, dim=1)
        value = value.repeat_interleave(num_queries_per_kv, dim=1)

    q_heads = query.permute(1, 0, 2).to(torch.float32)
    k_heads = key.permute(1, 2, 0).to(torch.float32)
    v_heads = value.permute(1, 0, 2).to(torch.float32)

    scores = torch.matmul(q_heads, k_heads) * scale
    scores = scores.masked_fill(~keep_mask.unsqueeze(0), finfo_min)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v_heads).permute(1, 0, 2).to(query.dtype)


def _run_vendor_tree_attention_rowwise(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    prefix_len: int,
    tree_mask: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Use Ascend fused attention on each speculative row's compact visible KV."""
    import torch_npu

    q_len = query.shape[0]
    current_key = key[prefix_len:]
    current_value = value[prefix_len:]
    outputs = []

    for row in range(q_len):
        visible_current = torch.nonzero(tree_mask[row, :q_len].eq(0), as_tuple=False).flatten()
        visible_key = torch.cat((key[:prefix_len], current_key[visible_current]), dim=0).contiguous()
        visible_value = torch.cat((value[:prefix_len], current_value[visible_current]), dim=0).contiguous()
        visible_len = visible_key.shape[0]
        row_query = query[row : row + 1].contiguous()
        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=row_query,
            key=visible_key,
            value=visible_value,
            input_layout="TND",
            actual_seq_lengths=[1],
            actual_seq_lengths_kv=[visible_len],
            num_key_value_heads=num_kv_heads,
            num_heads=num_heads,
            scale=scale,
            sparse_mode=0,
        )
        outputs.append(attn_output)

    return torch.cat(outputs, dim=0)


def _run_vendor_tree_attention_batched(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens: list[int],
    actual_seq_lengths_q: list[int],
    tree_mask: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    """Pack all speculative rows into a single FIA launch.

    Each speculative row is rewritten as an independent single-query request
    whose KV sequence is exactly the row-visible prefix + ancestor/current
    tokens. This preserves the exact-tree semantics while collapsing the
    expensive per-row fused-attention launch into one batched launch.
    """
    import torch_npu

    total_start = time.perf_counter()
    row_plans: list[tuple[int, int, int, torch.Tensor]] = []
    output_indices: list[int] = []
    visible_lens: list[int] = []
    q_start = 0
    kv_start = 0
    prefix_copy_tokens = 0
    current_copy_tokens = 0

    plan_start = time.perf_counter()
    for req_idx, q_end in enumerate(actual_seq_lengths_q):
        q_len = q_end - q_start
        seq_len = seq_lens[req_idx]
        if q_len <= 1:
            q_start = q_end
            kv_start += seq_len
            continue
        if seq_len < q_len:
            raise ValueError(f"tree exact attention requires seq_len >= q_len, got {seq_len} < {q_len}")

        prefix_len = seq_len - q_len
        visible_rows, visible_counts, cached_current_copy_tokens = _get_cached_non_root_visible_rows(
            tree_mask,
            q_len,
            query.device,
        )
        for local_row, query_index in enumerate(range(q_start + 1, q_end)):
            visible_current = visible_rows[local_row]
            row_plans.append((query_index, kv_start, prefix_len, visible_current))
            output_indices.append(query_index)
            visible_lens.append(prefix_len + visible_counts[local_row])
            prefix_copy_tokens += prefix_len
        current_copy_tokens += cached_current_copy_tokens

        q_start = q_end
        kv_start += seq_len
    plan_ms = (time.perf_counter() - plan_start) * 1000.0

    if not row_plans:
        empty_indices = torch.empty((0,), dtype=torch.long, device=query.device)
        empty_output = torch.empty((0, num_heads, query.shape[-1]), dtype=query.dtype, device=query.device)
        return empty_indices, empty_output, {
            "num_rows": 0,
            "total_visible_tokens": 0,
            "max_visible_tokens": 0,
            "prefix_copy_tokens": 0,
            "current_copy_tokens": 0,
            "plan_ms": plan_ms,
            "pack_ms": 0.0,
            "fia_ms": 0.0,
            "total_ms": (time.perf_counter() - total_start) * 1000.0,
        }

    output_indices_t = torch.tensor(output_indices, dtype=torch.long, device=query.device)
    packed_query = query.index_select(0, output_indices_t).contiguous()
    total_visible_tokens = sum(visible_lens)
    max_visible_tokens = max(visible_lens)

    pack_start = time.perf_counter()
    packed_index_parts: list[torch.Tensor] = []
    prefix_index_cache: dict[tuple[int, int], torch.Tensor] = {}
    for _, req_kv_start, prefix_len, visible_current in row_plans:
        if prefix_len > 0:
            cache_key = (req_kv_start, prefix_len)
            prefix_indices = prefix_index_cache.get(cache_key)
            if prefix_indices is None:
                prefix_indices = torch.arange(
                    req_kv_start,
                    req_kv_start + prefix_len,
                    dtype=torch.long,
                    device=key.device,
                )
                prefix_index_cache[cache_key] = prefix_indices
            packed_index_parts.append(prefix_indices)

        if visible_current.numel() > 0:
            packed_index_parts.append(visible_current.to(dtype=torch.long) + req_kv_start + prefix_len)
    packed_indices = torch.cat(packed_index_parts, dim=0)
    packed_key = key.index_select(0, packed_indices).contiguous()
    packed_value = value.index_select(0, packed_indices).contiguous()
    pack_ms = (time.perf_counter() - pack_start) * 1000.0

    actual_seq_lengths_q = list(range(1, packed_query.shape[0] + 1))
    actual_seq_lengths_kv: list[int] = []
    running_kv_len = 0
    for visible_len in visible_lens:
        running_kv_len += visible_len
        actual_seq_lengths_kv.append(running_kv_len)

    fia_start = time.perf_counter()
    attn_output, _ = torch_npu.npu_fused_infer_attention_score(
        query=packed_query,
        key=packed_key,
        value=packed_value,
        input_layout="TND",
        actual_seq_lengths=actual_seq_lengths_q,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        num_key_value_heads=num_kv_heads,
        num_heads=num_heads,
        scale=scale,
        sparse_mode=0,
    )
    fia_ms = (time.perf_counter() - fia_start) * 1000.0
    return output_indices_t, attn_output, {
        "num_rows": len(row_plans),
        "total_visible_tokens": total_visible_tokens,
        "max_visible_tokens": max_visible_tokens,
        "prefix_copy_tokens": prefix_copy_tokens,
        "current_copy_tokens": current_copy_tokens,
        "plan_ms": plan_ms,
        "pack_ms": pack_ms,
        "fia_ms": fia_ms,
        "total_ms": (time.perf_counter() - total_start) * 1000.0,
    }


def run_fast_tree_attention_from_paged_cache(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    plan: TreePagedPlan,
    scale: float,
    num_queries_per_kv: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    """Pack visible KV for non-root rows directly from paged cache.

    This is the exact/correct batched path. It removes the intermediate
    request-major ragged KV materialization and gathers cache tokens only in
    the final visible-token order required by FIA.
    """
    del num_queries_per_kv
    import torch_npu

    total_start = time.perf_counter()
    num_heads = query.shape[1]
    num_kv_heads = key_cache.shape[2]
    head_size = query.shape[-1]
    flat_key = key_cache.view(-1, num_kv_heads, head_size)
    flat_value = value_cache.view(-1, num_kv_heads, head_size)
    if plan.num_rows == 0:
        empty_output = torch.empty((0, num_heads, head_size), dtype=query.dtype, device=query.device)
        return plan.output_indices, empty_output, {
            "num_rows": 0,
            "total_visible_tokens": 0,
            "max_visible_tokens": 0,
            "prefix_copy_tokens": 0,
            "current_copy_tokens": 0,
            "pack_ms": 0.0,
            "fia_ms": 0.0,
            "total_ms": (time.perf_counter() - total_start) * 1000.0,
        }

    packed_query = query.index_select(0, plan.output_indices).contiguous()

    pack_start = time.perf_counter()
    packed_key = flat_key.index_select(0, plan.packed_slots).contiguous()
    packed_value = flat_value.index_select(0, plan.packed_slots).contiguous()
    pack_ms = (time.perf_counter() - pack_start) * 1000.0

    fia_start = time.perf_counter()
    attn_output, _ = torch_npu.npu_fused_infer_attention_score(
        query=packed_query,
        key=packed_key,
        value=packed_value,
        input_layout="TND",
        actual_seq_lengths=plan.actual_seq_lengths_q,
        actual_seq_lengths_kv=plan.actual_seq_lengths_kv,
        num_key_value_heads=num_kv_heads,
        num_heads=num_heads,
        scale=scale,
        sparse_mode=0,
    )
    fia_ms = (time.perf_counter() - fia_start) * 1000.0
    return plan.output_indices, attn_output, {
        "num_rows": plan.num_rows,
        "total_visible_tokens": plan.total_visible_tokens,
        "max_visible_tokens": plan.max_visible_tokens,
        "prefix_copy_tokens": plan.prefix_copy_tokens,
        "current_copy_tokens": plan.current_copy_tokens,
        "pack_ms": pack_ms,
        "fia_ms": fia_ms,
        "total_ms": (time.perf_counter() - total_start) * 1000.0,
    }


def run_split_current_tree_attention(
    packed_query: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    plan: TreeSplitPlan,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    import torch_npu

    if plan.num_rows == 0:
        empty_output = torch.empty_like(packed_query)
        empty_lse = torch.empty((0, num_heads, 1), dtype=torch.float32, device=packed_query.device)
        return empty_output[:0], empty_lse, {
            "pack_ms": 0.0,
            "fia_ms": 0.0,
            "total_ms": 0.0,
        }

    total_start = time.perf_counter()
    pack_start = time.perf_counter()
    packed_key = current_key.index_select(0, plan.current_token_indices).contiguous()
    packed_value = current_value.index_select(0, plan.current_token_indices).contiguous()
    pack_ms = (time.perf_counter() - pack_start) * 1000.0

    fia_start = time.perf_counter()
    attn_output, attn_lse = torch_npu.npu_fused_infer_attention_score(
        query=packed_query,
        key=packed_key,
        value=packed_value,
        input_layout="TND",
        actual_seq_lengths=plan.actual_seq_lengths_q,
        actual_seq_lengths_kv=plan.current_actual_seq_lengths_kv,
        num_key_value_heads=num_kv_heads,
        num_heads=num_heads,
        scale=scale,
        sparse_mode=0,
        softmax_lse_flag=True,
    )
    fia_ms = (time.perf_counter() - fia_start) * 1000.0
    attn_output = attn_output.view(plan.num_rows, num_heads, packed_query.shape[-1])
    attn_lse = attn_lse.view(plan.num_rows, num_heads, 1)
    return attn_output, attn_lse, {
        "pack_ms": pack_ms,
        "fia_ms": fia_ms,
        "total_ms": (time.perf_counter() - total_start) * 1000.0,
    }


def merge_split_tree_attention_outputs(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    current_output: torch.Tensor,
    current_lse: torch.Tensor,
) -> torch.Tensor:
    import torch_npu

    num_rows, num_heads, head_size = prefix_output.shape
    prefix_flat = prefix_output.contiguous().view(num_rows * num_heads, head_size).to(torch.float32)
    current_flat = current_output.contiguous().view(num_rows * num_heads, head_size).to(torch.float32)
    prefix_lse_flat = prefix_lse.contiguous().view(num_rows * num_heads).to(torch.float32)
    current_lse_flat = current_lse.contiguous().view(num_rows * num_heads).to(torch.float32)

    merged_flat, _ = torch_npu.npu_attention_update(
        [prefix_lse_flat, current_lse_flat],
        [prefix_flat, current_flat],
        0,
    )
    return merged_flat.view(num_rows, num_heads, head_size).to(prefix_output.dtype)


def gather_paged_kv_to_ragged(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: list[int],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    """Gather paged KV cache into a ragged packed tensor ordered by request."""
    batch_size = len(seq_lens)
    block_tables = block_tables[:batch_size]
    max_blocks_per_seq = block_tables.shape[1]
    block_size = key_cache.shape[1]
    max_tokens_padded = max_blocks_per_seq * block_size

    flat_ids = block_tables.reshape(-1).to(torch.long)
    gathered_k = key_cache[flat_ids].reshape(batch_size, max_tokens_padded, *key_cache.shape[2:])
    gathered_v = value_cache[flat_ids].reshape(batch_size, max_tokens_padded, *value_cache.shape[2:])

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=key_cache.device)
    positions = torch.arange(max_tokens_padded, dtype=torch.long, device=key_cache.device)
    valid_mask = (positions.unsqueeze(0) < seq_lens_t.unsqueeze(1)).reshape(-1)

    dense_k = gathered_k.reshape(-1, *key_cache.shape[2:])[valid_mask]
    dense_v = gathered_v.reshape(-1, *value_cache.shape[2:])[valid_mask]
    return dense_k, dense_v


def run_exact_tree_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens: list[int],
    actual_seq_lengths_q: list[int],
    tree_mask: torch.Tensor,
    scale: float,
    num_queries_per_kv: int,
) -> torch.Tensor:
    """Compute exact tree attention on top of ragged dense KV.

    The KV tensor layout is request-major packed ragged storage. For each request,
    all prefix tokens remain visible while the last ``q_len`` speculative rows use
    the provided tree mask to express branching visibility.
    """
    num_tokens, num_heads, head_size = query.shape
    output = torch.empty((num_tokens, num_heads, head_size), dtype=query.dtype, device=query.device)

    q_start = 0
    kv_start = 0
    exact_impl = os.getenv("VLLM_ASCEND_TREE_EXACT_IMPL", "vendor_rowwise")
    use_vendor_rowwise = exact_impl != "python_dense"

    for req_idx, q_end in enumerate(actual_seq_lengths_q):
        q_len = q_end - q_start
        seq_len = seq_lens[req_idx]
        if q_len <= 0:
            q_start = q_end
            kv_start += seq_len
            continue
        if seq_len < q_len:
            raise ValueError(f"tree exact attention requires seq_len >= q_len, got {seq_len} < {q_len}")

        prefix_len = seq_len - q_len
        q_req = query[q_start:q_end]
        k_req = key[kv_start : kv_start + seq_len]
        v_req = value[kv_start : kv_start + seq_len]

        keep_mask = torch.ones((q_len, seq_len), dtype=torch.bool, device=query.device)
        req_tree_mask = tree_mask[:q_len, :q_len].to(device=query.device)
        keep_mask[:, prefix_len:seq_len] = req_tree_mask.eq(0)

        if use_vendor_rowwise:
            try:
                req_output = _run_vendor_tree_attention_rowwise(
                    query=q_req,
                    key=k_req,
                    value=v_req,
                    prefix_len=prefix_len,
                    tree_mask=req_tree_mask,
                    scale=scale,
                    num_heads=num_heads,
                    num_kv_heads=k_req.shape[1],
                )
            except Exception:
                req_output = _run_python_tree_attention(
                    query=q_req,
                    key=k_req,
                    value=v_req,
                    keep_mask=keep_mask,
                    scale=scale,
                    num_queries_per_kv=num_queries_per_kv,
                )
        else:
            req_output = _run_python_tree_attention(
                query=q_req,
                key=k_req,
                value=v_req,
                keep_mask=keep_mask,
                scale=scale,
                num_queries_per_kv=num_queries_per_kv,
            )
        output[q_start:q_end] = req_output

        q_start = q_end
        kv_start += seq_len

    return output


def run_fast_tree_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens: list[int],
    actual_seq_lengths_q: list[int],
    tree_mask: torch.Tensor,
    scale: float,
    num_queries_per_kv: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute speculative rows with a single batched FIA launch.

    Row0/root decode is intentionally excluded and should be computed by the
    caller using the baseline decode path to preserve the validated root-row
    semantics.
    """
    del num_queries_per_kv
    if query.numel() == 0:
        empty_indices = torch.empty((0,), dtype=torch.long, device=query.device)
        empty_output = torch.empty_like(query)
        return empty_indices, empty_output[:0], {
            "num_rows": 0,
            "total_visible_tokens": 0,
            "max_visible_tokens": 0,
            "prefix_copy_tokens": 0,
            "current_copy_tokens": 0,
            "plan_ms": 0.0,
            "pack_ms": 0.0,
            "fia_ms": 0.0,
            "total_ms": 0.0,
        }

    return _run_vendor_tree_attention_batched(
        query=query,
        key=key,
        value=value,
        seq_lens=seq_lens,
        actual_seq_lengths_q=actual_seq_lengths_q,
        tree_mask=tree_mask,
        scale=scale,
        num_heads=query.shape[1],
        num_kv_heads=key.shape[1],
    )


_TREE_EXACT_DEBUG_COUNT = 0


def _matches_debug_filters(layer_name: str | None, filter_spec: str | None) -> bool:
    if not filter_spec:
        return True
    if layer_name is None:
        return False
    filters = [item.strip() for item in filter_spec.split(",") if item.strip()]
    if not filters:
        return True
    return any(layer_name == item or layer_name.startswith(f"{item}.") for item in filters)


def maybe_dump_tree_exact_debug(payload: dict) -> None:
    trace_path = os.getenv("VLLM_ASCEND_TREE_EXACT_DEBUG_PATH")
    if not trace_path:
        return

    tp_rank = payload.get("tp_rank")
    if os.getenv("VLLM_ASCEND_TREE_EXACT_DEBUG_ALL_TP", "0") != "1" and tp_rank not in (None, 0):
        return

    if not _matches_debug_filters(
        payload.get("layer_name"),
        os.getenv("VLLM_ASCEND_TREE_EXACT_DEBUG_LAYERS"),
    ):
        return

    global _TREE_EXACT_DEBUG_COUNT
    max_steps = int(os.getenv("VLLM_ASCEND_TREE_EXACT_DEBUG_STEPS", "16"))
    if _TREE_EXACT_DEBUG_COUNT >= max_steps:
        return

    payload = dict(payload)
    payload["index"] = _TREE_EXACT_DEBUG_COUNT
    with open(trace_path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _TREE_EXACT_DEBUG_COUNT += 1
