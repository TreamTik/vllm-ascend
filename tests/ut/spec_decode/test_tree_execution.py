import torch

from vllm_ascend.spec_decode.tree_execution import (
    build_tree_attn_mask,
    build_tree_spec_decode_metadata,
    build_tree_query_position_offsets,
    build_tree_query_slot_offsets,
    build_tree_step_layout,
)
from vllm_ascend.spec_decode.tree_runtime import build_spec_tree_runtime


def test_build_tree_step_layout_groups_nodes_by_depth():
    runtime = build_spec_tree_runtime(((0,), (1,), (0, 0), (0, 1), (1, 0)))
    layouts = build_tree_step_layout(runtime, batch_size=2)

    assert len(layouts) == 2
    assert [item.node_index for item in layouts[0].items] == [0, 1, 5, 6]
    assert [item.runtime_node_index for item in layouts[0].items] == [0, 1, 0, 1]
    assert [item.runtime_node_index for item in layouts[1].items] == [2, 3, 4, 2, 3, 4]


def test_build_tree_spec_decode_metadata_offsets_are_contiguous():
    runtime = build_spec_tree_runtime(((0,), (1,), (0, 0), (0, 1), (1, 0)))
    metadata = build_tree_spec_decode_metadata(runtime, batch_size=2)

    assert metadata.request_node_offsets == (0, 5, 10)
    assert metadata.request_frontier_offsets == ((0, 2, 5), (5, 7, 10))
    assert metadata.node_parent_indices.tolist() == [-1, -1, 0, 0, 1, -1, -1, 0, 0, 1]
    assert metadata.node_depths.tolist() == [1, 1, 2, 2, 2, 1, 1, 2, 2, 2]


def test_build_tree_spec_decode_metadata_separates_runtime_and_execution_order():
    runtime = build_spec_tree_runtime(((0,), (0, 0), (0, 1), (1,), (1, 0)))

    metadata = build_tree_spec_decode_metadata(runtime, batch_size=1)

    assert metadata.execution_runtime_node_indices == (0, 3, 1, 2, 4)
    assert metadata.runtime_node_to_execution_index == (0, 2, 3, 1, 4)


def test_build_tree_query_position_offsets_follow_tree_depth():
    runtime = build_spec_tree_runtime(((0,), (0, 0), (0, 1), (1,), (1, 0)))

    offsets = build_tree_query_position_offsets(runtime)

    assert offsets == (1, 1, 2, 2, 2)


def test_build_tree_query_slot_offsets_assign_unique_slots_to_siblings():
    runtime = build_spec_tree_runtime(((0,), (1,), (0, 0), (0, 1), (1, 0)))

    offsets = build_tree_query_slot_offsets(runtime)

    assert offsets == (1, 2, 3, 4, 5)


def test_build_tree_attn_mask_masks_non_ancestor_branches():
    runtime = build_spec_tree_runtime(((0,), (0, 0), (0, 1), (1,), (1, 0)))

    attn_mask = build_tree_attn_mask(runtime, depth=2)

    assert attn_mask.tolist() == [
        [0, 1, 1, 1, 1],
        [1, 0, 1, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [1, 0, 1, 1, 0],
    ]


def test_tree_attn_mask_differs_from_plain_causal_mask_for_branching_tree():
    runtime = build_spec_tree_runtime(((0,), (1,), (0, 0), (0, 1), (1, 0)))

    tree_mask = build_tree_attn_mask(runtime, depth=2)
    causal_mask = torch.triu(torch.ones((runtime.node_count, runtime.node_count), dtype=torch.int8), diagonal=1)

    assert tree_mask.tolist() != causal_mask.tolist()
    assert tree_mask[2, 0].item() == 0
    assert tree_mask[2, 1].item() == 1
