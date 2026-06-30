import torch

from vllm_ascend.spec_decode.tree_execution import build_tree_spec_decode_metadata
from vllm_ascend.spec_decode.tree_runtime import build_spec_tree_runtime
from vllm_ascend.spec_decode.tree_verifier import (
    pack_tree_output_tokens,
    select_tree_accept_paths,
)


def test_select_tree_accept_paths_prefers_deepest_matching_branch():
    runtime = build_spec_tree_runtime(((0,), (1,), (0, 0), (0, 1), (0, 1, 0), (1, 0)))
    metadata = build_tree_spec_decode_metadata(runtime, batch_size=1)
    draft_token_ids = torch.tensor([11, 12, 21, 22, 31, 41], dtype=torch.int32)
    target_token_ids = torch.tensor([11, 22, 41, 12, 31, 99, 77], dtype=torch.int32)

    accepted = select_tree_accept_paths(metadata, draft_token_ids, target_token_ids)

    assert accepted[0].node_indices == (0, 3, 4)
    assert accepted[0].token_ids == (11, 22, 31)
    assert accepted[0].bonus_token_id == 77
    assert accepted[0].terminal_row_index == 6


def test_pack_tree_output_tokens_keeps_existing_placeholder_shape():
    accepted = (
        type("Accepted", (), {"token_ids": (11, 21), "bonus_token_id": 101}),
        type("Accepted", (), {"token_ids": (12,), "bonus_token_id": 102}),
    )

    packed = pack_tree_output_tokens(accepted, max_path_len=2, placeholder_token_id=-1)

    assert packed.tolist() == [[11, 21, 101], [12, 102, -1]]


def test_select_tree_accept_paths_handles_non_bfs_runtime_order():
    runtime = build_spec_tree_runtime(((0,), (0, 0), (0, 1), (1,), (1, 0)))
    metadata = build_tree_spec_decode_metadata(runtime, batch_size=1)
    draft_token_ids = torch.tensor([11, 21, 22, 12, 31], dtype=torch.int32)
    target_token_ids = torch.tensor([11, 22, 41, 42, 99, 77], dtype=torch.int32)

    accepted = select_tree_accept_paths(metadata, draft_token_ids, target_token_ids)

    assert accepted[0].node_indices == (0, 2)
    assert accepted[0].token_ids == (11, 22)
    assert accepted[0].bonus_token_id == 99
    assert accepted[0].terminal_row_index == 4
