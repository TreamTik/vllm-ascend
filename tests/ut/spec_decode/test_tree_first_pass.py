from types import SimpleNamespace

import torch

from vllm_ascend.spec_decode.tree_first_pass import build_tree_first_pass_inputs


def test_build_tree_first_pass_inputs_keeps_only_terminal_rows():
    common_attn_metadata = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=6,
        num_input_tokens=6,
        max_query_len=6,
        decode_token_per_req=6,
        graph_pad_size=0,
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 6], dtype=torch.int32),
        positions=torch.tensor([60, 61, 61, 62, 62, 62], dtype=torch.int32),
        positions_cpu=torch.tensor([60, 61, 61, 62, 62, 62], dtype=torch.int32),
        slot_mapping=torch.tensor([100, 101, 102, 103, 104, 105], dtype=torch.int32),
        slot_mapping_cpu=torch.tensor([100, 101, 102, 103, 104, 105], dtype=torch.int32),
        seq_lens=torch.tensor([63], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([63], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([63], dtype=torch.int32),
        num_computed_tokens_cpu=torch.tensor([62], dtype=torch.int32),
        _num_computed_tokens_cpu=torch.tensor([62], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([63], dtype=torch.int32),
    )
    target_positions = torch.tensor([60, 61, 61, 62, 62, 62], dtype=torch.int32)
    target_hidden_states = torch.arange(24, dtype=torch.float32).view(6, 4)

    tree_inputs = build_tree_first_pass_inputs(
        target_token_ids=torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32),
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=torch.tensor([777], dtype=torch.int32),
        terminal_row_indices=torch.tensor([3], dtype=torch.int32),
        common_attn_metadata=common_attn_metadata,
    )

    assert tree_inputs.num_tokens == 1
    assert tree_inputs.token_ids.tolist() == [777]
    assert tree_inputs.positions.tolist() == [62]
    assert tree_inputs.hidden_states.tolist() == target_hidden_states[3:4].tolist()
    assert tree_inputs.token_indices_to_sample.tolist() == [0]
    assert tree_inputs.common_attn_metadata.query_start_loc.tolist() == [0, 1]
    assert tree_inputs.common_attn_metadata.slot_mapping.tolist() == [103]
    assert tree_inputs.common_attn_metadata.seq_lens.tolist() == [63]
    assert tree_inputs.common_attn_metadata.num_computed_tokens_cpu.tolist() == [62]
