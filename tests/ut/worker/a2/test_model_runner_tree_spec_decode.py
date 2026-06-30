from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend.spec_decode.tree_execution import build_tree_spec_decode_metadata
from vllm_ascend.spec_decode.tree_runtime import build_spec_tree_runtime
from vllm_ascend.spec_decode.tree_verifier import AcceptedTreePath
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.model_runner_v1 import TreeAcceptedPathSlotCopy


def test_calc_tree_spec_decode_metadata_uses_runtime_node_count():
    runner = object.__new__(NPUModelRunner)
    runner.device = torch.device("cpu")
    runner.input_ids = MagicMock()
    runner.input_ids.gpu = torch.arange(64, dtype=torch.int32)
    runner.drafter = MagicMock()
    runner.drafter._tree_runtime.node_count = 5
    runner.drafter._tree_runtime.paths = ((0,), (1,), (0, 0), (0, 1), (1, 0))
    runner.drafter._tree_runtime.parent_index = (-1, -1, 0, 0, 1)
    runner.drafter._tree_runtime.depth = (1, 1, 2, 2, 2)
    runner.drafter._tree_runtime.nodes_by_depth = ((0, 1), (2, 3, 4))
    runner.drafter._tree_runtime.children = ((2, 3), (4,), (), (), ())
    runner.drafter._tree_runtime.leaf_indices = (2, 3, 4)

    metadata = runner._calc_tree_spec_decode_metadata(batch_size=2)

    assert metadata.request_node_offsets == (0, 5, 10)
    assert metadata.runtime.paths == ((0,), (1,), (0, 0), (0, 1), (1, 0))
    assert metadata.node_parent_indices.tolist() == [-1, -1, 0, 0, 1, -1, -1, 0, 0, 1]
    assert metadata.num_draft_tokens == [5, 5]
    assert metadata.cu_num_draft_tokens.tolist() == [5, 10]
    assert metadata.target_logits_indices.tolist() == list(range(12))
    assert metadata.bonus_logits_indices.tolist() == [0, 6]


def test_calc_tree_spec_decode_metadata_reorders_bfs_inputs_back_to_runtime_order():
    runner = object.__new__(NPUModelRunner)
    runner.device = torch.device("cpu")
    runner.input_ids = MagicMock()
    runner.input_ids.gpu = torch.tensor([100, 11, 12, 21, 22, 31], dtype=torch.int32)
    runner.drafter = MagicMock()
    runner.drafter._tree_runtime = build_spec_tree_runtime(((0,), (0, 0), (0, 1), (1,), (1, 0)))

    metadata = runner._calc_tree_spec_decode_metadata(batch_size=1)

    assert metadata.execution_runtime_node_indices == (0, 3, 1, 2, 4)
    assert metadata.runtime_node_to_execution_index == (0, 2, 3, 1, 4)
    assert metadata.draft_token_ids.tolist() == [11, 21, 22, 12, 31]


def test_prepare_tree_spec_decode_bootstrap_state_exposes_runner_fields():
    runner = object.__new__(NPUModelRunner)
    runner.device = torch.device("cpu")
    runner.pin_memory = False
    runner.max_num_reqs = 3
    runner.max_num_tokens = 9
    runner.compilation_config = SimpleNamespace(mode=None, cudagraph_mode=None)
    runner.model_config = SimpleNamespace(enforce_eager=True)
    runner.speculative_config = SimpleNamespace(
        use_eagle=lambda: True,
    )

    with patch("vllm_ascend.worker.model_runner_v1.get_c_env", return_value="true"):
        runner._prepare_tree_spec_decode_bootstrap_state(
            SimpleNamespace(
                speculative_config=runner.speculative_config,
                scheduler_config=SimpleNamespace(max_num_seqs=runner.max_num_reqs),
            ),
            runner.device,
        )

    assert runner.pcp_size == 1
    assert runner.dcp_size == 1
    assert runner.pcp_rank == 0
    assert runner.dcp_rank == 0
    assert runner.use_eagle is True
    assert runner.enable_enpu is True
    assert runner.query_start_loc.cpu.shape[0] == runner.max_num_reqs + 2
    assert runner.query_start_loc.gpu.shape[0] == runner.max_num_reqs + 2


def test_prepare_tree_spec_decode_bootstrap_state_preserves_existing_runner_fields():
    runner = object.__new__(NPUModelRunner)
    runner.device = torch.device("cpu")
    runner.pin_memory = False
    runner.max_num_reqs = 2
    runner.max_num_tokens = 8
    runner.compilation_config = SimpleNamespace(mode=None, cudagraph_mode=None)
    runner.model_config = SimpleNamespace(enforce_eager=True)
    runner.speculative_config = SimpleNamespace(
        use_eagle=lambda: False,
    )
    runner.pcp_size = 4
    runner.dcp_size = 2
    runner.pcp_rank = 3
    runner.dcp_rank = 1
    runner.use_eagle = "preset"
    runner.enable_enpu = "preset"
    existing_query_start_loc = SimpleNamespace(cpu=torch.tensor([1]), gpu=torch.tensor([2]))
    runner.query_start_loc = existing_query_start_loc

    with patch("vllm_ascend.worker.model_runner_v1.get_c_env", return_value=None):
        runner._prepare_tree_spec_decode_bootstrap_state(
            SimpleNamespace(
                speculative_config=runner.speculative_config,
                scheduler_config=SimpleNamespace(max_num_seqs=runner.max_num_reqs),
            ),
            runner.device,
        )

    assert runner.pcp_size == 4
    assert runner.dcp_size == 2
    assert runner.pcp_rank == 3
    assert runner.dcp_rank == 1
    assert runner.use_eagle == "preset"
    assert runner.enable_enpu == "preset"
    assert runner.query_start_loc is existing_query_start_loc


def test_patch_tree_spec_decode_attn_metadata_overlays_tree_mask():
    runner = object.__new__(NPUModelRunner)
    runner.device = torch.device("cpu")

    runtime = build_spec_tree_runtime(((0,), (1,), (0, 0), (0, 1), (1, 0)))
    metadata = build_tree_spec_decode_metadata(runtime, batch_size=1)
    base_mask = torch.triu(torch.ones((8, 8), dtype=torch.int8), diagonal=1)
    attn_metadata = {"layer": SimpleNamespace(attn_mask=base_mask.clone())}

    runner._patch_tree_spec_decode_attn_metadata(attn_metadata, metadata)

    expected = base_mask.clone()
    expected[: runtime.node_count + 1, : runtime.node_count + 1] = (
        torch.tensor(
            [
                [0, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1],
                [0, 1, 0, 1, 1, 1],
                [0, 0, 1, 0, 1, 1],
                [0, 0, 1, 1, 0, 1],
                [0, 1, 0, 1, 1, 0],
            ],
            dtype=torch.int8,
        )
    )
    assert attn_metadata["layer"].attn_mask.tolist() == expected.tolist()

def test_prepare_inputs_padded_tree_uses_terminal_row_indices_for_hidden_state_selection():
    proposer = object.__new__(AscendSpecDecodeBaseProposer)
    proposer.device = torch.device("cpu")
    proposer.runner = SimpleNamespace(
        tree_terminal_row_indices=torch.tensor([5], dtype=torch.int32),
        attn_state=None,
        actual_seq_lengths_q=None,
        decode_token_per_req=1,
    )
    proposer._tree_runtime = object()
    proposer.pcp_size = 1
    proposer.arange = torch.arange(16, dtype=torch.int32)
    runtime = build_spec_tree_runtime(((0,), (1,), (0, 0), (0, 1), (0, 1, 0), (1, 0)))
    metadata = build_tree_spec_decode_metadata(runtime, batch_size=1)
    metadata = type(metadata)(
        **{
            **metadata.__dict__,
            "cu_num_draft_tokens": torch.tensor([6], dtype=torch.int32),
        }
    )
    common_attn_metadata = SimpleNamespace(
        num_reqs=1,
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 6], dtype=torch.int32),
        seq_lens_cpu=None,
        _seq_lens_cpu=None,
        seq_lens_cpu_upper_bound=None,
        num_actual_tokens=6,
        num_input_tokens=6,
        actual_seq_lengths_q=None,
        block_table_tensor=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(6, dtype=torch.int32),
        slot_mapping_cpu=None,
        positions=None,
        positions_cpu=None,
        num_computed_tokens_cpu=None,
        _num_computed_tokens_cpu=None,
        seq_lens=torch.tensor([6], dtype=torch.int32),
        is_prefilling=torch.tensor([False]),
    )

    with patch("vllm_ascend.spec_decode.llm_base_proposer.HAS_TRITON", False):
        _, token_indices, token_indices_to_sample, num_rejected_tokens = (
            AscendSpecDecodeBaseProposer.prepare_inputs_padded(
                proposer,
                common_attn_metadata,
                metadata,
                torch.tensor([4], dtype=torch.int32),
            )
        )

    assert token_indices.tolist() == [0, 1, 2, 3, 4, 5]
    assert token_indices_to_sample.tolist() == [5]
    assert num_rejected_tokens.tolist() == [3]


def test_plan_tree_accepted_path_slot_copies_rewrites_branch_slot_to_canonical_prefix():
    runner = object.__new__(NPUModelRunner)
    runner.input_batch = SimpleNamespace(
        block_table=[SimpleNamespace(block_size=128)],
    )
    runner.tree_accepted_paths = (
        AcceptedTreePath(
            node_indices=(1,),
            token_ids=(20412,),
            bonus_token_id=99489,
            terminal_row_index=2,
        ),
    )

    runtime = build_spec_tree_runtime(((0,), (1,), (0, 0), (0, 1), (1, 0)))
    metadata = build_tree_spec_decode_metadata(runtime, batch_size=1)
    common_attn_metadata = SimpleNamespace(
        slot_mapping=torch.tensor([205, 206, 207, 208, 209, 210], dtype=torch.int32),
        positions=torch.tensor([77, 78, 78, 79, 79, 79], dtype=torch.int32),
        block_table_tensor=torch.tensor([[1, 2]], dtype=torch.int32),
    )

    copies = runner._plan_tree_accepted_path_slot_copies(
        metadata,
        common_attn_metadata,
    )

    assert copies == [
        TreeAcceptedPathSlotCopy(
            request_index=0,
            runtime_node_index=1,
            token_id=20412,
            position=78,
            source_row_index=2,
            source_slot=207,
            destination_slot=206,
        )
    ]


def test_copy_tree_accepted_path_kv_slots_copies_tensor_pages_once():
    runner = object.__new__(NPUModelRunner)
    runner.device = torch.device("cpu")

    key_cache = torch.arange(16, dtype=torch.float32).view(4, 2, 2)
    value_cache = (torch.arange(16, dtype=torch.float32) + 100).view(4, 2, 2)
    extra_cache = (torch.arange(16, dtype=torch.float32) + 200).view(4, 2, 2)
    shared_cache = (key_cache, value_cache, extra_cache)
    runner.kv_caches = [shared_cache, shared_cache]

    slot_copy = TreeAcceptedPathSlotCopy(
        request_index=0,
        runtime_node_index=1,
        token_id=20412,
        position=78,
        source_row_index=2,
        source_slot=3,
        destination_slot=1,
    )
    expected_key = key_cache[3].clone()
    expected_value = value_cache[3].clone()
    expected_extra = extra_cache[3].clone()

    runner._copy_tree_accepted_path_kv_slots([slot_copy])

    assert torch.equal(key_cache[1], expected_key)
    assert torch.equal(value_cache[1], expected_value)
    assert torch.equal(extra_cache[1], expected_extra)
