from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend.spec_decode.tree_runtime import build_spec_tree_runtime


class TreeDraftModel:
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append({key: value.clone() if torch.is_tensor(value) else value for key, value in kwargs.items()})
        input_ids = kwargs["input_ids"].to(torch.float32)
        last_hidden_states = torch.zeros(input_ids.shape[0], 4, dtype=torch.float32)
        last_hidden_states[:, 0] = input_ids
        hidden_states = torch.zeros_like(last_hidden_states)
        hidden_states[:, 0] = input_ids + 100
        return last_hidden_states, hidden_states

    def compute_logits(self, hidden_states):
        token_ids = hidden_states[:, 0].to(torch.long) + 10
        logits = torch.full((hidden_states.shape[0], self.vocab_size), -1000.0)
        for row, token_id in enumerate(token_ids.tolist()):
            logits[row, token_id] = 1000.0
            logits[row, token_id + 1] = 999.0
            logits[row, token_id + 2] = 998.0
        return logits


class TreeAwareSpecConfig:
    def __init__(self, parsed_tree):
        self._parsed_tree = parsed_tree

    @property
    def parsed_speculative_token_tree(self):
        return self._parsed_tree

    def use_speculative_token_tree(self):
        return self._parsed_tree is not None


def make_proposer(speculative_config, method="eagle", use_cuda_graph=False):
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.speculative_config = speculative_config
    proposer.method = method
    proposer.use_cuda_graph = use_cuda_graph
    return proposer


def test_init_speculative_token_tree_runtime_ignores_legacy_v023_config():
    proposer = make_proposer(SimpleNamespace(speculative_token_tree="[(0,), (0, 0)]"))

    AscendSpecDecodeBaseProposer._init_speculative_token_tree_runtime(proposer)

    assert proposer.parsed_speculative_token_tree is None
    assert proposer._tree_runtime is None


def test_init_speculative_token_tree_runtime_builds_runtime_for_eager_eagle():
    proposer = make_proposer(
        TreeAwareSpecConfig(((0,), (1,), (0, 0), (0, 1), (1, 0))),
        method="eagle3",
        use_cuda_graph=False,
    )

    AscendSpecDecodeBaseProposer._init_speculative_token_tree_runtime(proposer)

    assert proposer.parsed_speculative_token_tree == (
        (0,),
        (1,),
        (0, 0),
        (0, 1),
        (1, 0),
    )
    assert proposer._tree_runtime is not None
    assert proposer._tree_runtime.parent_index == (-1, -1, 0, 0, 1)


def test_init_speculative_token_tree_runtime_rejects_non_eager_mode():
    proposer = make_proposer(
        TreeAwareSpecConfig(((0,), (0, 0))),
        method="eagle",
        use_cuda_graph=True,
    )

    with pytest.raises(NotImplementedError, match="eager"):
        AscendSpecDecodeBaseProposer._init_speculative_token_tree_runtime(proposer)


def test_init_speculative_token_tree_runtime_rejects_unsupported_method():
    proposer = make_proposer(
        TreeAwareSpecConfig(((0,), (0, 0))),
        method="mtp",
        use_cuda_graph=False,
    )

    with pytest.raises(NotImplementedError, match="eagle"):
        AscendSpecDecodeBaseProposer._init_speculative_token_tree_runtime(proposer)


def test_run_tree_draft_returns_execution_order_tokens():
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.method = "eagle"
    proposer.num_speculative_tokens = 5
    proposer.device = torch.device("cpu")
    proposer.use_cuda_graph = False
    proposer.parallel_drafting = False
    proposer.pass_hidden_states_to_model = True
    proposer.supports_mm_inputs = False
    proposer.uses_mrope = False
    proposer.max_model_len = 1024
    proposer.kernel_block_size = 16
    proposer.attn_layer_names = ["layer.0"]
    proposer.arange = torch.arange(64, dtype=torch.int32)
    proposer.token_arange_np = torch.arange(64, dtype=torch.int32).numpy()
    proposer.hidden_states = torch.zeros((64, 4), dtype=torch.float32)
    proposer.input_ids = torch.zeros(64, dtype=torch.int32)
    proposer.positions = torch.zeros(64, dtype=torch.int32)
    proposer.slot_mapping_group = [torch.zeros(64, dtype=torch.int32) for _ in range(5)]
    proposer.seq_lens_group = [torch.zeros(64, dtype=torch.int32) for _ in range(5)]
    proposer.query_start_loc_group = [torch.zeros(64, dtype=torch.int32) for _ in range(5)]
    proposer.model = TreeDraftModel()
    proposer.model_returns_tuple = lambda: True
    proposer.maybe_pad_and_reduce = lambda hidden_states, positions: (hidden_states, positions)
    proposer.maybe_all_gather_and_unpad = lambda last_hidden_states, positions, hidden_states: (
        last_hidden_states,
        positions,
        hidden_states,
    )
    proposer.shallow_copy_metadata = lambda metadata: metadata
    proposer._tree_runtime = build_spec_tree_runtime(((0,), (0, 0), (0, 1), (1,), (1, 0)))
    proposer.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1024),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
    )
    proposer.runner = SimpleNamespace(uniform_decode_query_len=1)
    proposer._get_positions = lambda num_tokens: proposer.positions[:num_tokens]
    proposer._set_positions = lambda num_tokens, positions: proposer.positions[:num_tokens].copy_(positions)

    builder = MagicMock()
    builder.build_for_drafting.side_effect = lambda common_attn_metadata, draft_index: SimpleNamespace(
        attn_mask=torch.triu(torch.ones((32, 32), dtype=torch.int8), diagonal=1),
        block_tables=common_attn_metadata.block_table_tensor[: common_attn_metadata.num_reqs].clone(),
        slot_mapping=common_attn_metadata.slot_mapping[: common_attn_metadata.num_actual_tokens].clone(),
        seq_lens=common_attn_metadata.seq_lens[: common_attn_metadata.num_reqs].clone(),
        seq_lens_list=common_attn_metadata.seq_lens[: common_attn_metadata.num_reqs].tolist(),
        actual_seq_lengths_q=common_attn_metadata.query_start_loc_cpu[1 : common_attn_metadata.num_reqs + 1].tolist(),
        causal=True,
        num_prefills=0,
    )
    proposer.draft_attn_groups = [SimpleNamespace(get_metadata_builder=lambda: builder)]

    proposer.input_ids[:4] = torch.tensor([2, 3, 6, 7], dtype=torch.int32)
    proposer.positions[:4] = torch.tensor([5, 6, 8, 9], dtype=torch.int32)
    proposer.hidden_states[:4] = torch.arange(16, dtype=torch.float32).view(4, 4)

    common_attn_metadata = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=4,
        num_input_tokens=4,
        max_query_len=1,
        decode_token_per_req=1,
        attn_state=None,
        graph_pad_size=-1,
        block_table_tensor=torch.arange(0, 128, dtype=torch.int32).view(2, 64),
        seq_lens=torch.tensor([7, 10], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([7, 10], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([7, 10], dtype=torch.int32),
        num_computed_tokens_cpu=None,
        positions=proposer.positions[:4].clone(),
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 4], dtype=torch.int32),
        slot_mapping=torch.arange(4, dtype=torch.int32),
        causal=True,
    )
    forward_context = SimpleNamespace(attn_metadata=None, moe_layer_index=0)
    multi_steps_attn_metadata = [{"layer.0": SimpleNamespace(num_prefills=0)}]

    with (
        patch("vllm_ascend.spec_decode.llm_base_proposer.get_forward_context", return_value=forward_context),
        patch("vllm_ascend.spec_decode.llm_base_proposer.lmhead_tp_enable", return_value=False),
    ):
        draft_token_ids = AscendSpecDecodeBaseProposer._run_tree_draft(
            proposer,
            num_input_tokens=4,
            batch_size=2,
            token_indices_to_sample=torch.tensor([1, 3], dtype=torch.int64),
            target_positions=proposer.positions[:4],
            inputs_embeds=None,
            multi_steps_attn_metadata=multi_steps_attn_metadata,
            num_tokens=4,
            common_attn_metadata=common_attn_metadata,
        )

    assert draft_token_ids.tolist() == [[13, 14, 23, 24, 24], [17, 18, 27, 28, 28]]
    assert len(proposer.model.calls) == 2
    assert torch.equal(
        proposer.model.calls[1]["input_ids"],
        torch.tensor([13, 14, 17, 18], dtype=torch.int32),
    )
    draft_metadata = builder.build_for_drafting.call_args.args[0]
    assert draft_metadata.slot_mapping.tolist()[:4] == [7, 8, 1034, 1035]


def test_propose_tree_path_patches_first_draft_step_with_target_tree_mask():
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.method = "eagle"
    proposer.num_speculative_tokens = 2
    proposer.device = torch.device("cpu")
    proposer.use_cuda_graph = False
    proposer.parallel_drafting = False
    proposer.supports_mm_inputs = False
    proposer.uses_mrope = False
    proposer.use_compress = False
    proposer.enable_enpu = False
    proposer.pcp_size = 1
    proposer.dcp_size = 1
    proposer._tree_runtime = build_spec_tree_runtime(((0,), (1,), (0, 0), (0, 1), (1, 0)))
    proposer.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(use_mla=False),
        speculative_config=SimpleNamespace(),
    )
    proposer.runner = SimpleNamespace(
        _sync_metadata_across_dp=lambda num_input_tokens, is_draft_model: (num_input_tokens, None, None),
        get_model=lambda: None,
        input_batch=SimpleNamespace(lora_id_to_lora_request={}),
    )
    proposer.slot_mapping_group = [torch.zeros(32, dtype=torch.int32) for _ in range(2)]
    proposer.seq_lens_group = [torch.zeros(32, dtype=torch.int32) for _ in range(2)]
    proposer.query_start_loc_group = [torch.zeros(32, dtype=torch.int32) for _ in range(2)]
    proposer.token_indices_to_sample = torch.zeros(32, dtype=torch.int32)
    proposer.positions = torch.zeros(32, dtype=torch.int32)
    proposer.set_inputs_first_pass = MagicMock()
    proposer._runnable = MagicMock()
    proposer._run_tree_draft = MagicMock(return_value=torch.tensor([[11, 12], [21, 22]], dtype=torch.int64))

    built_attn_metadata = SimpleNamespace(
        causal=True,
        num_prefills=0,
        attn_mask=torch.triu(torch.ones((16, 16), dtype=torch.int8), diagonal=1),
    )
    builder = MagicMock()
    builder.build.return_value = built_attn_metadata
    proposer.draft_attn_groups = [SimpleNamespace(get_metadata_builder=lambda: builder)]
    proposer.attn_layer_names = ["layer.0"]

    target_token_ids = torch.tensor([2, 3, 6, 7], dtype=torch.int32)
    target_positions = torch.tensor([5, 6, 8, 9], dtype=torch.int32)
    target_hidden_states = torch.zeros((4, 4), dtype=torch.float32)
    next_token_ids = torch.tensor([13, 17], dtype=torch.int32)
    token_indices_to_sample = torch.tensor([1, 3], dtype=torch.int32)

    common_attn_metadata = MagicMock()
    common_attn_metadata.batch_size.return_value = 2
    common_attn_metadata.query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
    common_attn_metadata.query_start_loc_cpu = torch.tensor([0, 2, 4], dtype=torch.int32)
    common_attn_metadata.seq_lens = torch.tensor([7, 10], dtype=torch.int32)
    common_attn_metadata.seq_lens_cpu = torch.tensor([7, 10], dtype=torch.int32)
    common_attn_metadata._seq_lens_cpu = torch.tensor([7, 10], dtype=torch.int32)
    common_attn_metadata.num_reqs = 2
    common_attn_metadata.num_actual_tokens = 4
    common_attn_metadata.max_query_len = 1
    common_attn_metadata.block_table_tensor = torch.arange(0, 128, dtype=torch.int32).view(2, 64)
    common_attn_metadata.slot_mapping = torch.arange(4, dtype=torch.int32)
    common_attn_metadata.causal = True
    common_attn_metadata.num_computed_tokens_cpu = None
    common_attn_metadata.positions = target_positions.clone()

    proposer.set_inputs_first_pass.return_value = (
        4,
        token_indices_to_sample,
        common_attn_metadata,
        None,
    )

    forward_context = SimpleNamespace(moe_layer_index=0)

    with (
        patch("vllm_ascend.spec_decode.llm_base_proposer.get_forward_context", return_value=forward_context),
        patch("vllm_ascend.spec_decode.llm_base_proposer.set_ascend_forward_context", return_value=nullcontext()),
        patch.object(proposer, "attn_update_stack_num_spec_norm") as mock_linear_update,
    ):
        AscendSpecDecodeBaseProposer._propose(
            proposer,
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,
            token_indices_to_sample=token_indices_to_sample,
            common_attn_metadata=common_attn_metadata,
            target_model_batch_desc=SimpleNamespace(uniform=False),
            sampling_metadata=MagicMock(),
        )

    expected = torch.tensor(
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
    assert torch.equal(
        built_attn_metadata.attn_mask[: expected.shape[0], : expected.shape[1]],
        expected,
    )
    mock_linear_update.assert_not_called()


def test_propose_tree_path_dispatches_to_run_tree_draft():
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.method = "eagle"
    proposer.num_speculative_tokens = 2
    proposer.device = torch.device("cpu")
    proposer.use_cuda_graph = False
    proposer.parallel_drafting = False
    proposer.supports_mm_inputs = False
    proposer.uses_mrope = False
    proposer.use_compress = False
    proposer.enable_enpu = False
    proposer.pcp_size = 1
    proposer.dcp_size = 1
    proposer._tree_runtime = build_spec_tree_runtime(((0,), (0, 0)))
    proposer.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(use_mla=False),
        speculative_config=SimpleNamespace(),
    )
    proposer.runner = SimpleNamespace(
        _sync_metadata_across_dp=lambda num_input_tokens, is_draft_model: (num_input_tokens, None, None),
        get_model=lambda: None,
        input_batch=SimpleNamespace(lora_id_to_lora_request={}),
    )
    proposer.slot_mapping_group = [torch.zeros(32, dtype=torch.int32) for _ in range(2)]
    proposer.seq_lens_group = [torch.zeros(32, dtype=torch.int32) for _ in range(2)]
    proposer.query_start_loc_group = [torch.zeros(32, dtype=torch.int32) for _ in range(2)]
    proposer.token_indices_to_sample = torch.zeros(32, dtype=torch.int32)
    proposer.positions = torch.zeros(32, dtype=torch.int32)
    proposer.set_inputs_first_pass = MagicMock()
    proposer._runnable = MagicMock()
    proposer._run_tree_draft = MagicMock(return_value=torch.tensor([[11, 12], [21, 22]], dtype=torch.int64))

    builder = MagicMock()
    builder.build.return_value = SimpleNamespace(causal=True, num_prefills=0)
    proposer.draft_attn_groups = [SimpleNamespace(get_metadata_builder=lambda: builder)]
    proposer.attn_layer_names = ["layer.0"]

    target_token_ids = torch.tensor([2, 3, 6, 7], dtype=torch.int32)
    target_positions = torch.tensor([5, 6, 8, 9], dtype=torch.int32)
    target_hidden_states = torch.zeros((4, 4), dtype=torch.float32)
    next_token_ids = torch.tensor([13, 17], dtype=torch.int32)
    token_indices_to_sample = torch.tensor([1, 3], dtype=torch.int32)

    common_attn_metadata = MagicMock()
    common_attn_metadata.batch_size.return_value = 2
    common_attn_metadata.query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
    common_attn_metadata.query_start_loc_cpu = torch.tensor([0, 2, 4], dtype=torch.int32)
    common_attn_metadata.seq_lens = torch.tensor([7, 10], dtype=torch.int32)
    common_attn_metadata.seq_lens_cpu = torch.tensor([7, 10], dtype=torch.int32)
    common_attn_metadata._seq_lens_cpu = torch.tensor([7, 10], dtype=torch.int32)
    common_attn_metadata.num_reqs = 2
    common_attn_metadata.num_actual_tokens = 4
    common_attn_metadata.max_query_len = 1
    common_attn_metadata.block_table_tensor = torch.arange(0, 128, dtype=torch.int32).view(2, 64)
    common_attn_metadata.slot_mapping = torch.arange(4, dtype=torch.int32)
    common_attn_metadata.causal = True
    common_attn_metadata.num_computed_tokens_cpu = None
    common_attn_metadata.positions = target_positions.clone()

    proposer.set_inputs_first_pass.return_value = (
        4,
        token_indices_to_sample,
        common_attn_metadata,
        None,
    )

    forward_context = SimpleNamespace(moe_layer_index=0)

    with (
        patch("vllm_ascend.spec_decode.llm_base_proposer.get_forward_context", return_value=forward_context),
        patch("vllm_ascend.spec_decode.llm_base_proposer.set_ascend_forward_context", return_value=nullcontext()),
        patch.object(proposer, "attn_update_stack_num_spec_norm") as mock_linear_update,
    ):
        draft_token_ids = AscendSpecDecodeBaseProposer._propose(
            proposer,
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,
            token_indices_to_sample=token_indices_to_sample,
            common_attn_metadata=common_attn_metadata,
            target_model_batch_desc=SimpleNamespace(uniform=False),
            sampling_metadata=MagicMock(),
        )

    assert draft_token_ids.tolist() == [[11, 12], [21, 22]]
    proposer._run_tree_draft.assert_called_once()
    proposer._runnable.assert_not_called()
    mock_linear_update.assert_not_called()
