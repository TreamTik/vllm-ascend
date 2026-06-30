from unittest.mock import patch

import pytest
import torch

from tests.ut.base import TestBase
from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler
from vllm_ascend.sample.rejection_sampler import (
    expand_batch_to_tokens,
    expand_pytorch,
    rejection_greedy_sample_pytorch,
    rejection_random_sample_block_verify_pytorch,
    rejection_random_sample_pytorch,
    sample_recovered_tokens_blockwise_pytorch,
    sample_recovered_tokens_pytorch,
)
from vllm_ascend.spec_decode.tree_execution import build_tree_spec_decode_metadata
from vllm_ascend.spec_decode.tree_runtime import build_spec_tree_runtime
from vllm_ascend.spec_decode.tree_verifier import pack_tree_output_tokens

# Global constants
PLACEHOLDER_TOKEN_ID = -1
GREEDY_TEMPERATURE = 0.0
MAX_SPEC_LEN = 8  # Used as MAX_NUM_TOKENS in expand_batch_to_tokens


def mock_pin_memory(original_func):
    def func_wo_pin_memory(*args, **kwargs):
        if kwargs.get("pin_memory", False):
            kwargs["pin_memory"] = False
        return original_func(*args, **kwargs)

    return func_wo_pin_memory


class TestAscendRejectionSampler(TestBase):
    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_greedy_sample_pytorch(self):
        """Test greedy rejection sampling: stop when draft doesn't match, otherwise append bonus token"""
        batch_size = 2
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2, 4])
        num_draft_tokens = [2, 2]
        draft_token_ids = torch.tensor([10, 11, 20, 21])
        target_argmax = torch.tensor([10, 99, 20, 22])
        bonus_token_ids = torch.tensor([[100], [200]])

        is_greedy = torch.tensor([True, True])

        rejection_greedy_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            num_draft_tokens,
            max_spec_len,
            is_greedy,
        )

        assert output_token_ids[0, 0].item() == 10
        assert output_token_ids[0, 1].item() == 99
        assert output_token_ids[1, 0].item() == 20
        assert output_token_ids[1, 2].item() == PLACEHOLDER_TOKEN_ID

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_random_sample_pytorch(self):
        """Test random rejection sampling: accept based on uniform probability"""
        batch_size = 2
        max_spec_len = 3
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2, 1])
        draft_token_ids = torch.tensor([1, 0, 2])
        draft_probs = torch.tensor(
            [
                [0.0, 0.6, 0.0, 0.4],  # vocab_size=4
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.5, 0.0, 0.0],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.0, 0.8, 0.0, 0.2],
                [0.2, 0.1, 0.3, 0.4],
                [0.9, 0.1, 0.0, 0.0],
            ]
        )
        bonus_token_ids = torch.tensor([[100], [200]])
        recovered_token_ids = torch.tensor([1, 2, 3])
        uniform_probs = torch.tensor([0.7, 0.6, 0.5])
        is_greedy = torch.tensor([False, False])
        vocab_size = 4

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
        )

        assert output_token_ids[0, 0].item() == 1
        assert output_token_ids[0, 1].item() == 0
        assert output_token_ids[0, 2].item() == 100

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_random_sample_pytorch_rejects_placeholder(self):
        batch_size = 1
        max_spec_len = 1
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([1])
        draft_token_ids = torch.tensor([PLACEHOLDER_TOKEN_ID])
        target_probs = torch.tensor([[0.0, 0.0, 1.0]])
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([2])
        uniform_probs = torch.tensor([0.0])
        is_greedy = torch.tensor([False])

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            None,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size=3,
            IS_NGRAM=True,
        )

        assert output_token_ids.tolist() == [[2, PLACEHOLDER_TOKEN_ID]]

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_random_sample_pytorch_rejects_all_placeholder_mtp3(self):
        batch_size = 1
        max_spec_len = 3
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([3])
        draft_token_ids = torch.tensor([PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID])
        # Placeholder draft tokens must reject regardless of target probability.
        # The recovered token is passed in after recovery sampling.
        target_probs = torch.zeros((max_spec_len, 3))
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([2, 1, 0])
        uniform_probs = torch.tensor([0.0, 0.0, 0.0])
        is_greedy = torch.tensor([False])

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            None,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size=3,
            IS_NGRAM=True,
        )

        assert output_token_ids.tolist() == [[2, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]]

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_sample_recovered_tokens_pytorch_keeps_placeholder_distribution(self):
        output_token_ids = torch.empty(1, dtype=torch.int32)
        cu_num_draft_tokens = torch.tensor([1])
        draft_token_ids = torch.tensor([PLACEHOLDER_TOKEN_ID])
        target_probs = torch.tensor([[0.1, 0.2, 0.7]])
        q = torch.ones((1, 3), dtype=torch.float32)

        sample_recovered_tokens_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            None,
            target_probs,
            q,
            vocab_size=3,
            IS_NGRAM=True,
        )

        assert output_token_ids.tolist() == [2]

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_expand_pytorch(self):
        """Test expand_pytorch functionality"""
        input_ptr = torch.tensor([10, 20, 30], dtype=torch.int32)
        cu_num_tokens_ptr = torch.tensor([2, 5, 7])
        output_ptr = torch.empty(7, dtype=torch.int32)

        expand_pytorch(
            output_ptr,
            input_ptr,
            cu_num_tokens_ptr,
            replace_from=0,
            replace_to=0,
            MAX_NUM_TOKENS=MAX_SPEC_LEN,
        )

        expected = torch.tensor([10, 10, 20, 20, 20, 30, 30])
        assert torch.equal(output_ptr, expected)

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_expand_batch_to_tokens(self):
        """Test expand_batch_to_tokens wrapper"""
        x = torch.tensor([10, 20, 30])
        cu_num_tokens = torch.tensor([2, 5, 7])
        num_tokens = 7
        # Test PyTorch path
        with (
            patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", False),
            patch("vllm_ascend.sample.rejection_sampler.expand_pytorch") as mock_pytorch,
        ):
            expand_batch_to_tokens(x, cu_num_tokens, num_tokens)
            mock_pytorch.assert_called_once()
            args = mock_pytorch.call_args[0]
            assert (args[1] == x).all()
            assert (args[2] == cu_num_tokens).all()

        # Test Triton kernel path
        with (
            patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", True),
            patch("vllm_ascend.sample.rejection_sampler.expand_triton") as mock_triton,
        ):
            expand_batch_to_tokens(x, cu_num_tokens, num_tokens)
            mock_triton.assert_called_once()
            call_args = mock_triton.call_args[0]
            assert (call_args[2] == x).all()
            assert (call_args[3] == cu_num_tokens).all()

        # Run actual function
        with patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", False):
            result = expand_batch_to_tokens(x, cu_num_tokens, num_tokens)
            expected = torch.tensor([10, 10, 20, 20, 20, 30, 30])
            assert torch.equal(result, expected)

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_sample_recovered_tokens_pytorch_ngram(self):
        """Test recovered token sampling under n-gram mode"""
        output_token_ids = torch.empty(2, dtype=torch.int32)
        cu_num_draft_tokens = torch.tensor([1, 2])
        draft_token_ids = torch.tensor([1, 2])
        draft_probs = None
        target_probs = torch.tensor(
            [
                [0.1, 0.2, 0.7],
                [0.3, 0.3, 0.4],
            ]
        )
        q = torch.tensor(
            [
                [0.1, 0.2, 0.7],
                [0.5, 0.4, 0.1],
            ]
        )
        vocab_size = 3

        sample_recovered_tokens_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q,
            vocab_size,
            IS_NGRAM=True,
        )

        assert output_token_ids[0].item() == 0
        assert output_token_ids[1].item() == 1

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_reduce_sample_recovered_tokens_pytorch_ngram(self):
        """Test recovered token sampling under n-gram mode"""
        output_token_ids = torch.empty(2, dtype=torch.int32)
        cu_num_draft_tokens = torch.tensor([1, 2])
        draft_token_ids = torch.tensor([1, 2])
        draft_probs = None
        target_probs = torch.tensor(
            [
                [0.1, 0.2, 0.7],
                [0.3, 0.3, 0.4],
            ]
        )
        q = torch.tensor(
            [
                [0.1, 0.2, 0.7],
                [0.5, 0.4, 0.1],
            ]
        )
        vocab_size = 3
        target_indices = torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 2],
            ]
        )
        enable_reduce_sampling = True
        sample_recovered_tokens_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q,
            vocab_size,
            IS_NGRAM=True,
            target_indices=target_indices,
            enable_reduce_sampling=enable_reduce_sampling,
        )

        assert output_token_ids[0].item() == 0
        assert output_token_ids[1].item() == 1

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_random_reduce_sample_block_verify_pytorch(self):
        """Test random rejection sampling for block verify: accept based on uniform probability"""
        batch_size = 2
        max_spec_len = 3
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2, 1])
        draft_token_ids = torch.tensor([1, 0, 2])
        draft_probs = torch.tensor(
            [
                [0.0, 0.6, 0.0, 0.4, 0.0],
                [0.1, 0.2, 0.3, 0.4, 0.0],
                [0.5, 0.5, 0.0, 0.0, 0.0],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.0, 0.8, 0.0, 0.2],
                [0.2, 0.1, 0.3, 0.4],
                [0.9, 0.1, 0.0, 0.0],
            ]
        )
        bonus_token_ids = torch.tensor([[100], [200]])
        recovered_token_ids = torch.tensor([1, 2, 3])
        uniform_probs = torch.tensor([0.7, 0.6, 0.5])
        is_greedy = torch.tensor([False, False])
        vocab_size = 5
        target_indices = torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ]
        )
        enable_reduce_sampling = True
        rejection_random_sample_block_verify_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
            target_indices=target_indices,
            enable_reduce_sampling=enable_reduce_sampling,
        )

        assert output_token_ids[0, 0].item() == 1
        assert output_token_ids[0, 1].item() == 0
        assert output_token_ids[0, 2].item() == 100

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_reduce_sample_recovered_tokens_blockwise_pytorch_ngram(self):
        """Test recovered token sampling for blockwise speculative decoding with n-gram."""
        output_token_ids = torch.empty(2, dtype=torch.int32)
        cu_num_draft_tokens = torch.tensor([1, 2])
        draft_token_ids = torch.tensor([1, 2])
        draft_probs = None
        target_probs = torch.tensor(
            [
                [0.1, 0.2, 0.7],
                [0.3, 0.3, 0.4],
            ]
        )
        q = torch.tensor(
            [
                [0.1, 0.2, 0.7],
                [0.5, 0.4, 0.1],
            ]
        )
        vocab_size = 3
        target_indices = torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 2],
            ]
        )
        enable_reduce_sampling = True
        sample_recovered_tokens_blockwise_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q,
            vocab_size,
            IS_NGRAM=True,
            target_indices=target_indices,
            enable_reduce_sampling=enable_reduce_sampling,
        )

        assert output_token_ids[0].item() == 0
        assert output_token_ids[1].item() == 1

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_reduce_sample_recovered_tokens_blockwise_pytorch(self):
        """Test recovered token sampling for blockwise speculative decoding."""
        output_token_ids = torch.empty(2, dtype=torch.int32)
        cu_num_draft_tokens = torch.tensor([1, 2])
        draft_token_ids = torch.tensor([0, 1])
        draft_probs = torch.tensor(
            [
                [0.6, 0.1, 0.3],
                [0.2, 0.7, 0.1],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.8, 0.1, 0.1],
                [0.3, 0.6, 0.1],
            ]
        )
        q = torch.tensor(
            [
                [0.5, 0.3, 0.2],
                [0.1, 0.8, 0.1],
            ]
        )
        vocab_size = 3
        target_indices = torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 2],
            ]
        )
        enable_reduce_sampling = True
        sample_recovered_tokens_blockwise_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q,
            vocab_size,
            IS_NGRAM=False,
            target_indices=target_indices,
            enable_reduce_sampling=enable_reduce_sampling,
        )
        assert output_token_ids[0].item() == 0
        assert output_token_ids[1].item() == 0

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_random_sample_block_verify_pytorch_standard(self):
        """Test block verify without reduce_sampling: standard full-vocab path."""
        batch_size = 2
        max_spec_len = 3
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2, 3])
        draft_token_ids = torch.tensor([1, 0, 2])
        draft_probs = torch.tensor(
            [
                [0.0, 0.6, 0.0, 0.4],
                [0.2, 0.0, 0.3, 0.5],
                [0.0, 0.0, 0.5, 0.5],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.0, 0.8, 0.0, 0.2],
                [0.1, 0.0, 0.3, 0.6],
                [0.0, 0.0, 0.9, 0.1],
            ]
        )
        bonus_token_ids = torch.tensor([[100], [200]])
        recovered_token_ids = torch.tensor([99, 88, 77])
        uniform_probs = torch.tensor([0.7, 0.6, 0.5])
        is_greedy = torch.tensor([False, False])
        vocab_size = 4

        rejection_random_sample_block_verify_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
        )

        assert output_token_ids[0, 0].item() == 1
        assert output_token_ids[0, 1].item() == 0
        assert output_token_ids[0, 2].item() == 100
        assert output_token_ids[1, 0].item() == 2
        assert output_token_ids[1, 1].item() == 200


class TestEntropyVerify(TestBase):
    """Test ENTROPY_VERIFY mode in rejection sampling.

    Entropy verify modifies the acceptance threshold based on the entropy
    of the original target distribution:
    - High entropy (uncertain) → lower effective threshold → more accepting
    - Low entropy (certain)   → higher effective threshold → stricter
    """

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_standard_high_entropy_accepts_more(self):
        """High entropy (uniform-like) makes acceptance easier via lower threshold."""
        batch_size = 2
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2, 1])
        draft_token_ids = torch.tensor([1, 0, 2])
        draft_probs = torch.tensor(
            [
                [0.6, 0.4, 0.0],
                [0.2, 0.8, 0.0],
                [0.5, 0.5, 0.0],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.8, 0.2, 0.0],
                [0.1, 0.9, 0.0],
                [0.9, 0.1, 0.0],
            ]
        )
        bonus_token_ids = torch.tensor([[100], [200]])
        recovered_token_ids = torch.tensor([99, 88, 77])
        uniform_probs = torch.tensor([0.7, 0.6, 0.5])
        is_greedy = torch.tensor([False, False])
        vocab_size = 3

        ori_target_probs = torch.tensor(
            [
                [0.8, 0.19, 0.01],
                [0.09, 0.9, 0.01],
                [0.9, 0.09, 0.01],
            ]
        )

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
            ENTROPY_VERIFY=True,
            POSTERIOR_THRESHOLD=0.95,
            POSTERIOR_ALPHA=0.4,
            EPSILON=1e-10,
            ori_target_probs=ori_target_probs,
        )

        assert output_token_ids[0, 0].item() == 99
        assert output_token_ids[0, 1].item() == -1
        assert output_token_ids[0, 2].item() == -1

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_standard_low_entropy_stricter(self):
        """Low entropy (peaked distribution) keeps threshold near POSTERIOR_THRESHOLD."""
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([1, 0])
        draft_probs = torch.tensor(
            [
                [0.6, 0.4, 0.0],
                [0.8, 0.2, 0.0],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.8, 0.2, 0.0],
                [0.1, 0.9, 0.0],
            ]
        )
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99, 88])
        uniform_probs = torch.tensor([0.7, 0.6])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        ori_target_probs = torch.tensor(
            [
                [0.8, 0.19, 0.01],
                [0.09, 0.9, 0.01],
            ]
        )

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
            ENTROPY_VERIFY=True,
            POSTERIOR_THRESHOLD=0.95,
            POSTERIOR_ALPHA=0.4,
            EPSILON=1e-10,
            ori_target_probs=ori_target_probs,
        )

        assert output_token_ids[0, 0].item() == 99
        assert output_token_ids[0, 1].item() == -1
        assert output_token_ids[0, 2].item() == -1

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_block_verify(self):
        """Entropy verify with block verify mode."""
        batch_size = 2
        max_spec_len = 3
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2, 1])
        draft_token_ids = torch.tensor([1, 0, 2])
        draft_probs = torch.tensor(
            [
                [0.6, 0.4, 0.0, 0.0],
                [0.2, 0.8, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.8, 0.2, 0.0, 0.0],
                [0.1, 0.9, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
            ]
        )
        bonus_token_ids = torch.tensor([[100], [200]])
        recovered_token_ids = torch.tensor([99, 88, 77])
        uniform_probs = torch.tensor([0.7, 0.6, 0.5])
        is_greedy = torch.tensor([False, False])
        vocab_size = 4

        ori_target_probs = torch.tensor(
            [
                [0.8, 0.18, 0.01, 0.01],
                [0.88, 0.9, 0.01, 0.01],
                [0.9, 0.08, 0.01, 0.01],
            ]
        )

        rejection_random_sample_block_verify_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
            ENTROPY_VERIFY=True,
            POSTERIOR_THRESHOLD=0.95,
            POSTERIOR_ALPHA=0.4,
            EPSILON=1e-10,
            ori_target_probs=ori_target_probs,
        )

        assert output_token_ids[0, 0].item() == 99
        assert output_token_ids[0, 1].item() == -1
        assert output_token_ids[0, 2].item() == -1

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_ngram(self):
        """ENTROPY_VERIFY with IS_NGRAM: draft_probs=None, draft_token_probs=1.0.

        In NGRAM mode, acceptance depends on target_prob alone (since
        draft_prob=1.0). Entropy verify lowers the threshold for high-entropy
        tokens, making acceptance easier when the target distribution is
        uncertain.
        """
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([0, 1])
        draft_probs = None
        target_probs = torch.tensor(
            [
                [0.6, 0.2, 0.2],
                [0.1, 0.1, 0.8],
            ]
        )
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99, 88])
        uniform_probs = torch.tensor([0.7, 0.6])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        ori_target_probs = torch.tensor(
            [
                [0.6, 0.2, 0.2],
                [0.1, 0.1, 0.8],
            ]
        )

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=True,
            ENTROPY_VERIFY=True,
            POSTERIOR_THRESHOLD=0.95,
            POSTERIOR_ALPHA=0.4,
            EPSILON=1e-10,
            ori_target_probs=ori_target_probs,
        )

        assert output_token_ids[0, 0].item() == 0
        assert output_token_ids[0, 1].item() == 88

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_block_verify_ngram(self):
        """ENTROPY_VERIFY + IS_NGRAM + block_verify combined.

        Tests the interaction of all three modes: NGRAM (draft_probs=None),
        block verify (cumulative acceptance), and entropy-based threshold
        adjustment.
        """
        batch_size = 1
        max_spec_len = 3
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([0, 1])
        draft_probs = None
        target_probs = torch.tensor(
            [
                [0.6, 0.2, 0.2, 0.0],
                [0.1, 0.1, 0.8, 0.0],
            ]
        )
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99, 88])
        uniform_probs = torch.tensor([0.7, 0.6])
        is_greedy = torch.tensor([False])
        vocab_size = 4

        ori_target_probs = torch.tensor(
            [
                [0.6, 0.2, 0.2, 0.0],
                [0.1, 0.1, 0.8, 0.0],
            ]
        )

        rejection_random_sample_block_verify_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=True,
            ENTROPY_VERIFY=True,
            POSTERIOR_THRESHOLD=0.95,
            POSTERIOR_ALPHA=0.4,
            EPSILON=1e-10,
            ori_target_probs=ori_target_probs,
        )

        assert output_token_ids[0, 0].item() == 0
        assert output_token_ids[0, 1].item() == 88

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_no_ori_probs_fallback(self):
        """When ori_target_probs is None, fallback to target_probs for entropy."""
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([1, 0])
        draft_probs = torch.tensor(
            [
                [0.6, 0.4, 0.0],
                [0.8, 0.2, 0.0],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.35, 0.33, 0.32],
                [0.34, 0.34, 0.32],
            ]
        )
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99, 88])
        uniform_probs = torch.tensor([0.7, 0.6])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
            ENTROPY_VERIFY=True,
            POSTERIOR_THRESHOLD=0.95,
            POSTERIOR_ALPHA=0.4,
            EPSILON=1e-10,
            ori_target_probs=None,
        )

        assert output_token_ids[0, 0].item() == 1
        assert output_token_ids[0, 1].item() in (0, 88)
        assert output_token_ids[0, 2].item() == 100


def test_tree_output_tokens_pack_to_existing_layout():
    accepted = (
        type("Accepted", (), {"token_ids": (7, 8), "bonus_token_id": 9}),
    )

    packed = pack_tree_output_tokens(accepted, max_path_len=2, placeholder_token_id=-1)

    assert packed.shape == (1, 3)
    assert packed.tolist() == [[7, 8, 9]]


def test_tree_sampler_forward_returns_existing_packed_tensor_shape():
    runtime = build_spec_tree_runtime(((0,), (1,), (0, 0), (0, 1), (0, 1, 0), (1, 0)))
    metadata = build_tree_spec_decode_metadata(runtime, batch_size=1)
    object.__setattr__(metadata, "draft_token_ids", torch.tensor([11, 12, 21, 22, 31, 41], dtype=torch.int32))
    object.__setattr__(metadata, "num_draft_tokens", [6])
    object.__setattr__(metadata, "cu_num_draft_tokens", torch.tensor([6], dtype=torch.int32))
    object.__setattr__(metadata, "target_logits_indices", torch.arange(6, dtype=torch.int32))
    object.__setattr__(metadata, "bonus_logits_indices", torch.tensor([6], dtype=torch.int32))
    object.__setattr__(metadata, "logits_indices", torch.arange(7, dtype=torch.int32))

    sampler = AscendRejectionSampler.__new__(AscendRejectionSampler)
    sampler.sampler = lambda **kwargs: type(
        "BonusOut",
        (),
        {"sampled_token_ids": torch.tensor([[101]], dtype=torch.int32), "logprobs_tensors": None},
    )()
    sampler.top_k = None
    sampler.is_processed_logprobs_mode = False

    logits = torch.full((7, 128), -1.0, dtype=torch.float32)
    logits[0, 11] = 1.0
    logits[1, 12] = 1.0
    logits[2, 21] = 1.0
    logits[3, 22] = 1.0
    logits[4, 31] = 1.0
    logits[5, 99] = 1.0
    logits[6, 101] = 1.0

    sampling_metadata = type(
        "SamplingMetadataStub",
        (),
            {
                "no_penalties": True,
                "bad_words_token_ids": None,
                "allowed_token_ids_mask": None,
                "output_token_ids": [],
                "spec_token_ids": None,
                "logitsprocs": type("LogitsProcsStub", (), {"non_argmax_invariant": []})(),
                "all_greedy": True,
            },
        )()

    output = AscendRejectionSampler.forward(
        sampler,
        metadata=metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=sampling_metadata,
    )

    assert output.sampled_token_ids.shape == (1, 4)
    assert output.sampled_token_ids.tolist() == [[11, 22, 31, 101]]


def test_tree_sampler_forward_rejects_non_greedy_mode():
    runtime = build_spec_tree_runtime(((0,),))
    metadata = build_tree_spec_decode_metadata(runtime, batch_size=1)
    object.__setattr__(metadata, "draft_token_ids", torch.tensor([11], dtype=torch.int32))
    object.__setattr__(metadata, "num_draft_tokens", [1])
    object.__setattr__(metadata, "cu_num_draft_tokens", torch.tensor([1], dtype=torch.int32))
    object.__setattr__(metadata, "target_logits_indices", torch.tensor([0], dtype=torch.int32))
    object.__setattr__(metadata, "bonus_logits_indices", torch.tensor([1], dtype=torch.int32))
    object.__setattr__(metadata, "logits_indices", torch.tensor([0, 1], dtype=torch.int32))

    sampler = AscendRejectionSampler.__new__(AscendRejectionSampler)
    sampler.sampler = lambda **kwargs: None
    sampler.top_k = None
    sampler.is_processed_logprobs_mode = False

    logits = torch.zeros((2, 32), dtype=torch.float32)
    sampling_metadata = type(
        "SamplingMetadataStub",
        (),
        {
            "no_penalties": True,
            "bad_words_token_ids": None,
            "allowed_token_ids_mask": None,
            "output_token_ids": [],
            "spec_token_ids": None,
            "logitsprocs": type("LogitsProcsStub", (), {"non_argmax_invariant": []})(),
            "all_greedy": False,
        },
    )()

    with pytest.raises(NotImplementedError, match="greedy"):
        AscendRejectionSampler.forward(
            sampler,
            metadata=metadata,
            draft_probs=None,
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
