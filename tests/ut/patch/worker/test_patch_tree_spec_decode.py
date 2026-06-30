from types import SimpleNamespace
from unittest.mock import patch

from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.patch.worker.patch_tree_spec_decode import (
    _ORIGINAL_CREATE_SPECULATIVE_DRAFTER,
    _patched_create_speculative_drafter,
)


def _make_runner(spec_config):
    runner = object.__new__(GPUModelRunner)
    runner.speculative_config = spec_config
    runner.vllm_config = object()
    runner.device = object()
    return runner


def test_tree_spec_decode_patch_uses_ascend_eagle_proposer():
    spec_config = SimpleNamespace(
        method="eagle3",
        use_speculative_token_tree=lambda: True,
    )
    runner = _make_runner(spec_config)

    with patch(
        "vllm_ascend.patch.worker.patch_tree_spec_decode.AscendEagleProposer",
        return_value="ascend-drafter",
    ) as mock_proposer:
        drafter = _patched_create_speculative_drafter(runner)

    assert drafter == "ascend-drafter"
    mock_proposer.assert_called_once_with(runner.vllm_config, runner.device, runner)


def test_tree_spec_decode_patch_delegates_when_tree_disabled():
    spec_config = SimpleNamespace(
        method="eagle3",
        use_speculative_token_tree=lambda: False,
    )
    runner = _make_runner(spec_config)

    with patch.object(
        _ORIGINAL_CREATE_SPECULATIVE_DRAFTER.__self__ if hasattr(_ORIGINAL_CREATE_SPECULATIVE_DRAFTER, "__self__") else GPUModelRunner,
        "_create_speculative_drafter",
        return_value="upstream-drafter",
    ):
        pass

    with patch(
        "vllm_ascend.patch.worker.patch_tree_spec_decode._ORIGINAL_CREATE_SPECULATIVE_DRAFTER",
        return_value="upstream-drafter",
    ) as mock_original:
        drafter = _patched_create_speculative_drafter(runner)

    assert drafter == "upstream-drafter"
    mock_original.assert_called_once_with(runner)
