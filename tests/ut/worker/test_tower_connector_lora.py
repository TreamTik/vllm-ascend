from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.lora.layers import LoRAMappingType
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def _build_runner(*, supports_tower_connector_lora: bool) -> NPUModelRunner:
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.device = torch.device("cpu")
    runner.pin_memory = False
    runner.observability_config = None
    runner.encoder_cudagraph_manager = None
    runner.is_multimodal_pruning_enabled = False
    runner.requires_sequential_video_encoding = False
    runner.encoder_cache = {}
    runner.maybe_save_ec_to_connector = MagicMock()
    runner.lora_config = SimpleNamespace(enable_tower_connector_lora=True)
    runner.input_batch = SimpleNamespace(
        req_id_to_index={
            "req-with-lora": 0,
            "req-no-lora": 1,
        },
        request_lora_mapping=torch.tensor([7, 0], dtype=torch.int64),
        lora_id_to_lora_request={
            7: LoRARequest("vision-lora", 7, "/tmp/vision-lora"),
        },
    )
    runner.model = MagicMock()
    runner.model.get_num_mm_encoder_tokens.side_effect = lambda num_embeds: num_embeds + 1
    runner.model.get_mm_mapping.return_value = SimpleNamespace(connector=True)
    runner.model.get_num_mm_connector_tokens.side_effect = lambda num_tokens: num_tokens + 10
    runner.model.embed_multimodal.return_value = [
        torch.ones((7, 4)),
        torch.ones((6, 4)),
        torch.ones((2, 4)),
    ]
    runner.lora_manager = MagicMock()
    runner.lora_manager.supports_tower_connector_lora.return_value = supports_tower_connector_lora
    runner.requests = {
        "req-with-lora": SimpleNamespace(
            mm_features=[
                MultiModalFeatureSpec(
                    data={"image": SimpleNamespace(data=torch.ones((1, 1)))},
                    modality="image",
                    identifier="img-0",
                    mm_position=PlaceholderRange(offset=0, length=2),
                ),
                MultiModalFeatureSpec(
                    data={"image": SimpleNamespace(data=torch.ones((1, 1)))},
                    modality="image",
                    identifier="img-1",
                    mm_position=PlaceholderRange(offset=2, length=3),
                ),
            ]
        ),
        "req-no-lora": SimpleNamespace(
            mm_features=[
                MultiModalFeatureSpec(
                    data={"image": SimpleNamespace(data=torch.ones((1, 1)))},
                    modality="image",
                    identifier="img-2",
                    mm_position=PlaceholderRange(offset=0, length=1),
                ),
            ]
        ),
    }
    return runner


def _scheduler_output():
    return SimpleNamespace(
        scheduled_encoder_inputs={
            "req-with-lora": [1, 0],
            "req-no-lora": [0],
        }
    )


def _grouped_mm_batch(_mm_kwargs, *, device, pin_memory):
    del device, pin_memory
    yield "image", 3, {"image": torch.ones((3, 1, 1))}


def test_execute_mm_encoder_builds_tower_and_connector_mappings():
    runner = _build_runner(supports_tower_connector_lora=True)

    with patch(
        "vllm.v1.worker.gpu_model_runner.group_and_batch_mm_kwargs",
        side_effect=_grouped_mm_batch,
    ):
        outputs = runner._execute_mm_encoder(_scheduler_output())

    assert len(outputs) == 3
    assert runner.lora_manager.set_active_adapters.call_count == 2

    tower_requests, tower_mapping = runner.lora_manager.set_active_adapters.call_args_list[0].args
    assert tower_requests == {LoRARequest("vision-lora", 7, "/tmp/vision-lora")}
    assert tower_mapping.type is LoRAMappingType.TOWER
    assert tower_mapping.prompt_mapping == (7, 7, 0)
    assert tower_mapping.index_mapping == (7, 7, 7, 7, 7, 7, 7, 0, 0)

    connector_requests, connector_mapping = runner.lora_manager.set_active_adapters.call_args_list[1].args
    assert connector_requests == {LoRARequest("vision-lora", 7, "/tmp/vision-lora")}
    assert connector_mapping.type is LoRAMappingType.CONNECTOR
    assert connector_mapping.prompt_mapping == (7, 7, 0)
    assert connector_mapping.index_mapping == ((7,) * 14 + (7,) * 13 + (0,) * 12)


def test_execute_mm_encoder_skips_mm_lora_activation_when_not_supported():
    runner = _build_runner(supports_tower_connector_lora=False)

    with patch(
        "vllm.v1.worker.gpu_model_runner.group_and_batch_mm_kwargs",
        side_effect=_grouped_mm_batch,
    ):
        outputs = runner._execute_mm_encoder(_scheduler_output())

    assert len(outputs) == 3
    runner.lora_manager.set_active_adapters.assert_not_called()
