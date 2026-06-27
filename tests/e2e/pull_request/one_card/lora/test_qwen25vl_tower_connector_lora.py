from __future__ import annotations

import multiprocessing
import queue
import traceback
from unittest.mock import patch

from vllm import SamplingParams

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.one_card.lora.test_tower_connector_lora_utils import (
    PROMPT,
    build_llm,
    build_temp_lora_dir,
    create_lora_request,
    create_synthetic_tower_connector_lora,
    get_qwen25vl_model_path,
    load_local_test_image,
    shutdown_llm,
)


def _run_offline_case(result_queue: multiprocessing.queues.Queue) -> None:
    llm = None
    with build_temp_lora_dir() as lora_dir:
        try:
            llm = build_llm(get_qwen25vl_model_path())
            create_synthetic_tower_connector_lora(lora_dir)

            lora_request = create_lora_request(lora_dir)
            assert llm.llm_engine.add_lora(lora_request)
            assert llm.llm_engine.list_loras() == {1}

            image = load_local_test_image()
            inputs = [{"prompt": PROMPT, "multi_modal_data": {"image": image}}]
            sampling_params = SamplingParams(temperature=0.0, max_tokens=16)

            base_outputs = llm.generate(inputs, sampling_params, use_tqdm=False)
            lora_outputs = llm.generate(
                inputs,
                sampling_params,
                lora_request=lora_request,
                use_tqdm=False,
            )

            base_text = base_outputs[0].outputs[0].text
            lora_text = lora_outputs[0].outputs[0].text

            assert base_text.strip()
            assert lora_text.strip()
            result_queue.put(("ok", (base_text, lora_text)))
        except BaseException:
            result_queue.put(("err", traceback.format_exc()))
            raise
        finally:
            shutdown_llm(llm)


@patch.dict("os.environ", {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
@wait_until_npu_memory_free(target_free_percentage=0.9, max_wait_seconds=300)
def test_qwen25vl_tower_connector_lora_offline():
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_run_offline_case,
        args=(result_queue,),
    )

    proc.start()
    proc.join(timeout=1800)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=10)
        raise AssertionError("Timed out waiting for offline tower/connector LoRA test subprocess.")

    try:
        status, payload = result_queue.get_nowait()
    except queue.Empty:
        status, payload = ("err", "No result returned from offline tower/connector LoRA test subprocess.")

    assert proc.exitcode == 0, payload
    assert status == "ok", payload
