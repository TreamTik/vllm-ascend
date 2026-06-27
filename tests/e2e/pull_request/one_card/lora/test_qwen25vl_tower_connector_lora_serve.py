from __future__ import annotations

import multiprocessing
import queue
import traceback

from vllm.multimodal.utils import encode_image_url
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free
from tests.e2e.pull_request.one_card.lora.test_tower_connector_lora_utils import (
    build_temp_lora_dir,
    create_synthetic_tower_connector_lora,
    get_qwen25vl_model_path,
    load_local_test_image,
    set_tower_connector_test_threads,
)

ONLINE_GPU_MEMORY_UTILIZATION = "0.85"


def _run_online_serve_case(result_queue: multiprocessing.queues.Queue) -> None:
    try:
        with build_temp_lora_dir() as lora_dir:
            create_synthetic_tower_connector_lora(lora_dir)

            set_tower_connector_test_threads()
            port = get_open_port()
            server_args = [
                "--trust-remote-code",
                "--enable-lora",
                "--enable-tower-connector-lora",
                "--enforce-eager",
                "--max-model-len",
                "4096",
                "--max-num-seqs",
                "1",
                "--max-loras",
                "2",
                "--max-lora-rank",
                "8",
                "--gpu-memory-utilization",
                ONLINE_GPU_MEMORY_UTILIZATION,
                "--mm-processor-cache-gb",
                "0",
                "--limit-mm-per-prompt",
                '{"image": 1}',
                "--lora-modules",
                f"vision_tower_connector={lora_dir}",
                "--port",
                str(port),
            ]
            env_dict = {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
            }

            with RemoteOpenAIServer(
                get_qwen25vl_model_path(),
                server_args,
                server_host="127.0.0.1",
                server_port=port,
                env_dict=env_dict,
                auto_port=False,
                max_wait_seconds=1800,
            ) as server:
                client = server.get_client()
                image_url = encode_image_url(load_local_test_image())
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": "Describe the image briefly."},
                        ],
                    }
                ]

                response = client.chat.completions.create(
                    model="vision_tower_connector",
                    messages=messages,
                    max_tokens=24,
                    temperature=0.0,
                )

                assert response.choices
                content = response.choices[0].message.content
                assert content is not None
                assert content.strip()
                result_queue.put(("ok", content))
    except BaseException:
        result_queue.put(("err", traceback.format_exc()))
        raise


@wait_until_npu_memory_free(target_free_percentage=0.9, max_wait_seconds=300)
def test_qwen25vl_tower_connector_lora_openai_serve():
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_run_online_serve_case,
        args=(result_queue,),
    )

    proc.start()
    proc.join(timeout=1800)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=10)
        raise AssertionError("Timed out waiting for online tower/connector LoRA test subprocess.")

    try:
        status, payload = result_queue.get_nowait()
    except queue.Empty:
        status, payload = ("err", "No result returned from online tower/connector LoRA test subprocess.")

    assert proc.exitcode == 0, payload
    assert status == "ok", payload
