import gc
import json
import os
import tempfile
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file
from vllm import LLM
from vllm.lora.request import LoRARequest

from tests.e2e.conftest import cleanup_dist_env_and_memory

MODEL_ENV_VAR = "VLLM_ASCEND_QWEN25VL_MODEL"
DEFAULT_MODEL_PATH = "/data/codex-persist/home/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-3B-Instruct"
LOCAL_IMAGE_PATH = (
    Path(__file__).resolve().parents[3] / "prompts" / "qwen.png"
)

PROMPT = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "What is in the image?<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def get_qwen25vl_model_path() -> str:
    model_path = os.getenv(MODEL_ENV_VAR, DEFAULT_MODEL_PATH)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Qwen2.5-VL model path does not exist: {model_path}. "
            f"Set {MODEL_ENV_VAR} if needed."
        )
    return model_path


def set_tower_connector_test_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def load_local_test_image() -> Image.Image:
    return Image.open(LOCAL_IMAGE_PATH).convert("RGB")


def build_llm(model_path: str) -> LLM:
    set_tower_connector_test_threads()
    return LLM(
        model=model_path,
        enable_lora=True,
        enable_tower_connector_lora=True,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=1,
        max_loras=2,
        max_lora_rank=8,
        tensor_parallel_size=1,
        mm_processor_cache_gb=0,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        disable_log_stats=True,
    )


def shutdown_llm(llm: LLM | None) -> None:
    if llm is None:
        return
    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()
    cleanup_dist_env_and_memory()


def create_synthetic_tower_connector_lora(
    tmp_dir: str,
    *,
    rank: int = 8,
    seed: int = 20260626,
) -> str:
    target_specs = [
        ("qkv", "visual.blocks.0.attn.qkv", 1280, 3840),
        ("down_proj", "visual.blocks.0.mlp.down_proj", 3420, 1280),
        ("0", "visual.merger.mlp.0", 5120, 5120),
        ("2", "visual.merger.mlp.2", 5120, 2048),
    ]

    gen = torch.Generator(device="cpu").manual_seed(seed)
    weights: dict[str, torch.Tensor] = {}
    target_modules: list[str] = []

    for target_suffix, module_name, in_features, out_features in target_specs:
        lora_a = torch.randn(
            (rank, in_features),
            generator=gen,
            dtype=torch.float16,
        ) * 0.35
        lora_b = torch.randn(
            (out_features, rank),
            generator=gen,
            dtype=torch.float16,
        ) * 0.35
        key_prefix = f"base_model.model.model.{module_name}"
        weights[f"{key_prefix}.lora_A.weight"] = lora_a.contiguous()
        weights[f"{key_prefix}.lora_B.weight"] = lora_b.contiguous()
        target_modules.append(target_suffix)

    adapter_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": get_qwen25vl_model_path(),
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "r": rank,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": target_modules,
        "use_rslora": False,
        "use_dora": False,
    }

    os.makedirs(tmp_dir, exist_ok=True)
    with open(
        os.path.join(tmp_dir, "adapter_config.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(adapter_config, f)
    save_file(weights, os.path.join(tmp_dir, "adapter_model.safetensors"))
    return tmp_dir


def create_lora_request(lora_path: str) -> LoRARequest:
    return LoRARequest("vision_tower_connector", 1, lora_path)


def build_temp_lora_dir() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(prefix="tower-connector-lora-")
