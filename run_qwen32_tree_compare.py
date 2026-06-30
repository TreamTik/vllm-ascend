import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path


UPSTREAM_ROOT = Path("/root/worktrees/vllm-spec-tree-v023")
ASCEND_ROOT = Path("/root/worktrees/vllm-ascend-clean")
CANN_ROOT = Path("/usr/local/Ascend/cann-9.0.0")
ATB_HOME = Path("/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1")
ARCH = os.uname().machine
CANN_BIN = [
    str(CANN_ROOT / "bin"),
    str(CANN_ROOT / "tools/ccec_compiler/bin"),
    str(CANN_ROOT / "tools/bishengir/bin"),
    str(CANN_ROOT / "tools/profiler/bin"),
    str(CANN_ROOT / "tools/ascend_system_advisor/asys"),
    str(CANN_ROOT / "tools/show_kernel_debug_data"),
    str(CANN_ROOT / "tools/msobjdump"),
]

BASE_MODEL = "/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-32B-W8A8"
DRAFT_MODEL = (
    "/root/.cache/modelscope/hub/models/RedHatAI/"
    "Qwen3-32B-speculator.eagle3"
)
RESULT_PATH = Path("/tmp/qwen32_tree_compare_result.json")

PROMPT = (
    "小明有 3 个盒子。第一个盒子里有 5 个红球和 2 个蓝球，"
    "第二个盒子里有 4 个红球和 4 个蓝球，"
    "第三个盒子里有 1 个红球和 6 个蓝球。"
    "如果随机选择一个盒子，再从中随机取出一个球，取到红球的概率是多少？"
    "请严格按这个格式回答："
    "第一句只写最终分数结果；"
    "第二句写总概率公式；"
    "第三句写三个盒子的单独红球概率。"
    "不要换行，也不要复述题目。"
)
MESSAGES = [{"role": "user", "content": PROMPT}]
MAX_TOKENS = 96
STOP = ["\n", "<|im_end|>"]
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

TREE_SPECULATIVE_CONFIG = {
    "method": "eagle3",
    "model": DRAFT_MODEL,
    "draft_tensor_parallel_size": 1,
    "num_speculative_tokens": 5,
    "speculative_token_tree": "[(0,), (1,), (0, 0), (0, 1), (1, 0)]",
}

COMMON_LLM_KWARGS = {
    "model": BASE_MODEL,
    "tensor_parallel_size": 4,
    "quantization": "ascend",
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.9,
    "disable_log_stats": False,
    "enforce_eager": True,
}


def _merge_env_paths(current_value: str, wanted_paths: list[str]) -> str:
    current = [p for p in current_value.split(":") if p]
    merged: list[str] = []
    for path in wanted_paths + current:
        if path and path not in merged:
            merged.append(path)
    return ":".join(merged)


def _bootstrap_runtime_env() -> None:
    python_paths = [
        str(UPSTREAM_ROOT),
        str(ASCEND_ROOT),
        str(CANN_ROOT / "python/site-packages"),
        str(CANN_ROOT / "opp/built-in/op_impl/ai_core/tbe"),
    ]
    ld_library_paths = [
        str(ATB_HOME / "lib"),
        str(CANN_ROOT / "lib64"),
        str(CANN_ROOT / "lib64/plugin/opskernel"),
        str(CANN_ROOT / "lib64/plugin/nnengine"),
        str(CANN_ROOT / f"opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/{ARCH}"),
        "/usr/local/Ascend/driver/lib64",
        "/usr/local/Ascend/driver/lib64/common",
        "/usr/local/Ascend/driver/lib64/driver",
    ]

    old_pythonpath = os.environ.get("PYTHONPATH", "")
    old_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    old_path = os.environ.get("PATH", "")
    new_pythonpath = _merge_env_paths(old_pythonpath, python_paths)
    new_ld_library_path = _merge_env_paths(old_ld_library_path, ld_library_paths)
    new_path = _merge_env_paths(old_path, CANN_BIN)

    os.environ["PYTHONPATH"] = new_pythonpath
    os.environ["LD_LIBRARY_PATH"] = new_ld_library_path
    os.environ["PATH"] = new_path
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "1800")
    os.environ.setdefault("ATB_HOME_PATH", str(ATB_HOME))
    os.environ.setdefault("ASCEND_HOME_PATH", str(CANN_ROOT))
    os.environ.setdefault("ASCEND_TOOLKIT_HOME", str(CANN_ROOT))
    os.environ.setdefault("ASCEND_OPP_PATH", str(CANN_ROOT / "opp"))
    os.environ.setdefault("ASCEND_AICPU_PATH", str(CANN_ROOT))
    os.environ.setdefault("TOOLCHAIN_HOME", str(CANN_ROOT / "toolkit"))
    os.environ.setdefault("ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE", "0")
    os.environ.setdefault("ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE", "0")
    os.environ.setdefault("ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE", "0")
    os.environ.setdefault("ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT", "1")
    os.environ.setdefault("ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT", "5")
    os.environ.setdefault("ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE", "1")
    os.environ.setdefault("ATB_COMPARE_TILING_EVERY_KERNEL", "0")
    os.environ.setdefault("ATB_SHARE_MEMORY_NAME_SUFFIX", "")
    os.environ.setdefault("ATB_MATMUL_SHUFFLE_K_ENABLE", "1")
    os.environ.setdefault("LCCL_DETERMINISTIC", "0")
    os.environ.setdefault("LCCL_PARALLEL", "0")

    for path in reversed(python_paths):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    if os.environ.get("_QWEN32_TREE_BOOTSTRAPPED") != "1":
        os.environ["_QWEN32_TREE_BOOTSTRAPPED"] = "1"
        if (
            old_pythonpath != new_pythonpath
            or old_ld_library_path != new_ld_library_path
            or old_path != new_path
        ):
            os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)


_bootstrap_runtime_env()

import acl  # noqa: E402
import torch  # noqa: E402
import vllm  # noqa: E402
import vllm_ascend  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.distributed.parallel_state import (  # noqa: E402
    destroy_distributed_environment,
    destroy_model_parallel,
)
from vllm.v1.metrics.reader import Counter, Vector  # noqa: E402


def cleanup() -> None:
    gc.collect()
    try:
        destroy_model_parallel()
    except Exception:
        pass
    try:
        destroy_distributed_environment()
    except Exception:
        pass
    try:
        torch.npu.empty_cache()
    except Exception:
        pass


def summarize_metrics(metrics, num_speculative_tokens: int | None) -> dict:
    summary: dict[str, object] = {}
    if not metrics:
        return summary

    num_drafts = 0
    accepted_per_pos: list[float] | None = None

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts" and isinstance(metric, Counter):
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos" and isinstance(metric, Vector):
            accepted_per_pos = list(metric.values)
        elif isinstance(metric, Counter):
            summary[metric.name] = metric.value
        elif isinstance(metric, Vector):
            summary[metric.name] = list(metric.values)

    if accepted_per_pos is not None and num_drafts:
        limit = len(accepted_per_pos)
        if num_speculative_tokens is not None:
            limit = min(limit, num_speculative_tokens)
        summary["acceptance_per_pos"] = [
            accepted_per_pos[idx] / num_drafts for idx in range(limit)
        ]
        summary["num_drafts"] = num_drafts

    return summary


def collect_metrics(llm) -> tuple[list | None, str | None]:
    try:
        return llm.get_metrics(), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def compare_token_sequences(
    baseline_token_ids: list[int], tree_token_ids: list[int]
) -> dict[str, object]:
    first_diff_index = None
    for idx, (baseline_token_id, tree_token_id) in enumerate(
        zip(baseline_token_ids, tree_token_ids)
    ):
        if baseline_token_id != tree_token_id:
            first_diff_index = idx
            break

    if first_diff_index is None and len(baseline_token_ids) != len(tree_token_ids):
        first_diff_index = min(len(baseline_token_ids), len(tree_token_ids))

    return {
        "same_token_ids": baseline_token_ids == tree_token_ids,
        "first_diff_index": first_diff_index,
        "baseline_token_id_at_diff": (
            baseline_token_ids[first_diff_index]
            if first_diff_index is not None and first_diff_index < len(baseline_token_ids)
            else None
        ),
        "tree_token_id_at_diff": (
            tree_token_ids[first_diff_index]
            if first_diff_index is not None and first_diff_index < len(tree_token_ids)
            else None
        ),
    }


def run_case(
    mode: str,
    messages: list[dict[str, str]],
    rendered_prompt: str,
    speculative_config: dict | None = None,
) -> dict:
    llm = None
    case_start = time.time()
    load_done = case_start
    old_sample_trace_label = os.environ.get("VLLM_ASCEND_SAMPLE_TRACE_LABEL")
    try:
        print(f"[run] starting mode={mode}", flush=True)
        if os.environ.get("VLLM_ASCEND_SAMPLE_TRACE_PATH"):
            os.environ["VLLM_ASCEND_SAMPLE_TRACE_LABEL"] = mode
        llm_kwargs = dict(COMMON_LLM_KWARGS)
        if speculative_config is not None:
            llm_kwargs["speculative_config"] = speculative_config

        llm = LLM(**llm_kwargs)
        load_done = time.time()
        print(
            f"[run] loaded mode={mode} load_seconds={load_done - case_start:.2f}",
            flush=True,
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            stop=STOP,
        )
        outputs = llm.chat(
            messages,
            sampling_params,
            use_tqdm=False,
            chat_template_kwargs=CHAT_TEMPLATE_KWARGS,
        )
        gen_done = time.time()
        metrics, metrics_error = collect_metrics(llm)

        sample = outputs[0].outputs[0]
        result = {
            "status": "ok",
            "mode": mode,
            "load_seconds": load_done - case_start,
            "generate_seconds": gen_done - load_done,
            "total_seconds": gen_done - case_start,
            "token_count": len(sample.token_ids),
            "token_ids": list(sample.token_ids),
            "text": sample.text,
            "messages": messages,
            "rendered_prompt": rendered_prompt,
            "max_tokens": MAX_TOKENS,
            "stop": STOP,
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
            "metrics": summarize_metrics(
                metrics,
                speculative_config.get("num_speculative_tokens")
                if speculative_config
                else None,
            ),
        }
        if metrics_error is not None:
            result["metrics_error"] = metrics_error
        print(
            json.dumps(
                {
                    "mode": mode,
                    "status": "ok",
                    "load_seconds": round(result["load_seconds"], 2),
                    "generate_seconds": round(result["generate_seconds"], 2),
                    "token_count": result["token_count"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return result
    except Exception:
        return {
            "status": "error",
            "mode": mode,
            "load_seconds": load_done - case_start,
            "total_seconds": time.time() - case_start,
            "traceback": traceback.format_exc(),
        }
    finally:
        if old_sample_trace_label is None:
            os.environ.pop("VLLM_ASCEND_SAMPLE_TRACE_LABEL", None)
        else:
            os.environ["VLLM_ASCEND_SAMPLE_TRACE_LABEL"] = old_sample_trace_label
        if llm is not None:
            del llm
        cleanup()


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
    rendered_prompt = tokenizer.apply_chat_template(
        MESSAGES,
        tokenize=False,
        add_generation_prompt=True,
        **CHAT_TEMPLATE_KWARGS,
    )
    result = {
        "meta": {
            "cwd": os.getcwd(),
            "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "vllm_file": vllm.__file__,
            "vllm_ascend_file": vllm_ascend.__file__,
            "acl_file": getattr(acl, "__file__", "built-in"),
            "base_model": BASE_MODEL,
            "draft_model": DRAFT_MODEL,
        },
        "prompt": PROMPT,
        "messages": MESSAGES,
        "rendered_prompt": rendered_prompt,
        "max_tokens": MAX_TOKENS,
        "stop": STOP,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        "baseline": {"status": "starting"},
        "tree": {"status": "pending"},
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    baseline = run_case("baseline", MESSAGES, rendered_prompt)
    result["baseline"] = baseline
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    tree = run_case("tree-eagle3", MESSAGES, rendered_prompt, TREE_SPECULATIVE_CONFIG)
    result["tree"] = tree
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    if baseline.get("status") == "ok" and tree.get("status") == "ok":
        result["comparison"] = compare_token_sequences(
            baseline["token_ids"], tree["token_ids"]
        )
        result["comparison"]["same_text"] = baseline["text"] == tree["text"]
        RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print(f"[done] wrote result to {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
