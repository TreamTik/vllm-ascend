import json
import os

import torch
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.qwen3 import Qwen3Attention, Qwen3DecoderLayer
from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention, Qwen3MoeDecoderLayer
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLForConditionalGeneration,
    pos_embed_interpolate_native,
)

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.rotary_embedding import AscendMRotaryEmbedding


_QWEN3_ATTN_TRACE_COUNT = 0
_QWEN3_BLOCK_TRACE_COUNT = 0


def _maybe_dump_qwen3_attn_trace(self) -> None:
    trace_path = os.getenv("VLLM_ASCEND_QWEN3_ATTN_TRACE_PATH")
    if not trace_path:
        return

    global _QWEN3_ATTN_TRACE_COUNT
    max_steps = int(os.getenv("VLLM_ASCEND_QWEN3_ATTN_TRACE_STEPS", "32"))
    if _QWEN3_ATTN_TRACE_COUNT >= max_steps:
        return

    attn_backend = None
    get_backend = getattr(self.attn, "get_attn_backend", None)
    if callable(get_backend):
        try:
            backend_cls = get_backend()
            attn_backend = backend_cls.get_name() if hasattr(backend_cls, "get_name") else str(backend_cls)
        except Exception as exc:  # pragma: no cover - debug trace only
            attn_backend = f"<error:{type(exc).__name__}:{exc}>"

    payload = {
        "index": _QWEN3_ATTN_TRACE_COUNT,
        "pid": os.getpid(),
        "module": type(self).__name__,
        "attn_wrapper_cls": type(self.attn).__name__,
        "attn_impl_cls": type(getattr(self.attn, "impl", None)).__name__,
        "attn_backend": attn_backend,
        "layer_name": getattr(self.attn, "layer_name", None),
    }
    with open(trace_path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _QWEN3_ATTN_TRACE_COUNT += 1


def _matches_trace_filters(layer_name: str | None, filter_spec: str | None) -> bool:
    if not filter_spec:
        return True
    if layer_name is None:
        return False
    filters = [item.strip() for item in filter_spec.split(",") if item.strip()]
    if not filters:
        return True
    return any(layer_name == item or layer_name.startswith(f"{item}.") for item in filters)


def _maybe_dump_qwen3_block_trace(
    self,
    positions: torch.Tensor,
    layer_input_hidden: torch.Tensor,
    layer_input_residual: torch.Tensor | None,
    pre_attn_hidden: torch.Tensor,
    attn_output: torch.Tensor,
    post_attn_hidden: torch.Tensor,
    post_attn_residual: torch.Tensor,
    mlp_output: torch.Tensor,
) -> None:
    trace_path = os.getenv("VLLM_ASCEND_QWEN3_BLOCK_TRACE_PATH")
    if not trace_path:
        return

    tp_rank = get_tensor_model_parallel_rank()
    if os.getenv("VLLM_ASCEND_QWEN3_BLOCK_TRACE_ALL_TP", "0") != "1" and tp_rank != 0:
        return

    layer_name = getattr(getattr(self.self_attn, "attn", None), "layer_name", None)
    block_name = layer_name.removesuffix(".self_attn.attn") if layer_name else None
    if not _matches_trace_filters(block_name, os.getenv("VLLM_ASCEND_QWEN3_BLOCK_TRACE_LAYERS")):
        return

    global _QWEN3_BLOCK_TRACE_COUNT
    max_steps = int(os.getenv("VLLM_ASCEND_QWEN3_BLOCK_TRACE_STEPS", "128"))
    if _QWEN3_BLOCK_TRACE_COUNT >= max_steps:
        return

    try:
        forward_context = get_forward_context()
    except AssertionError:
        return

    attn_metadata_raw = getattr(forward_context, "attn_metadata", None)
    if attn_metadata_raw is None:
        return

    if isinstance(attn_metadata_raw, dict):
        attn_metadata = attn_metadata_raw.get(layer_name)
    elif isinstance(attn_metadata_raw, list):
        return
    else:
        attn_metadata = attn_metadata_raw

    if attn_metadata is None:
        return

    slot_mapping = getattr(attn_metadata, "slot_mapping", None)
    actual_seq_lengths_q = getattr(attn_metadata, "actual_seq_lengths_q", None)
    seq_lens = getattr(attn_metadata, "seq_lens_list", None)
    if slot_mapping is None or not actual_seq_lengths_q or seq_lens is None:
        return

    max_q_len = max(
        q_end - (actual_seq_lengths_q[idx - 1] if idx > 0 else 0)
        for idx, q_end in enumerate(actual_seq_lengths_q)
    )
    trace_max_q_len = int(os.getenv("VLLM_ASCEND_QWEN3_BLOCK_TRACE_MAX_Q_LEN", "8"))
    if max_q_len > trace_max_q_len:
        return

    head_dim = max(1, int(os.getenv("VLLM_ASCEND_QWEN3_BLOCK_TRACE_HEAD", "128")))
    current_input_ids = getattr(_EXTRA_CTX, "input_ids", None)

    def _vector_head(tensor: torch.Tensor) -> list[float]:
        return tensor[:head_dim].detach().to(torch.float32).cpu().tolist()

    def _vector_norm(tensor: torch.Tensor) -> float:
        return float(tensor.detach().to(torch.float32).norm().item())

    def _serialize_row_slice(value, start: int, end: int):
        if value is None:
            return None
        try:
            sliced = value[start:end]
        except Exception:
            return None
        if isinstance(sliced, torch.Tensor):
            return sliced.detach().cpu().tolist()
        try:
            return torch.as_tensor(sliced).detach().cpu().tolist()
        except Exception:
            try:
                return list(sliced)
            except Exception:
                return None

    requests = []
    q_start = 0
    for req_idx, q_end in enumerate(actual_seq_lengths_q):
        q_len = q_end - q_start
        if q_len <= 0:
            q_start = q_end
            continue
        seq_len = seq_lens[req_idx]
        prefix_len = seq_len - q_len
        req_slice = slice(q_start, q_end)
        requests.append(
            {
                "request_index": req_idx,
                "q_len": q_len,
                "seq_len": seq_len,
                "prefix_len": prefix_len,
                "current_input_ids": _serialize_row_slice(current_input_ids, q_start, q_end),
                "current_input_positions": _serialize_row_slice(positions, q_start, q_end),
                "current_slots": slot_mapping[req_slice].detach().cpu().tolist(),
                "layer_input_hidden_heads": [_vector_head(layer_input_hidden[q_start + row_idx]) for row_idx in range(q_len)],
                "layer_input_hidden_norms": [_vector_norm(layer_input_hidden[q_start + row_idx]) for row_idx in range(q_len)],
                "layer_input_residual_heads": (
                    [_vector_head(layer_input_residual[q_start + row_idx]) for row_idx in range(q_len)]
                    if layer_input_residual is not None
                    else None
                ),
                "layer_input_residual_norms": (
                    [_vector_norm(layer_input_residual[q_start + row_idx]) for row_idx in range(q_len)]
                    if layer_input_residual is not None
                    else None
                ),
                "pre_attn_hidden_heads": [_vector_head(pre_attn_hidden[q_start + row_idx]) for row_idx in range(q_len)],
                "pre_attn_hidden_norms": [_vector_norm(pre_attn_hidden[q_start + row_idx]) for row_idx in range(q_len)],
                "attn_output_heads": [_vector_head(attn_output[q_start + row_idx]) for row_idx in range(q_len)],
                "attn_output_norms": [_vector_norm(attn_output[q_start + row_idx]) for row_idx in range(q_len)],
                "post_attn_hidden_heads": [_vector_head(post_attn_hidden[q_start + row_idx]) for row_idx in range(q_len)],
                "post_attn_hidden_norms": [_vector_norm(post_attn_hidden[q_start + row_idx]) for row_idx in range(q_len)],
                "post_attn_residual_heads": [_vector_head(post_attn_residual[q_start + row_idx]) for row_idx in range(q_len)],
                "post_attn_residual_norms": [_vector_norm(post_attn_residual[q_start + row_idx]) for row_idx in range(q_len)],
                "mlp_output_heads": [_vector_head(mlp_output[q_start + row_idx]) for row_idx in range(q_len)],
                "mlp_output_norms": [_vector_norm(mlp_output[q_start + row_idx]) for row_idx in range(q_len)],
            }
        )
        q_start = q_end

    payload = {
        "index": _QWEN3_BLOCK_TRACE_COUNT,
        "pid": os.getpid(),
        "tp_rank": tp_rank,
        "block_name": block_name,
        "layer_name": layer_name,
        "label": os.getenv("VLLM_ASCEND_QWEN3_BLOCK_TRACE_LABEL"),
        "attn_state": getattr(getattr(attn_metadata, "attn_state", None), "name", None),
        "tree_spec_decode_enabled": getattr(attn_metadata, "tree_spec_decode_enabled", None),
        "num_actual_tokens": getattr(attn_metadata, "num_actual_tokens", None),
        "requests": requests,
    }
    with open(trace_path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _QWEN3_BLOCK_TRACE_COUNT += 1


def tensor_parallel_wrap(func):
    def wrap(*args, **kwargs):
        deepstack_input_embeds = func(*args, **kwargs)
        if deepstack_input_embeds is None:
            return deepstack_input_embeds
        try:
            flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled
        except (AssertionError, AttributeError, KeyError):
            flash_comm_v1_enabled = False
        if flash_comm_v1_enabled:
            tp_size = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()
            deepstack_input_embeds.tensors = {
                k: v.chunk(tp_size)[tp_rank] for k, v in deepstack_input_embeds.tensors.items()
            }
        return deepstack_input_embeds

    return wrap


def forward_with_split_qkv_rmsnorm_mrope(self, positions: torch.Tensor, hidden_states: torch.Tensor):
    _maybe_dump_qwen3_attn_trace(self)
    qkv, _ = self.qkv_proj(hidden_states)
    if isinstance(self.rotary_emb, AscendMRotaryEmbedding):
        cos_sin = self.rotary_emb.cos_sin_cache[positions]
        if cos_sin.device != qkv.device:
            cos_sin = cos_sin.to(qkv.device)
        if cos_sin.dtype != qkv.dtype:
            cos_sin = cos_sin.to(qkv.dtype)
        q, k, v, _ = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
            qkv=qkv,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            cos_sin=cos_sin,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            eps=self.q_norm.variance_epsilon,
            mrope_section=self.rotary_emb.mrope_section,
            is_interleaved=self.rotary_emb.mrope_interleaved,
            rope_dim=self.rotary_emb.rotary_dim,
        )
    else:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


def forward_qwen3_decoder_layer_with_block_trace(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
):
    layer_input_hidden = hidden_states
    layer_input_residual = residual

    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    pre_attn_hidden = hidden_states
    attn_output = self.self_attn(
        positions=positions,
        hidden_states=hidden_states,
    )

    post_attn_hidden, post_attn_residual = self.post_attention_layernorm(attn_output, residual)
    mlp_output = self.mlp(post_attn_hidden)

    _maybe_dump_qwen3_block_trace(
        self,
        positions,
        layer_input_hidden,
        layer_input_residual,
        pre_attn_hidden,
        attn_output,
        post_attn_hidden,
        post_attn_residual,
        mlp_output,
    )
    return mlp_output, post_attn_residual


Qwen3Attention.forward = forward_with_split_qkv_rmsnorm_mrope
Qwen3MoeAttention.forward = forward_with_split_qkv_rmsnorm_mrope
Qwen3DecoderLayer.forward = forward_qwen3_decoder_layer_with_block_trace
Qwen3MoeDecoderLayer.forward = forward_qwen3_decoder_layer_with_block_trace
Qwen3VLForConditionalGeneration._get_deepstack_input_embeds = tensor_parallel_wrap(
    Qwen3VLForConditionalGeneration._get_deepstack_input_embeds
)


def _fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> torch.Tensor:
    outputs = []
    for t, h, w in grid_thw:
        outputs.append(
            pos_embed_interpolate_native(
                self.pos_embed.weight,
                t,
                h,
                w,
                self.num_grid_per_side,
                self.spatial_merge_size,
                self.dtype,
            )
        )
    return torch.cat(outputs, dim=0)


Qwen3_VisionTransformer.fast_pos_embed_interpolate = _fast_pos_embed_interpolate


def patch_qwen3_vl_moe_pp_layer_range():
    try:
        from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMForCausalLM
    except Exception:
        return

    if not hasattr(Qwen3MoeLLMForCausalLM, "start_layer"):
        Qwen3MoeLLMForCausalLM.start_layer = property(lambda self: self.model.start_layer)

    if not hasattr(Qwen3MoeLLMForCausalLM, "end_layer"):
        Qwen3MoeLLMForCausalLM.end_layer = property(lambda self: self.model.end_layer)


patch_qwen3_vl_moe_pp_layer_range()
