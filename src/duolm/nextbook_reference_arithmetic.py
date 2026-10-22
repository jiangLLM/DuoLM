"""Opt-in V3 reference FP16 arithmetic, extracted from the gated R7 experiment.

No profiler, receipt writer, per-forward tensor hash, or global ATen override.
Only the installing model's authenticated N100 score forwards use these methods.
"""
from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
import threading
import types

import torch
from torch.nn import functional as F

MODE = "reference_fp16"
SG_SOURCE_SHA256 = {
    "sglang.srt.models.qwen3": "b53f8a8f1a695b334e2bd6d2c10c55f3515a6b4db7ba4e04ffe22e4bd15d43fd",
    "sglang.srt.models.utils": "a161b4d26f5d8722b33976937514f3f646b85be22bd2504ce8ac2aabfb2575fd",
    "sglang.srt.layers.layernorm": "a0e2fe7d6523dd3a030111dac77642c7d7a52f67bf668b9559d7cd2acb5ae4e5",
    "sglang.srt.layers.activation": "c732ab311f98e78b4f22edc7f26c3d928be40dce100dea24754984ddc6c37bc3",
    "sglang.srt.layers.rotary_embedding.base": "96bb5bf2c934bebe035d718e886d0c94095f1b590eeb52ab6a71e903643d407f",
    "sglang.srt.layers.radix_attention": "2956010f919a8b219bdf6ba0263f4b77d8c169af1d9745f0e3dee779e282fb1a",
    "sglang.srt.layers.quantization.unquant": "9f071f09eba9522e8d2b3e26fa0f12c0a6564da22b8062d5c8a12df9db4202bb",
    "sglang.srt.batch_invariant_ops.batch_invariant_ops": "d13b28a71affd6737e87e98a57340de4ed3d48c0f20a25770bba7c88aea8a5a1",
}


def source_hashes():
    return {Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def require(condition, message):
    if not condition:
        raise ValueError("Reference arithmetic: " + message)


def validate_model_config(config):
    """Equivalent explicit default-RoPE forms; SG may normalize AutoConfig again."""
    expected = {"num_hidden_layers": 28, "head_dim": 128, "hidden_size": 1024,
                "intermediate_size": 3072, "num_attention_heads": 16, "num_key_value_heads": 8,
                "hidden_act": "silu", "rms_norm_eps": 1e-6, "use_sliding_window": False}
    for name, value in expected.items():
        require(hasattr(config, name), "Model config missing field: " + name)
        require(getattr(config, name) == value, "Model config differs: " + name)
    require(hasattr(config, "rope_scaling"), "Model config missing field: rope_scaling")
    theta_present = hasattr(config, "rope_theta")
    theta = getattr(config, "rope_theta", None)
    scaling = config.rope_scaling
    if theta_present:
        require(theta == 1000000, "Model config differs: rope_theta")
    if scaling is None:
        require(theta_present, "Default RoPE has no explicit theta")
        representation = "legacy_default"
    else:
        require(type(scaling) is dict, "Model config rope_scaling must be a dictionary or null")
        require(set(scaling) <= {"rope_theta", "rope_type", "type"}, "Model config rope_scaling has unsupported fields")
        require(scaling.get("rope_type") == "default", "Model config differs: rope_scaling.rope_type")
        if "type" in scaling:
            require(scaling["type"] == "default", "Model config differs: rope_scaling.type alias")
        if "rope_theta" in scaling:
            require(scaling["rope_theta"] == 1000000, "Model config differs: rope_scaling.rope_theta")
        else:
            require(theta_present, "Default RoPE has no explicit theta")
        representation = "normalized_default"
    return {"representation": representation, "rope_theta_present": theta_present,
            "rope_theta": theta, "rope_scaling": scaling}


def inv_freq_cpu():
    with torch.autocast(device_type="cpu", enabled=False):
        value = 1.0 / (1000000.0 ** (torch.arange(0, 128, 2, dtype=torch.int64, device="cpu").float() / 128))
    require(hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
            == "138c99b109d7affbfba059e435670918fe4531bce4709b6e86f3f22f7ef80f6e", "RoPE frequency bytes differ")
    return value


def rope(positions, query, key, inv_freq):
    with torch.autocast(device_type=query.device.type, enabled=False):
        freqs = positions[None, :, None].float() * inv_freq[None, None, :].float()
        phase = torch.cat((freqs, freqs), dim=-1)
        cos, sin = phase.cos().half()[0, :, None, :], phase.sin().half()[0, :, None, :]
        def rotate(value):
            shaped = value.reshape(positions.numel(), -1, 128)
            rotated = torch.cat((-shaped[..., 64:], shaped[..., :64]), dim=-1)
            first = shaped * cos
            second = rotated * sin
            return (first + second).reshape(value.shape)
        return rotate(query), rotate(key)


def mean_lastdim(squared):
    out = torch.empty((*squared.shape[:-1], 1), device=squared.device, dtype=torch.float32)
    return torch.ops.aten.mean.out(squared, [-1], True, out=out)


def rmsnorm(x, weight, eps, residual=None):
    with torch.autocast(device_type=x.device.type, enabled=False):
        summed = x if residual is None else residual + x
        widened = summed.to(torch.float32)
        variance = mean_lastdim(widened.pow(2))
        normalized = widened * torch.rsqrt(variance + eps)
        result = weight * normalized.to(x.dtype)
    return result if residual is None else (result, summed)


def silu_and_mul(value):
    gate, up = value.chunk(2, dim=-1)
    with torch.autocast(device_type=value.device.type, enabled=False):
        activated = F.silu(gate)
        return activated * up


def linear(x, weight):
    out = torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=x.dtype)
    return torch.ops.aten.mm.out(x, weight.t(), out=out)


def independent_inputs(query, key, value, prefix):
    rows = torch.cat((torch.arange(prefix, device=query.device).expand(100, -1),
                      (prefix + 1 + 2 * torch.arange(100, device=query.device))[:, None]), dim=1)
    return tuple(x.index_select(0, rows.flatten()).view(100, prefix + 1, heads, 128).transpose(1, 2)
                 for x, heads in ((query, 16), (key, 8), (value, 8)))


def repeat_kv(value):
    batch, heads, length, width = value.shape
    return value[:, :, None, :, :].expand(batch, heads, 2, length, width).reshape(batch, heads * 2, length, width)


def explicit_mask(prefix, device):
    return torch.ones((100, 1, prefix + 1, prefix + 1), device=device, dtype=torch.bool).tril()


def combine(original_output, independent_output, prefix):
    result = original_output.clone()
    result[:prefix].copy_(independent_output[0, :prefix])
    endpoints = prefix + 1 + 2 * torch.arange(100, device=result.device)
    result.index_copy_(0, endpoints, independent_output[:, -1])
    return result


def attention(query, key, value, prefix, original_output):
    q, k, v = independent_inputs(query, key, value, prefix)
    k, v = repeat_kv(k), repeat_kv(v)
    mask = explicit_mask(prefix, query.device)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with torch.autocast(device_type="cuda", enabled=False), sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0,
                                               is_causal=False, scale=128 ** -0.5)
    independent = output.transpose(1, 2).contiguous().view(100, prefix + 1, 2048)
    return combine(original_output, independent, prefix)


def _source_check():
    import importlib
    for name, expected in SG_SOURCE_SHA256.items():
        module = importlib.import_module(name)
        require(hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest() == expected,
                "Unsupported source: " + name)


def _class(value, module, name):
    require(type(value).__module__ == module and type(value).__name__ == name,
            "Unsupported class: " + name)


class _Runtime:
    def __init__(self, model, config):
        from sglang.srt.environ import envs
        from sglang.srt.batch_invariant_ops import batch_invariant_ops
        from sglang.srt.model_executor.runner import get_is_capture_mode
        require(config.get("arithmetic_mode") == MODE and config.get("variant") == "V3"
                and config.get("m") == 1 and config.get("prefix_cache") is False,
                "Only uncached V3 m=1 reference arithmetic is supported")
        require(config.get("arithmetic_source_sha256") == source_hashes(), "Arithmetic source binding differs")
        require(not hasattr(model, "_mixlm_reference_arithmetic") and "forward" not in model.__dict__,
                "Model arithmetic is already installed or overridden")
        _source_check()
        _class(model, "sglang.srt.models.qwen3", "Qwen3ForCausalLM")
        validate_model_config(model.config)
        require(model.quant_config is None and model.pp_group.world_size == 1
                and model.model.embed_tokens.tp_size == 1, "Unquantized TP/PP=1 required")
        require(envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get() is True,
                "Deterministic inference must expose Q/K norm modules")
        require(not torch.backends.cuda.matmul.allow_tf32, "Matmul TF32 must be disabled")
        self.model, self.signature = model, config["signature"]
        self.owner = (os.getpid(), threading.get_ident())
        self.current = None
        self.registry, self.library = batch_invariant_ops, None
        self.is_capture = get_is_capture_mode
        scalar = model.lm_head._mixlm_nextbook_scalar_weight
        table = model.model.embed_tokens._mixlm_nextbook_table
        require(scalar.dtype == torch.float32 and scalar.shape == (1, 1024) and scalar.device.type == "cuda",
                "Original FP32 CUDA scalar head required")
        require(table.dtype == torch.float16 and table.device == scalar.device
                and all(p.dtype == torch.float16 and p.device == scalar.device for p in model.parameters()),
                "Reader parameters/table must be CUDA FP16")
        self.device = scalar.device
        modules = dict(model.named_modules())
        norms, silus, attentions, ropes, methods = [], [], [], {}, {}
        self.linears = {}
        for i in range(28):
            base = f"model.layers.{i}"
            for suffix in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
                name = base + "." + suffix
                norms.append((name, modules[name]))
            silus.append(modules[base + ".mlp.act_fn"])
            preparation = modules[base + ".self_attn"]
            _class(preparation, "sglang.srt.models.qwen3", "Qwen3Attention")
            require((preparation.tp_size, preparation.num_heads, preparation.num_kv_heads, preparation.head_dim)
                    == (1, 16, 8, 128) and not preparation.use_fused_qk_norm_mrope,
                    "Unsupported attention preparation")
            rotary = preparation.rotary_emb
            ropes[id(rotary)] = rotary
            attn = modules[base + ".self_attn.attn"]
            _class(attn, "sglang.srt.layers.radix_attention", "RadixAttention")
            require(attn.layer_id == i and attn.tp_q_head_num == 16 and attn.tp_k_head_num == attn.tp_v_head_num == 8
                    and attn.head_dim == attn.qk_head_dim == attn.v_head_dim == 128
                    and attn.scaling == 128 ** -0.5 and attn.logit_cap == 0 and attn.sliding_window_size == -1
                    and not attn.is_cross_attention and attn.quant_method is None and "forward" not in attn.__dict__,
                    "Unsupported RadixAttention configuration")
            attentions.append(attn)
            for suffix, shape in (("self_attn.qkv_proj", (4096, 1024)), ("self_attn.o_proj", (1024, 2048)),
                                  ("mlp.gate_up_proj", (6144, 1024)), ("mlp.down_proj", (1024, 3072))):
                layer = modules[base + "." + suffix]
                method = layer.quant_method
                _class(method, "sglang.srt.layers.quantization.unquant", "UnquantizedLinearMethod")
                require(layer.weight.shape == shape and layer.bias is None and "apply" not in method.__dict__,
                        "Unsupported reader projection")
                self.linears[id(layer)] = layer
                methods[id(method)] = method
        norms.append(("model.norm", modules["model.norm"]))
        require(len({id(x) for _, x in norms}) == 113 and len({id(x) for x in silus}) == 28
                and len({id(x) for x in attentions}) == 28 and len(self.linears) == 112,
                "Expected distinct reader modules")
        for name, norm in norms:
            _class(norm, "sglang.srt.layers.layernorm", "RMSNorm")
            require(norm.has_weight and norm.variance_epsilon == 1e-6 and norm.variance_size_override is None
                    and norm.override_orig_dtype is None and not norm.fp32_residual and "forward" not in norm.__dict__
                    and norm.hidden_size == (128 if name.endswith(("q_norm", "k_norm")) else 1024),
                    "Unsupported RMSNorm configuration")
        for act in silus:
            _class(act, "sglang.srt.layers.activation", "SiluAndMul")
            require("forward" not in act.__dict__, "Activation already overridden")
        for rotary in ropes.values():
            _class(rotary, "sglang.srt.layers.rotary_embedding.base", "RotaryEmbedding")
            require(rotary.head_size == rotary.rotary_dim == 128 and rotary.base == 1000000
                    and rotary.is_neox_style is True and "forward" not in rotary.__dict__, "Unsupported RoPE")
        self.inv_freq = inv_freq_cpu().to(self.device)
        # All source/configuration checks precede the first method mutation.
        for rotary in ropes.values():
            self._patch_rope(rotary)
        for name, norm in norms:
            self._patch_norm(name, norm)
        for act in silus:
            self._patch_silu(act)
        for method in methods.values():
            self._patch_linear(method)
        for index, attn in enumerate(attentions):
            self._patch_attention(index, attn)
        self._patch_model()

    def _patch_rope(self, module):
        original = module.forward
        def forward(module, positions, query, key, offsets=None, fused_set_kv_buffer_arg=None):
            if self.current is None:
                return original(positions, query, key, offsets, fused_set_kv_buffer_arg)
            require(positions is self.current["positions"] and offsets is None and fused_set_kv_buffer_arg is None,
                    "Unsupported RoPE position/cache arguments")
            return rope(positions, query, key, self.inv_freq)
        module.forward = types.MethodType(forward, module)

    def _patch_norm(self, name, module):
        original = module.forward
        expected_residual = name == "model.norm" or name.endswith("post_attention_layernorm") or (
            name.endswith("input_layernorm") and name != "model.layers.0.input_layernorm")
        def forward(module, x, residual=None, post_residual_addition=None, quant_linear=None):
            if self.current is None:
                return original(x, residual, post_residual_addition, quant_linear)
            require(post_residual_addition is None and (residual is not None) == expected_residual
                    and (quant_linear is None or getattr(quant_linear, "quant_config", None) is None),
                    "Unsupported residual/quantization arguments")
            return rmsnorm(x, module.weight, module.variance_epsilon, residual)
        module.forward = types.MethodType(forward, module)

    def _patch_silu(self, module):
        original = module.forward
        def forward(module, value):
            return original(value) if self.current is None else silu_and_mul(value)
        module.forward = types.MethodType(forward, module)

    def _patch_linear(self, method):
        original = method.apply
        def apply(method, layer, x, bias=None):
            if self.current is None or id(layer) not in self.linears:
                return original(layer, x, bias)
            require(bias is None and layer is self.linears[id(layer)], "Unsupported reader linear arguments")
            return linear(x, layer.weight)
        method.apply = types.MethodType(apply, method)

    def _patch_attention(self, index, module):
        original = module.forward
        def forward(module, q, k, v, forward_batch, save_kv_cache=True, key_value_num_tokens=None, **kwargs):
            current = self.current
            if current is None:
                return original(q, k, v, forward_batch, save_kv_cache=save_kv_cache,
                                key_value_num_tokens=key_value_num_tokens, **kwargs)
            require(forward_batch is current["batch"] and index == current["attention_calls"]
                    and save_kv_cache is True and key_value_num_tokens is None and not kwargs,
                    "Unsupported attention owner/order/cache arguments")
            original_output = original(q, k, v, forward_batch, save_kv_cache=True)
            old_tf32 = torch.backends.cudnn.allow_tf32
            try:
                torch.backends.cudnn.allow_tf32 = False
                output = attention(q, k, v, current["prefix"], original_output)
            finally:
                torch.backends.cudnn.allow_tf32 = old_tf32
            current["attention_calls"] += 1
            return output
        module.forward = types.MethodType(forward, module)

    def _begin(self, input_ids, positions, batch, input_embeds, get_embedding, pp_proxy_tensors):
        from . import nextbook_score_head as scalar
        marks = getattr(batch, "mixlm_nextbook_scores", None)
        require(self.current is None, "Interleaved model forwards")
        if not marks or not any(mark is not None for mark in marks):
            return False
        require(self.owner == (os.getpid(), threading.get_ident()), "Wrong model worker/thread")
        config = scalar.runtime_config()
        require(config is not None and config["signature"] == self.signature
                and len(marks) == 1 and scalar._valid_mark(marks[0], config), "Unauthenticated score forward")
        require(not self.is_capture() and not torch.backends.cuda.matmul.allow_tf32,
                "CUDA capture/TF32 unsupported")
        require(input_embeds is None and not get_embedding and pp_proxy_tensors is None,
                "Unsupported input/embedding override")
        require(input_ids.ndim == positions.ndim == 1 and input_ids.shape == positions.shape
                and input_ids.device == positions.device == self.device
                and input_ids.dtype in (torch.int32, torch.int64) and positions.dtype in (torch.int32, torch.int64),
                "Unsupported model input geometry")
        require(batch.batch_size == 1 and batch.forward_mode.is_extend()
                and list(batch.extend_prefix_lens_cpu) == [0] and list(batch.extend_seq_lens_cpu) == [len(input_ids)],
                "One complete uncached prefill required")
        delimiters = batch.multi_item_delimiter_indices
        require(len(delimiters) == 1 and delimiters[0].device.type == "cpu", "CPU delimiters required")
        indices = delimiters[0].tolist()
        require(len(indices) == 101 and indices[0] > 0
                and indices == list(range(indices[0], indices[0] + 201, 2))
                and len(input_ids) == indices[0] + 201, "Exactly 100 single-book branches required")
        # CPU request_marks has already authenticated origin IDs and delimiters.
        require(self.registry.is_batch_invariant_mode_enabled() is True, "Deterministic registry disabled")
        if self.library is None:
            self.library = self.registry._batch_invariant_LIB
        require(self.library is not None and self.registry._batch_invariant_LIB is self.library,
                "Deterministic registry changed")
        self.current = {"batch": batch, "positions": positions, "prefix": indices[0], "attention_calls": 0}
        return True

    def _patch_model(self):
        original = self.model.forward
        def forward(model, input_ids, positions, forward_batch, input_embeds=None,
                    get_embedding=False, pp_proxy_tensors=None):
            active = self._begin(input_ids, positions, forward_batch, input_embeds, get_embedding, pp_proxy_tensors)
            try:
                result = original(input_ids, positions, forward_batch, input_embeds=input_embeds,
                                  get_embedding=get_embedding, pp_proxy_tensors=pp_proxy_tensors)
                require(not active or self.current["attention_calls"] == 28, "Incomplete reference attention path")
                return result
            finally:
                if active:
                    self.current = None
        self.model.forward = types.MethodType(forward, self.model)


def install(model, config):
    """Called once after the normal score overlay binds its table and FP32 head."""
    runtime = _Runtime(model, config)
    model._mixlm_reference_arithmetic = runtime
    return {"installed": True, "mode": MODE, "source_sha256": source_hashes(),
            "diagnostic_profiling": False, "diagnostic_tensor_hashing": False,
            "candidate_count": 100, "physical_batch_size": 1, "prefix_cache": False}
