"""Masked Perceiver readout and an explicitly calibrated A-to-B RMS bridge."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn


class _ResamplerLayer(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.encoder_norm = nn.LayerNorm(width)
        self.latent_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, latents, hidden, encoder_mask, latent_mask):
        queries = self.latent_norm(latents)
        # Valid learned latents can communicate as well as read every valid A token.
        keys = torch.cat((self.encoder_norm(hidden), queries), dim=1)
        valid = torch.cat((encoder_mask, latent_mask), dim=1)
        update, _ = self.attention(queries, keys, keys, key_padding_mask=~valid,
                                   need_weights=False)
        latents = latents + update
        latents = latents + self.ff(self.ff_norm(latents))
        return latents.masked_fill(~latent_mask.unsqueeze(-1), 0.0)


class PerceiverResampler(nn.Module):
    """A shared query bank; each item uses precisely its allocated prefix length."""
    def __init__(self, width: int, max_latents: int, layers: int = 2, heads: int = 8):
        super().__init__()
        for name, value in (("width", width), ("max_latents", max_latents), ("layers", layers), ("heads", heads)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if width % heads:
            raise ValueError("resampler width must be divisible by heads")
        self.width, self.max_latents, self.num_layers, self.heads = width, max_latents, layers, heads
        self.queries = nn.Parameter(torch.empty(max_latents, width))
        nn.init.normal_(self.queries, std=0.02)
        self.layers = nn.ModuleList([_ResamplerLayer(width, heads) for _ in range(layers)])

    def forward(self, hidden, attention_mask, counts):
        if hidden.ndim != 3 or hidden.shape[-1] != self.width:
            raise ValueError("encoder hidden must have shape [items, tokens, width]")
        if tuple(attention_mask.shape) != tuple(hidden.shape[:2]):
            raise ValueError("encoder mask shape does not match hidden states")
        if len(counts) != len(hidden) or not counts:
            raise ValueError("one nonempty latent count is required per item")
        if any(isinstance(n, bool) or not isinstance(n, int) or not 0 < n <= self.max_latents for n in counts):
            raise ValueError("latent counts must be positive integers within the query bank")
        mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
        if not mask.any(-1).all():
            raise ValueError("every encoder item must contain a valid token")
        max_count = max(counts)
        latent_mask = torch.arange(max_count, device=hidden.device)[None, :] < torch.tensor(counts, device=hidden.device)[:, None]
        latents = self.queries[:max_count].unsqueeze(0).expand(len(hidden), -1, -1)
        for layer in self.layers:
            latents = layer(latents, hidden, mask, latent_mask)
        return latents


class CalibratedRMSBridge(nn.Module):
    """Unit RMS states scaled by a fixed native-embedding reference and learned gain."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.log_gain = nn.Parameter(torch.zeros(()))
        # The placeholder is never used: forward rejects uncalibrated bridges.
        self.register_buffer("rms_reference", torch.tensor(1.0))
        self.register_buffer("calibrated", torch.tensor(False))
        self._calibration = None

    def set_reference(self, reference: float, metadata: dict):
        if bool(self.calibrated):
            raise ValueError("bridge calibration is fixed; create a fresh model to recalibrate")
        if not math.isfinite(reference) or reference <= 0:
            raise ValueError("bridge RMS reference must be finite and positive")
        with torch.no_grad():
            self.rms_reference.fill_(reference)
            self.calibrated.fill_(True)
        self._calibration = copy.deepcopy(metadata)

    def forward(self, states):
        if not bool(self.calibrated):
            raise RuntimeError("RMS bridge must be calibrated on training items before use")
        # Reductions and scalar scaling stay FP32 under BF16 training autocast.
        values = states.float()
        normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        return normalized * self.rms_reference.float() * self.log_gain.float().exp()

    def get_extra_state(self):
        return {"calibration": copy.deepcopy(self._calibration)}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or set(state) != {"calibration"}:
            raise ValueError("invalid bridge calibration state")
        self._calibration = copy.deepcopy(state["calibration"])

    def metadata(self):
        return {"type": "calibrated_rms", "eps": self.eps, "gain": "exp(log_gain)",
                "initial_log_gain": 0.0, "calibrated": bool(self.calibrated),
                "rms_reference": float(self.rms_reference), "calibration": copy.deepcopy(self._calibration)}
