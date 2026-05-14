"""
Mimi encoder/decoder + latent-space processing chain.

Stages:
  mimi_encode  ->  SilenceGate  ->  HeliumTemporalTransformer
  ->  bipartite_soft_merge  ->  DepthTransformer  ->  unmerge
  ->  scale correction  ->  mimi_decode
"""

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DEVICE, MIMI_SR
from latency import LATENCY_HUB


# ─────────────────────────────────────────────────────────────────────
# Mimi encode / decode
# ─────────────────────────────────────────────────────────────────────
def _mimi_continuous(mimi, inp: torch.Tensor) -> torch.Tensor:
    """Run encoder sub-modules to get pre-quantisation latents [B, T, D]."""
    emb = mimi.encoder(inp)
    emb = mimi.encoder_transformer(emb.transpose(1, 2))[0]
    emb = mimi.downsample(emb.transpose(1, 2))
    return emb.transpose(1, 2)


def mimi_encode(models: dict, audio_np: np.ndarray) -> torch.Tensor:
    """24 kHz float32 waveform -> continuous latents [1, T, D]."""
    inp = models["mimi_fe"](
        raw_audio=audio_np, sampling_rate=MIMI_SR, return_tensors="pt"
    ).input_values.to(DEVICE)
    if inp.dim() == 2:
        inp = inp.unsqueeze(1)
    with torch.no_grad():
        return _mimi_continuous(models["mimi"], inp)


def _to_mono_1d(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().float().numpy()
    x = np.squeeze(np.asarray(x, dtype=np.float32))
    if x.ndim == 0:
        return np.array([float(x)], dtype=np.float32)
    if x.ndim == 1:
        return x
    x = x.mean(axis=int(np.argmin(x.shape)))
    return np.squeeze(x).reshape(-1).astype(np.float32)


def mimi_decode(models: dict, emb: torch.Tensor) -> np.ndarray:
    """Continuous latents -> 24 kHz float32 waveform."""
    mimi = models["mimi"]
    with torch.no_grad():
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)
        emb_cf = emb.transpose(1, 2).contiguous()
        codes  = mimi.quantizer.encode(emb_cf)
        if codes.shape[0] != emb.shape[0]:
            codes = codes.transpose(0, 1).contiguous()
        out = mimi.decode(audio_codes=codes)
        if hasattr(out, "audio_values"):
            out = out.audio_values
        elif isinstance(out, (tuple, list)):
            out = out[0]
    return _to_mono_1d(out)


# ─────────────────────────────────────────────────────────────────────
# [3] Silence Gate
# ─────────────────────────────────────────────────────────────────────
class SilenceGate(nn.Module):
    def __init__(self, scale: float = 0.5):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor):
        energy    = x.norm(dim=-1)
        threshold = self.scale * energy.median(dim=-1).values.unsqueeze(-1)
        return x, energy < threshold, energy


# ─────────────────────────────────────────────────────────────────────
# [5] Helium Temporal Transformer (TTT)
# ─────────────────────────────────────────────────────────────────────
class HeliumTemporalTransformer(nn.Module):
    def __init__(self, d: int, heads: int = 8, layers: int = 2,
                 ff: int = 4, drop: float = 0.1):
        super().__init__()
        self.H  = heads
        self.hd = d // heads
        self.dp = nn.Dropout(drop)
        self.ls = nn.ModuleList([
            nn.ModuleDict({
                "n1":  nn.LayerNorm(d),
                "qkv": nn.Linear(d, 3 * d,     bias=False),
                "out": nn.Linear(d, d,           bias=False),
                "n2":  nn.LayerNorm(d),
                "gu":  nn.Linear(d, 2 * d * ff,  bias=False),
                "dn":  nn.Linear(d * ff, d,      bias=False),
            })
            for _ in range(layers)
        ])
        self.out_scale = nn.Parameter(torch.ones(d) * 0.1)
        self.gate      = nn.Parameter(torch.full((1,), -4.0))

    def _rope(self, T: int, dev):
        th = 10000.0 ** (-2 * torch.arange(self.hd // 2, device=dev).float() / self.hd)
        t  = torch.arange(T, device=dev).float()
        f  = torch.cat([torch.outer(t, th)] * 2, dim=-1)
        return f.cos()[None, :, None, :], f.sin()[None, :, None, :]

    @staticmethod
    def _rot(x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D  = x.shape
        cos, sin = self._rope(T, x.device)
        residual = x
        for l in self.ls:
            qkv     = l["qkv"](l["n1"](x)).reshape(B, T, 3, self.H, self.hd)
            q, k, v = qkv.unbind(2)
            q = q * cos + self._rot(q) * sin
            k = k * cos + self._rot(k) * sin
            a = F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
            x = x + self.dp(l["out"](a.transpose(1, 2).reshape(B, T, D)))
            g, u = l["gu"](l["n2"](x)).chunk(2, dim=-1)
            x    = x + self.dp(l["dn"](F.silu(g) * u))
        alpha = torch.sigmoid(self.gate)
        return (1.0 - alpha) * residual + alpha * (x * self.out_scale)


# ─────────────────────────────────────────────────────────────────────
# [6] Bipartite Soft Merge (BSM)
# ─────────────────────────────────────────────────────────────────────
@dataclass
class MergeResult:
    merged_x:        torch.Tensor
    merge_map:       torch.Tensor
    original_length: int
    merged_length:   int


def bipartite_soft_merge(
    x: torch.Tensor,
    r: int,
    min_tokens: int = 10,
    local_window: int = 8,
    protected_tokens: int = 2,
    importance_alpha: float = 0.3,
) -> MergeResult:
    B, T, D = x.shape
    Ts = (T + 1) // 2
    Td = T // 2
    r  = min(r, Ts, max(T - min_tokens, 0))
    if r <= 0:
        mm = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        return MergeResult(x, mm.clone(), T, T)

    src, dst = x[:, ::2, :], x[:, 1::2, :]
    with torch.no_grad():
        sn, dn = F.normalize(src, dim=-1), F.normalize(dst, dim=-1)
        if 0 < local_window < Td:
            k  = local_window
            st = (torch.arange(Ts, device=x.device) - k // 2).clamp(0, Td - k)
            gi = (st.unsqueeze(1) + torch.arange(k, device=x.device)).unsqueeze(0).expand(B, -1, -1)
            dl = dn.gather(1, gi.reshape(B, Ts * k).unsqueeze(-1).expand(B, Ts * k, D)).reshape(B, Ts, k, D)
            sc = (sn.unsqueeze(2) * dl).sum(-1)
            nm, bl = sc.max(-1)
            ni = gi.gather(-1, bl.unsqueeze(-1)).squeeze(-1)
        else:
            sc = torch.bmm(sn, dn.transpose(1, 2))
            nm, ni = sc.max(dim=-1)
        if importance_alpha > 0:
            imp = src.norm(dim=-1)
            nm -= importance_alpha * (imp - imp.amin(-1, keepdim=True)) / (
                  imp.amax(-1, keepdim=True) - imp.amin(-1, keepdim=True) + 1e-6)
        if protected_tokens > 0:
            nm[:, :min(protected_tokens, Ts)] = -1.0
        ei = nm.argsort(-1, descending=True)
        ui, si = ei[:, r:], ei[:, :r]
        di = ni.gather(-1, si)

    unm = src.gather(1, ui.unsqueeze(-1).expand(B, Ts - r, D))
    stm = src.gather(1, si.unsqueeze(-1).expand(B, r, D))
    dst = dst.clone()
    dst.scatter_reduce_(1, di.unsqueeze(-1).expand(B, r, D), stm, reduce="mean", include_self=True)
    mx = torch.cat([unm, dst], dim=1)
    db = Ts - r
    mm = torch.zeros(B, T, dtype=torch.long, device=x.device)
    if Td > 0:
        dp = torch.arange(Td, device=x.device)
        mm[:, dp * 2 + 1] = (db + dp).unsqueeze(0)
    if Ts - r > 0:
        mm.scatter_(1, ui * 2, torch.arange(Ts - r, device=x.device).unsqueeze(0).expand(B, -1))
    mm.scatter_(1, si * 2, db + di)
    return MergeResult(mx, mm, T, mx.shape[1])


def unmerge(merged_x: torch.Tensor, mr: MergeResult) -> torch.Tensor:
    B, _, D = merged_x.shape
    return torch.gather(merged_x, 1,
                        mr.merge_map.unsqueeze(-1).expand(B, mr.original_length, D))


# ─────────────────────────────────────────────────────────────────────
# [7] Depth Transformer
# ─────────────────────────────────────────────────────────────────────
class DepthTransformer(nn.Module):
    def __init__(self, d: int, heads: int = 8, layers: int = 3,
                 ff: int = 4, drop: float = 0.1, max_len: int = 512):
        super().__init__()
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.in_norm   = nn.LayerNorm(d)
        enc_layer      = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=d * ff,
            dropout=drop, activation="gelu", batch_first=True, norm_first=True)
        self.enc       = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm      = nn.LayerNorm(d)
        self.out_scale = nn.Parameter(torch.ones(d) * 0.1)
        self.gate      = nn.Parameter(torch.full((1,), -4.0))
        for layer in self.enc.layers:
            for name, p in layer.named_parameters():
                if "weight" in name and p.dim() == 2:
                    nn.init.xavier_uniform_(p, gain=0.5)
                elif "bias" in name:
                    nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T        = x.size(1)
        x_normed = self.in_norm(x) + self.pe[:, :T, :]
        x_enc    = self.norm(self.enc(x_normed)) * self.out_scale
        alpha    = torch.sigmoid(self.gate)
        return (1.0 - alpha) * x + alpha * x_enc


# ─────────────────────────────────────────────────────────────────────
# Full Mimi pipeline (encode -> all stages -> decode)
# ─────────────────────────────────────────────────────────────────────
class MimiPipeline:
    """
    Wraps the complete Mimi latent-space processing chain.
    Takes 24 kHz audio in, returns 24 kHz audio out.
    """

    def __init__(self, models: dict, r_merge: int = 15):
        d = self._probe_d(models)
        self.silence_gate = SilenceGate(scale=0.5).to(DEVICE)
        self.helium       = HeliumTemporalTransformer(d, layers=2).to(DEVICE)
        self.depth_tf     = DepthTransformer(d, layers=3).to(DEVICE)
        self.r_merge      = r_merge
        self.models       = models
        print(f"[MimiPipeline] d_model={d}")

    @staticmethod
    def _probe_d(models: dict) -> int:
        with torch.no_grad():
            p = torch.zeros(1, 1, MIMI_SR, device=DEVICE)
            return _mimi_continuous(models["mimi"], p).shape[-1]

    def run(self, audio_np: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        embeddings = mimi_encode(self.models, audio_np)
        t_encode = (time.perf_counter() - t0) * 1000
        with torch.no_grad():
            x = embeddings.clone()
            t1 = time.perf_counter()
            x, _, _ = self.silence_gate(x)
            t_gate = (time.perf_counter() - t1) * 1000

            t2 = time.perf_counter()
            x  = self.helium(x)
            t_helium = (time.perf_counter() - t2) * 1000

            t3 = time.perf_counter()
            mr = bipartite_soft_merge(
                x, r=self.r_merge, min_tokens=10,
                local_window=8, protected_tokens=2, importance_alpha=0.3)
            t_merge = (time.perf_counter() - t3) * 1000

            t4 = time.perf_counter()
            x  = self.depth_tf(mr.merged_x)
            t_depth = (time.perf_counter() - t4) * 1000

            t5 = time.perf_counter()
            x  = unmerge(x, mr)
            native = embeddings[0].float().norm(dim=-1).mean()
            curr   = x[0].float().norm(dim=-1).mean()
            x      = x * (native / (curr + 1e-8))
            t_unmerge = (time.perf_counter() - t5) * 1000

        t6 = time.perf_counter()
        wav = mimi_decode(self.models, x)
        t_decode = (time.perf_counter() - t6) * 1000

        LATENCY_HUB.publish(
            {
                "[2] Mimi Encoder": t_encode,
                "[3] Silence Gate": t_gate,
                "[4] Squeezeformer (passthrough)": 0.0,
                "[5] Helium TTT": t_helium,
                "[6] BSM Merge": t_merge,
                "[7] Depth Transformer": t_depth,
                "[8] Unmerge + Rescale": t_unmerge,
                "[9] Mimi Decoder": t_decode,
            }
        )
        return wav
