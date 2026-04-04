from __future__ import annotations
import copy
import glob
import io
import lzma
import math
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
except ImportError:
    def flash_attn_3_func(q, k, v, causal=True):
        # Fallback for CPU/MPS or when flash_attn is missing
        # Supports Grouped Query Attention (GQA) by broadcasting KV heads
        q = q * (q.size(-1)**-0.5)
        # q: [B, T, H, D], k: [B, T, Hkv, D], v: [B, T, Hkv, D]
        # Transpose to [B, H, T, D] for matmul
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Broadcast k, v if num_heads != num_kv_heads
        if q.size(1) != k.size(1):
            k = k.repeat_interleave(q.size(1) // k.size(1), dim=1)
            v = v.repeat_interleave(q.size(1) // v.size(1), dim=1)
            
        attr = torch.matmul(q, k.transpose(-2, -1))
        if causal:
            mask = torch.triu(torch.ones(attr.size(-2), attr.size(-1), device=q.device), diagonal=1).bool()
            attr = attr.masked_fill(mask, float('-inf'))
        attn = F.softmax(attr, dim=-1)
        return torch.matmul(attn, v).transpose(1, 2)

# --- Best Practices Environment Toggles ---
USE_COMPILE = int(os.environ.get("USE_COMPILE", "1"))
SKIP_INIT_VAL = int(os.environ.get("SKIP_INIT_VAL", "0"))

# --- Project Golf Ball Hyperparameters ---

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    
    val_batch_size = 524_288
    val_loss_every = 4000
    train_log_every = 500
    
    iterations = 20000
    warmdown_iters = 3500
    warmup_steps = 20
    train_batch_tokens = 786_432
    train_seq_len = 2048
    eval_seq_len = 2048
    max_wallclock_seconds = 600.0  # 10 minutes
    
    vocab_size = 1024
    num_layers = 11
    model_dim = 512
    num_heads = 8
    num_kv_heads = 4
    mlp_mult = 3.0
    
    tie_embeddings = True
    rope_base = 10000.0
    logit_softcap = 30.0
    rope_dims = 16
    
    # Optimizer Hparams
    matrix_lr = 0.025
    muon_momentum = 0.99
    muon_backend_steps = 5
    muon_wd = 0.05
    
    scalar_lr = 0.025
    tied_embed_lr = 0.035
    tied_embed_init_std = 0.005
    
    beta1 = 0.9
    beta2 = 0.95
    adam_eps = 1e-8
    grad_clip_norm = 0.3
    qk_gain_init = 0.5
    
    # Advanced Techniques
    xsa_last_n = 4
    qat_enabled = False
    late_qat_threshold = 0.15
    eval_stride = 64
    
    # TTT Hparams
    ttt_enabled = True
    ttt_min_lr = 0.0001
    ttt_max_grad_norm = 1.0
    ttt_lr = 0.002
    ttt_epochs = 3
    ttt_chunk_tokens = 32768
    ttt_freeze_blocks = 2
    ttt_momentum = 0.9
    ttt_batch_seqs = 32
    ttt_grad_clip = 1.0


# --- Batched Newton-Schulz orthogonalization ---

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    was_2d = G.ndim == 2
    if was_2d: G = G.unsqueeze(0)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed: X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed: X = X.mT
    if was_2d: X = X.squeeze(0)
    return X.to(G.dtype)

# --- Parallel Muon optimizer ---

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay))
        self._built = False

    def _build(self):
        self._distributed = dist.is_available() and dist.is_initialized()
        self._world_size = dist.get_world_size() if self._distributed else 1
        ws = self._world_size
        self._bank_meta = []
        for group in self.param_groups:
            for p in group["params"]:
                B = p.shape[0]
                padded_B = ((B + ws - 1) // ws) * ws
                shard_B = padded_B // ws
                tail = p.shape[1:]
                dev = p.device
                self._bank_meta.append({
                    'p': p, 'B': B,
                    'padded_grad': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard_mom': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'full_update': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    'scale': max(1, p.shape[-2] / p.shape[-1]) ** 0.5,
                })
        self._bank_meta.sort(key=lambda m: -m['p'].numel())
        self._built = True

    def launch_reduce_scatters(self):
        if not self._built: self._build()
        if not self._distributed: return
        self._rs_futures = []
        for m in self._bank_meta:
            p = m['p']
            if p.grad is None:
                self._rs_futures.append(None)
                continue
            pg = m['padded_grad']
            pg[:m['B']].copy_(p.grad.bfloat16())
            if pg.shape[0] > m['B']: pg[m['B']:].zero_()
            fut = dist.reduce_scatter_tensor(m['shard'], pg, op=dist.ReduceOp.AVG, async_op=True)
            self._rs_futures.append(fut)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        if not self._built: self._build()
        for group in self.param_groups:
            lr, momentum, backend_steps, nesterov, wd = group["lr"], group["momentum"], group["backend_steps"], group["nesterov"], group.get("weight_decay", 0.0)
            prev_ag_handle, prev_m = None, None
            sharded = self._distributed and hasattr(self, '_rs_futures')
            for i, m in enumerate(self._bank_meta):
                p = m['p']
                if p.grad is None: continue
                if prev_ag_handle is not None:
                    prev_ag_handle.wait()
                    pp = prev_m['p']
                    upd = prev_m['full_update'][:prev_m['B']]
                    if wd > 0.0: pp.data.mul_(1.0 - lr * wd)
                    pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])
                if sharded and self._rs_futures[i] is not None:
                    self._rs_futures[i].wait()
                    g, buf = m['shard'], m['shard_mom']
                else:
                    g = p.grad.bfloat16()
                    state = self.state[p]
                    if "momentum_buffer" not in state: state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                update = zeropower_via_newtonschulz5(update, steps=backend_steps)
                if sharded:
                    prev_ag_handle = dist.all_gather_into_tensor(m['full_update'], update, async_op=True)
                    prev_m = m
                else:
                    if wd > 0.0: p.data.mul_(1.0 - lr * wd)
                    p.add_(update.to(dtype=p.dtype), alpha=-lr * m['scale'])
            if prev_ag_handle is not None:
                prev_ag_handle.wait()
                pp = prev_m['p']
                upd = prev_m['full_update'][:prev_m['B']]
                if wd > 0.0: pp.data.mul_(1.0 - lr * wd)
                pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])
            if hasattr(self, '_rs_futures'): del self._rs_futures
        return loss

# --- Transformer modules ---

class RMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                scale = (w32.abs().amax(dim=1) / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        self.dim, self.base, self.train_seq_len, self.rope_dims = dim, base, train_seq_len, rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached, self._cos_cached, self._sin_cached = 0, None, None
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            rd = self.rope_dims
            inv_freq = self.inv_freq.to(device)
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached, self._sin_cached = freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        return torch.cat((torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1), x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float = 0.5):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, dim // num_heads
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims, self.use_xsa = 0, False
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=2048)

    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        y_g = y.reshape(B, T, Hkv, H // Hkv, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x: Tensor, q_w: Tensor, k_w: Tensor, v_w: Tensor, out_w: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = F.linear(x, q_w.to(x.dtype)).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = F.linear(x, k_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = F.linear(x, v_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q, k = apply_rotary_emb(q, cos, sin, self.rope_dims), apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa: y = self._xsa_efficient(y, v)
        return F.linear(y.reshape(bsz, seqlen, dim), out_w.to(x.dtype))

class MLP(nn.Module):
    def forward(self, x: Tensor, up_w: Tensor, down_w: Tensor) -> Tensor:
        # Squared LeakyReLU(0.5) activation
        x = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.5)
        return F.linear(x.square(), down_w.to(x.dtype))

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: float, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm, self.mlp_norm = RMSNorm(), RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP()
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))))

    def forward(self, x: Tensor, x0: Tensor, q_w: Tensor, k_w: Tensor, v_w: Tensor, out_w: Tensor, up_w: Tensor, down_w: Tensor) -> Tensor:
        mix = self.resid_mix.to(x.dtype)
        x_in = mix[0] * x + mix[1] * x0
        x = x_in + self.attn_scale.to(x.dtype) * self.attn(self.attn_norm(x_in), q_w, k_w, v_w, out_w)
        return x + self.mlp_scale.to(x.dtype) * self.mlp(self.mlp_norm(x), up_w, down_w)

class GPT(nn.Module):
    def __init__(self, h: Hyperparameters):
        super().__init__()
        self.h = h
        self.tok_emb = nn.Embedding(h.vocab_size, h.model_dim)
        self.num_encoder = h.num_layers // 2
        self.num_decoder = h.num_layers - self.num_encoder
        self.skip_weights = nn.Parameter(torch.ones(min(self.num_encoder, self.num_decoder), h.model_dim))
        
        head_dim = h.model_dim // h.num_heads
        kv_dim, mlp_dim = h.num_kv_heads * head_dim, int(h.mlp_mult * h.model_dim)
        self.num_layers = h.num_layers
        self.qo_bank = nn.Parameter(torch.empty(2 * h.num_layers, h.model_dim, h.model_dim))
        self.kv_bank = nn.Parameter(torch.empty(2 * h.num_layers, kv_dim, h.model_dim))
        self.mlp_up_bank = nn.Parameter(torch.empty(h.num_layers, mlp_dim, h.model_dim))
        self.mlp_down_bank = nn.Parameter(torch.empty(h.num_layers, h.model_dim, mlp_dim))
        
        self.blocks = nn.ModuleList([Block(h.model_dim, h.num_heads, h.num_kv_heads, h.mlp_mult, h.rope_base, getattr(h, 'qk_gain_init', 0.5)) for _ in range(h.num_layers)])
        for i, block in enumerate(self.blocks):
            block.attn.rope_dims = h.rope_dims
            block.attn.rotary = Rotary(head_dim, base=h.rope_base, train_seq_len=2048, rope_dims=h.rope_dims)
            if i >= h.num_layers - h.xsa_last_n: block.attn.use_xsa = True
            
        self.final_norm = RMSNorm()
        self.logit_softcap = h.logit_softcap
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=self.h.tied_embed_init_std)
        proj_scale = 1.0 / math.sqrt(2 * self.num_layers)
        for i in range(self.num_layers):
            nn.init.orthogonal_(self.qo_bank.data[i], gain=1.0)
            nn.init.zeros_(self.qo_bank.data[self.num_layers + i])
            nn.init.orthogonal_(self.kv_bank.data[i], gain=1.0)
            nn.init.orthogonal_(self.kv_bank.data[self.num_layers + i], gain=1.0)
            nn.init.orthogonal_(self.mlp_up_bank.data[i], gain=1.0)
            nn.init.zeros_(self.mlp_down_bank.data[i])
            self.qo_bank.data[self.num_layers + i].mul_(proj_scale)
            self.mlp_down_bank.data[i].mul_(proj_scale)

    def forward(self, input_ids: Tensor, target_ids: Tensor | None = None) -> Tensor:
        n = self.num_layers
        x = F.rms_norm(self.tok_emb(input_ids), (self.h.model_dim,))
        x0, skips = x, []
        for i in range(self.num_encoder):
            x = self.blocks[i](x, x0, self.qo_bank[i], self.kv_bank[i], self.kv_bank[n+i], self.qo_bank[n+i], self.mlp_up_bank[i], self.mlp_down_bank[i])
            skips.append(x)
        for i in range(self.num_decoder):
            bi = self.num_encoder + i
            if skips: x = x + self.skip_weights[i] * skips.pop()
            x = self.blocks[bi](x, x0, self.qo_bank[bi], self.kv_bank[bi], self.kv_bank[n+bi], self.qo_bank[n+bi], self.mlp_up_bank[bi], self.mlp_down_bank[bi])
        x = self.final_norm(x)
        logits = self.logit_softcap * torch.tanh(F.linear(x, self.tok_emb.weight) / self.logit_softcap)
        if target_ids is None: return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)).float(), target_ids.view(-1), reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        return self.forward(input_ids)

# --- Project Golf Ball Compression Engine ---

def quantize_per_row_mse_search(t: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    clip_range = (2**(bits-1)) - 1
    best_q, best_s, best_err = None, None, float('inf')
    for pct in [0.999, 0.9995, 0.9999, 0.99999, 1.0]:
        row_clip = torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0 else t32.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1e-6).to(torch.float16)
        q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
        err = (t32 - q.float() * s.float()[:, None]).pow(2).mean().item()
        if err < best_err: best_q, best_s, best_err = q, s, err
    return best_q, best_s

def project_golf_quantize(model: GPT) -> bytes:
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    result, meta = {}, {}
    for name, t in sd.items():
        if "tok_emb" in name:
            result[name] = t.to(torch.float16)
            meta[name] = "fp16"
        elif t.numel() <= 65536 or t.ndim < 2:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
        else:
            bits = 5 if "mlp_down" in name else 6
            q, s = quantize_per_row_mse_search(t.view(t.shape[0], -1), bits)
            result[name + ".q"], result[name + ".s"] = q, s
            meta[name] = {"bits": bits, "shape": t.shape}
    buf = io.BytesIO()
    torch.save({"w": result, "m": meta}, buf)
    return lzma.compress(buf.getvalue(), preset=6)

# --- Data Loading ---

def load_data_shard(file: Path) -> Tensor:
    if file.stat().st_size < 1024:
        raise ValueError(f"File {file} is too small or corrupt (HuggingFace 0-byte download?)")
    header = np.fromfile(file, dtype="<i4", count=256)
    num_tokens = int(header[2])
    return torch.from_numpy(np.fromfile(file, dtype="<u2", count=num_tokens, offset=1024).astype(np.uint16))

def load_all_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: return torch.zeros(1, dtype=torch.uint16)
    tokens = torch.cat([load_data_shard(f) for f in files])
    usable = (tokens.numel() // seq_len) * seq_len
    return tokens[:usable+1]

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        self.idx, self.pos = 0, 0
        self.tokens = load_data_shard(self.files[0]) if self.files else torch.zeros(1, dtype=torch.uint16)
    def take(self, n: int) -> Tensor:
        chunks = []
        while n > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self.idx = (self.idx + 1) % len(self.files)
                self.tokens, self.pos = load_data_shard(self.files[self.idx]), 0
                continue
            k = min(n, avail)
            chunks.append(self.tokens[self.pos:self.pos+k])
            self.pos, n = self.pos + k, n - k
        return torch.cat(chunks) if len(chunks) > 1 else chunks[0]

class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, ws: int, device: torch.device):
        self.rank, self.ws, self.device, self.stream = rank, ws, device, TokenStream(pattern)
    def next_batch(self, tokens: int, seq_len: int, accum: int) -> tuple[Tensor, Tensor]:
        n = tokens // (self.ws * accum) + 1
        chunk = self.stream.take(n * self.ws).to(self.device, dtype=torch.int64)
        local = chunk[self.rank * n : self.rank * n + n]
        return local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)

def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device):
    bb, ls, bnd = np.zeros(vocab_size, dtype=np.int16), np.zeros(vocab_size, dtype=np.bool_), np.ones(vocab_size, dtype=np.bool_)
    for i in range(sp.vocab_size()):
        if sp.is_control(i) or sp.is_unknown(i): continue
        bnd[i] = False
        if sp.is_byte(i): bb[i] = 1
        else:
            p = sp.id_to_piece(i)
            if p.startswith("\u2581"): ls[i], p = True, p[1:]
            bb[i] = len(p.encode("utf-8"))
    return torch.tensor(bb, device=device), torch.tensor(ls, device=device), torch.tensor(bnd, device=device)

# --- Evaluation & TTT ---

def eval_val(args: Hyperparameters, model: nn.Module, rank: int, ws: int, device: torch.device, val_tokens: Tensor, luts: tuple):
    bb, ls, bnd = luts
    model.eval()
    loss_sum, tok_count, byte_count = 0.0, 0.0, 0.0
    seq_len = args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    my_s, my_e = (total_seqs * rank) // ws, (total_seqs * (rank + 1)) // ws
    with torch.no_grad():
        for i in range(my_s, my_e):
            raw = val_tokens[i*seq_len:i*seq_len+seq_len+1].to(device, dtype=torch.int64)
            x, y = raw[:-1].view(1, -1), raw[1:].view(1, -1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss_sum += model(x, y).item() * seq_len
            tok_count += seq_len
            tb = bb[y[0]].float() + (ls[y[0]] & ~bnd[x[0]]).float()
            byte_count += tb.sum().item()
    
    sums = torch.tensor([loss_sum, tok_count, byte_count], device=device)
    if dist.is_available() and dist.is_initialized(): dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    val_loss = sums[0] / sums[1]
    val_bpb = (val_loss / math.log(2.0)) * (sums[1] / sums[2])
    return val_loss.item(), val_bpb.item()

def eval_val_sliding_ttt(args: Hyperparameters, model: nn.Module, rank: int, world_size: int, device: torch.device, val_tokens: Tensor, luts: tuple):
    bb, ls, bnd = luts
    stride, ttt_chunk, seq_len = args.eval_stride, args.ttt_chunk_tokens, args.train_seq_len
    total = val_tokens.numel() - 1
    ttt_params = [p for name, p in model.named_parameters() if p.requires_grad and all(f"blocks.{bi}." not in name for bi in range(args.ttt_freeze_blocks))]
    optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
    loss_sum, tok_count, byte_count = 0.0, 0.0, 0.0
    
    num_chunks = (total + ttt_chunk - 1) // ttt_chunk
    for ci in range(num_chunks):
        c_start, c_end = ci * ttt_chunk, min((ci + 1) * ttt_chunk, total)
        # SCORE (Inference)
        model.eval()
        with torch.no_grad():
            # Sliding window scoring across the chunk
            for ws_coord in range(c_start, c_end, stride):
                end_coord = min(ws_coord + seq_len, total)
                if end_coord - ws_coord < 1: continue
                raw = val_tokens[ws_coord:end_coord+1].to(device, dtype=torch.int64)
                x, y = raw[:-1].view(1, -1), raw[1:].view(1, -1)
                s = 0 if ws_coord == 0 else max(len(raw)-1 - stride, 0)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model.forward_logits(x)
                    nll = F.cross_entropy(logits[0].float(), y[0], reduction="none")
                loss_sum += nll[s:].sum().item()
                tok_count += len(nll[s:])
                tb = bb[y[0, s:]].float() + (ls[y[0, s:]] & ~bnd[x[0, s:]]).float()
                byte_count += tb.sum().item()
        
        # TRAIN (TTT)
        if ci < num_chunks - 1:
            model.train()
            chunk_tokens = val_tokens[c_start:c_end+1].to(device, dtype=torch.int64)
            for _ in range(args.ttt_epochs):
                for i in range(0, len(chunk_tokens)-1, seq_len):
                    x_t, y_t = chunk_tokens[i:i+seq_len].view(1, -1), chunk_tokens[i+1:i+seq_len+1].view(1, -1)
                    if x_t.size(1) < 1: continue
                    optimizer.zero_grad()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = model(x_t, y_t)
                    loss.backward()
                    if dist.is_available() and dist.is_initialized() and world_size > 1:
                        for p in ttt_params: dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                    torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                    optimizer.step()
                    
    sums = torch.tensor([loss_sum, tok_count, byte_count], device=device)
    if dist.is_available() and dist.is_initialized(): dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    val_loss = sums[0] / sums[1]
    val_bpb = (val_loss / math.log(2.0)) * (sums[1] / sums[2])
    return val_loss.item(), val_bpb.item()

# --- Main Training Loop ---

def main():
    args = Hyperparameters()
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        rank, ws, local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, ws, local_rank = 0, 1, 0
    device = torch.device("cuda", local_rank)
    
    # --- SEEDING (after dist init) ---
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if rank == 0: print(f"Starting Project Golf Ball | Run ID: {args.run_id}")
    
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_all_tokens(args.val_files, args.train_seq_len)
    luts = build_sentencepiece_luts(sp, args.vocab_size, device)
    
    model = GPT(args).to(device).bfloat16()
    
    # Runpod Tip: Compile for competition speed (free wallclock time)
    if USE_COMPILE:
        if rank == 0: print("Compiling model (takes 3-4 mins)...")
        model = torch.compile(model)
        
    model.qo_bank.data = model.qo_bank.data.float()
    model.kv_bank.data = model.kv_bank.data.float()
    model.mlp_up_bank.data = model.mlp_up_bank.data.float()
    model.mlp_down_bank.data = model.mlp_down_bank.data.float()
    
    optimizer_muon = Muon([model.qo_bank, model.kv_bank, model.mlp_up_bank, model.mlp_down_bank], lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd)
    optimizer_adam = torch.optim.AdamW([p for name, p in model.named_parameters() if p.ndim < 2 or "tok_emb" in name], lr=args.scalar_lr, betas=(args.beta1, args.beta2), fused=True)
    
    train_loader = DistributedTokenLoader(args.train_files, rank, ws, device)
    t_start = time.perf_counter()
    
    if not SKIP_INIT_VAL:
        if rank == 0: print("Running initial validation...")
        val_loss, val_bpb = eval_val(args, model, rank, ws, device, val_tokens, luts)
        if rank == 0: print(f"Initial Val Loss: {val_loss:.4f} | Initial BPB: {val_bpb:.4f}")

    for step in range(args.iterations):
        elapsed = time.perf_counter() - t_start
        if elapsed > args.max_wallclock_seconds: break
        
        scale = 1.0 - (step / args.iterations)
        for pg in optimizer_muon.param_groups: pg['lr'] = args.matrix_lr * scale
        for pg in optimizer_adam.param_groups: pg['lr'] = (args.tied_embed_lr if any("tok_emb" in n for n, p in model.named_parameters() if p is pg['params'][0]) else args.scalar_lr) * scale
        
        optimizer_muon.zero_grad(); optimizer_adam.zero_grad()
        accum = 8 // ws if ws <= 8 else 1
        for _ in range(accum):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, accum)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            (loss / accum).backward()
            
        optimizer_muon.launch_reduce_scatters()
        for p in [p for name, p in model.named_parameters() if p.ndim < 2 or "tok_emb" in name]:
            if p.grad is not None and distributed: dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        optimizer_adam.step()
        optimizer_muon.step()
        
        if rank == 0 and step % args.train_log_every == 0:
            print(f"Step {step} | Loss {loss.item():.4f} | Time {elapsed:.1f}s | Progress {step/args.iterations:.1%}")

    # --- Post-training: Quantize and Evaluate ---
    if rank == 0:
        print("Training complete. Quantizing...")
        # If compiled, access the original model
        m_to_quant = model._orig_mod if hasattr(model, "_orig_mod") else model
        blob = project_golf_quantize(m_to_quant)
        with open("final_model.ptz", "wb") as f: f.write(blob)
        print(f"Final Artifact Size: {len(blob)} bytes")
        
        if args.ttt_enabled:
            print("Running Test-Time Training (TTT) Evaluator...")
            _, ttt_bpb = eval_val_sliding_ttt(args, m_to_quant, rank, ws, device, val_tokens, luts)
            print(f"Final TTT BPB: {ttt_bpb:.6f}")

if __name__ == "__main__": main()
