import torch
import torch.nn as nn
from torch.nn import functional as F
from config import *


# =========================================================
# RoPE — cached, applied to both Q and K
# =========================================================

_rope_cache: dict = {}

def get_rope_embeds(T, head_dim, device):
    key = (T, head_dim, str(device))
    if key not in _rope_cache:
        half_dim = head_dim // 2
        theta = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / half_dim))
        pos   = torch.arange(T, device=device).float()
        freqs = pos.unsqueeze(1) * theta.unsqueeze(0)          # (T, half_dim)
        freqs = torch.cat([freqs, freqs], dim=-1)              # (T, head_dim)
        # .detach() ensures these never accumulate grad — they are constants
        _rope_cache[key] = (
            freqs.cos()[None, None, :, :].detach(),
            freqs.sin()[None, None, :, :].detach(),
        )
    return _rope_cache[key]

def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

def apply_rope(x):
    """x: (B, n_heads, T, head_dim) — applies full RoPE across entire head_dim."""
    cos, sin = get_rope_embeds(x.shape[2], x.shape[3], x.device)
    return x * cos + rotate_half(x) * sin


# =========================================================
# FeedForward — SwiGLU, bias=False, hidden = 2/3 * 4 * dim
# =========================================================

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return F.silu(self.w1(x)) * self.w2(x)


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Standard LLaMA-style hidden dim: 2/3 * 4 * dim
        hidden_dim = int(dim * FEEDFORWARD_MULTIPLIER * 2 / 3)
        self.swiglu = SwiGLU(dim, hidden_dim)
        self.proj   = nn.Linear(hidden_dim, dim, bias=False)
        self.proj._is_residual_proj = True   # scaled init
        self.drop   = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.drop(self.proj(self.swiglu(x)))


# =========================================================
# Grouped Query Attention (GQA)
#
#   • RoPE applied to BOTH q and k (full head_dim each)
#   • N_KV_HEADS groups; set N_KV_HEADS == N_HEADS for full MHA,
#     N_KV_HEADS == 1 for pure MQA
#   • No learned absolute position embeddings — RoPE only
#   • bias=False on all projections
#   • Uses F.scaled_dot_product_attention (flash-attention path)
# =========================================================

class GroupedQueryAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0, \
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep      = n_heads // n_kv_heads
        self.head_dim   = dim // n_heads

        self.q = nn.Linear(dim, dim,                         bias=False)
        self.k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(dim, dim,                         bias=False)
        self.o._is_residual_proj = True   # scaled init

        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to BOTH q and k — full head_dim
        q = apply_rope(q)
        k = apply_rope(k)

        # Expand KV groups to match query heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=DROPOUT if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.o(out))


# =========================================================
# Transformer Block — Pre-LN
# =========================================================

class Block(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.ln1  = nn.LayerNorm(dim)
        self.ln2  = nn.LayerNorm(dim)
        self.attn = GroupedQueryAttention(dim, n_heads, n_kv_heads)
        self.ff   = FeedForward(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# =========================================================
# GPT
# =========================================================

class GPTConfig:
    def __init__(self, vocab_size, block_size=BLOCK_SIZE):
        self.vocab_size = vocab_size
        self.block_size = block_size


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        # N_HEADS comes from config; guard against float (e.g. N_LAYERS / 2)
        n_heads    = int(N_HEADS)
        n_kv_heads = int(N_KV_HEADS)

        assert HIDDEN_DIM % n_heads == 0, (
            f"HIDDEN_DIM ({HIDDEN_DIM}) must be divisible by N_HEADS ({n_heads}). "
            f"Current head_dim would be {HIDDEN_DIM / n_heads:.1f} — adjust either value in config.py."
        )

        self.embed = nn.Embedding(config.vocab_size, HIDDEN_DIM)

        self.blocks = nn.ModuleList([
            Block(HIDDEN_DIM, n_heads, n_kv_heads)
            for _ in range(N_LAYERS)
        ])

        self.ln   = nn.LayerNorm(HIDDEN_DIM)
        self.head = nn.Linear(HIDDEN_DIM, config.vocab_size, bias=False)

        # Initialise weights BEFORE tying so both matrices get proper init.
        # If tying happens first, _init_weights reinitialises embed.weight and
        # head.weight points to it — fine. But if called after, head.weight is
        # already an alias and the Linear branch of _init_weights would
        # overwrite the embedding init with a different std. Order matters.
        self.apply(self._init_weights)

        # Weight tying: output projection shares weights with embedding.
        # Done AFTER init so the single shared tensor has embedding-scale init.
        self.head.weight = self.embed.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Scale down residual-path projections (o-proj, ff proj) by
            # 1/sqrt(2*N_LAYERS) so the residual stream variance stays ~1
            # at initialisation regardless of depth (GPT-2 trick).
            std = 0.02
            if hasattr(module, "_is_residual_proj"):
                std = 0.02 / (2 * N_LAYERS) ** 0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape
        assert T <= self.block_size, \
            f"Sequence length {T} exceeds block_size {self.block_size}"

        x = self.embed(x)

        for block in self.blocks:
            x = block(x)

        x      = self.ln(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss