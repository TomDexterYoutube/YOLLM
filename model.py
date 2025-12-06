import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from config import *

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x):
    # Rotary positional embedding (RoPE)
    # x: (B, n_heads, T, head_dim)
    B, n_heads, T, head_dim = x.shape
    half_dim = head_dim // 2
    theta = 10000 ** (-torch.arange(0, half_dim, dtype=torch.float32, device=x.device) / half_dim)
    pos = torch.arange(T, dtype=torch.float32, device=x.device)
    freqs = torch.einsum('i,j->ij', pos, theta)
    emb = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)
    emb = emb[None, None, :, :]  # (1, 1, T, head_dim)
    x_rot = x[..., :half_dim] * emb[..., :half_dim] + rotate_half(x[..., :half_dim]) * emb[..., half_dim:]
    x = torch.cat([x_rot, x[..., half_dim:]], dim=-1)
    return x

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
    def forward(self, x):
        return F.silu(self.w1(x)) * self.w2(x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return x + self.fn(self.norm(x), **kwargs)

class MultiQueryAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = int(n_heads)
        self.head_dim = dim // self.n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, self.head_dim)
        self.v_proj = nn.Linear(dim, self.head_dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(DROPOUT)
        self.resid_dropout = nn.Dropout(DROPOUT)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = self.k_proj(x).unsqueeze(1).expand(B, self.n_heads, T, self.head_dim)    # (B, n_heads, T, head_dim)
        v = self.v_proj(x).unsqueeze(1).expand(B, self.n_heads, T, self.head_dim)    # (B, n_heads, T, head_dim)
        # RoPE
        q = apply_rope(q)
        k = apply_rope(k)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        att = att.masked_fill(causal_mask == 0, float('-inf'))
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))

class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = int(dim * FEEDFORWARD_MULTIPLIER)
        self.swiglu = SwiGLU(dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, x):
        return self.dropout(self.proj(self.swiglu(x)))

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attn = PreNormResidual(dim, MultiQueryAttention(dim, n_heads))
        self.ff = PreNormResidual(dim, FeedForward(dim))
    def forward(self, x, mask=None):
        x = self.attn(x, mask=mask)
        x = self.ff(x)
        return x

class GPTConfig:
    def __init__(self, vocab_size, block_size=BLOCK_SIZE, dim=HIDDEN_DIM, heads=N_HEADS, layers=N_LAYERS):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.dim = dim
        self.heads = int(heads)
        self.layers = int(layers)

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embed = nn.Embedding(config.vocab_size, config.dim)
        self.pos_embed = nn.Embedding(config.block_size, config.dim)
        self.blocks = nn.ModuleList([TransformerBlock(config.dim, config.heads) for _ in range(config.layers)])
        self.ln_f = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None, mask=None):
        B, T = x.size()
        tok = self.token_embed(x)
        pos = self.pos_embed(torch.arange(T, device=x.device))
        x = tok + pos
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, prompt, max_new_tokens=50, tokenizer=None, device=None):
        self.eval()
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for text generation.")
        if device is None:
            device = next(self.parameters()).device

        input_ids = tokenizer.encode(prompt)
        if hasattr(input_ids, "ids"):
            input_ids = input_ids.ids
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
        for _ in range(max_new_tokens):
            if input_ids.size(1) >= self.pos_embed.num_embeddings:
                break
            with torch.no_grad():
                logits, _ = self(input_ids)
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        try:
            return tokenizer.decode(input_ids[0].tolist())
        except Exception as e:
            return f"[DECODE ERROR: {e}]"
