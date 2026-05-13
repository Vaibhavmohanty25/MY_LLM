import torch
import torch.nn as nn
from rmsnorm import RMSNorm
from swiglu import SwiGLU
from attn_modern import CausalSelfAttentionModern
from kv_cache import KVCache

class BlockModern(nn.Module):
    """A Transformer block with RMSNorm, SwiGLU, and CausalSelfAttentionModern (GQA/RoPE)."""
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0, 
                 rope: bool = True, max_pos: int = 4096, 
                 sliding_window: int | None = None, attention_sink: int = 0,
                 n_kv_head: int | None = None):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttentionModern(
            n_embd=n_embd, n_head=n_head, dropout=dropout,
            rope=rope, max_pos=max_pos, 
            sliding_window=sliding_window, attention_sink=attention_sink,
            n_kv_head=n_kv_head
        )
        self.ln2 = RMSNorm(n_embd)
        self.ffn = SwiGLU(n_embd)

    def forward(self, x: torch.Tensor, kv_cache: KVCache | None = None, start_pos: int = 0):
        # x: (B, T, C)
        h, new_kv = self.attn(self.ln1(x), kv_cache=kv_cache, start_pos=start_pos)
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x, new_kv
