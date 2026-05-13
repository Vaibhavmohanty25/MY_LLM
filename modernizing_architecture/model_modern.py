import torch
import torch.nn as nn
from block_modern import BlockModern
from rmsnorm import RMSNorm
from utils import top_k_top_p_filtering

class GPTModern(nn.Module):
    """Modernized GPT architecture with RoPE, GQA, RMSNorm, and SwiGLU."""
    def __init__(self, vocab_size: int, n_layer: int, n_head: int, n_embd: int, 
                 block_size: int, dropout: float = 0.0, n_kv_head: int | None = None,
                 rope: bool = True, max_pos: int = 4096, 
                 sliding_window: int | None = None, attention_sink: int = 0,
                 use_rmsnorm: bool = True, use_swiglu: bool = True):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size, "n_layer": n_layer, "n_head": n_head,
            "n_embd": n_embd, "block_size": block_size, "dropout": dropout,
            "n_kv_head": n_kv_head or n_head, "rope": rope, "max_pos": max_pos,
            "sliding_window": sliding_window, "attention_sink": attention_sink,
            "use_rmsnorm": use_rmsnorm, "use_swiglu": use_swiglu
        }
        self.block_size = block_size
        
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            BlockModern(
                n_embd=n_embd, n_head=n_head, dropout=dropout,
                rope=rope, max_pos=max_pos, 
                sliding_window=sliding_window, attention_sink=attention_sink,
                n_kv_head=n_kv_head
            ) for _ in range(n_layer)
        ])
        self.ln_f = RMSNorm(n_embd) if use_rmsnorm else nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # weight tying
        self.tok_emb.weight = self.head.weight

        # init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        x = self.drop(x)

        for block in self.blocks:
            x, _ = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Simple generation loop. 
        Note: This implementation doesn't use KV caching for simplicity in this base class, 
        but could be optimized.
        """
        for _ in range(max_new_tokens):
            # crop to block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _, _ = self(idx_cond)
            # focus on last step
            logits = logits[:, -1, :] / temperature
            # filter
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
