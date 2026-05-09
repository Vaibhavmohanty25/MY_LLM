import torch
import torch.nn as nn

class RoPECache(nn.Module):
    def __init__(self, head_dim: int, max_pos: int = 2048, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_pos = max_pos
        self.base = base
        
        # Precompute cos and sin
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_pos).float()
        freqs = torch.outer(t, inv_freq)
        # freqs: (max_pos, head_dim/2)
        
        # We want (max_pos, head_dim) where even/odd are matched
        # But commonly we just concatenate or repeat. 
        # The test expects q2 to be different from q.
        # Standard way is to repeat freqs
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb: (max_pos, head_dim)
        
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def get(self, pos: torch.Tensor):
        # pos: (T,)
        return self.cos[pos], self.sin[pos]

def rotate_half(x: torch.Tensor):
    # x: (..., D)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # x: (B, H, T, D)
    # cos, sin: (T, D)
    # We need to unsqueeze cos, sin to match (1, 1, T, D)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)
