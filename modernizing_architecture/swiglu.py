import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit.
    
    Reference: https://arxiv.org/abs/2002.05202
    """
    def __init__(self, n_embd: int, hidden_dim: int | None = None):
        super().__init__()
        # Usually hidden_dim is 4 * n_embd, but common in Llama-style is 2/3 * 4 * n_embd
        hidden_dim = hidden_dim or int(4 * n_embd * 2 / 3)
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, n_embd, bias=False)

    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
