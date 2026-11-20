# transformer_block.py

import torch.nn as nn

from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.attention = MultiHeadAttention(hidden_size, num_heads)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x, past_kv=None):
        attention_output, kv_cache = self.attention(
            self.norm1(x), past_kv
        )
        x = x + attention_output

        x = x + self.mlp(self.norm2(x))

        return x, kv_cache