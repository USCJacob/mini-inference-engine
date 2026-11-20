import torch
import torch.nn as nn

from .transformer_block import TransformerBlock


class Transformer(nn.Module):
    def __init__(self, vocab_size=100, hidden_size=128,
                 num_heads=4, num_layers=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)

        self.lm_head = nn.Linear(
            hidden_size, vocab_size, bias=False
        )

    def forward(self, input_ids):
        # [B, T] -> [B, T, hidden_size]
        x = self.embedding(input_ids)

        # Pass through transformer blocks
        for layer in self.layers:
            x, _ = layer(x)

        x = self.norm(x)

        # [B, T, hidden_size] -> [B, T, vocab_size]
        logits = self.lm_head(x)

        return logits