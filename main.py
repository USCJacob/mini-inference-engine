import torch

from model.transformer import Transformer
from engine.generation import generate

model = Transformer(
    vocab_size=100,
    hidden_size=128,
    num_heads=4,
    num_layers=2
)

model.eval()

input_ids = torch.tensor([[1, 2, 3]])

with torch.inference_mode():
    output = generate(
        model,
        input_ids,
        max_new_tokens=10
    )

print("Input:", input_ids)
print("Output:", output)
print("Output shape:", output.shape)