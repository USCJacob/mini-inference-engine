import torch


@torch.inference_mode()
def generate(model, input_ids, max_new_tokens=10):

    for i in range(max_new_tokens):

        # [B, T] -> [B, T, vocab_size]
        logits = model(input_ids)

        # [B, vocab_size]
        last_logits = logits[:, -1, :]

        # [B, 1]
        next_token = torch.argmax(
            last_logits,
            dim=-1,
            keepdim=True
        )

        # [B, T] + [B, 1] -> [B, T+1]
        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1
        )

    return input_ids