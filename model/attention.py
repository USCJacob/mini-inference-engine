import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    q/k/v:
        [batch_size, num_heads, new_len, head_dim]

    past_kv:
        历史 K/V，各为：
        [batch_size, num_heads, past_len, head_dim]

    返回：
        output:
            [batch_size, num_heads, new_len, head_dim]
        present_kv:
            包含历史和本次 token 的完整 K/V
    """
    new_len = q.shape[-2]
    head_dim = q.shape[-1]
    past_len = 0

    # 1. 拼接历史 KV cache。
    if past_kv is not None:
        past_k, past_v = past_kv
        past_len = past_k.shape[-2]

        # dim=-2 是序列维度。
        k = torch.cat([past_k, k], dim=-2)
        v = torch.cat([past_v, v], dim=-2)

    present_kv = (k, v)
    total_len = k.shape[-2]

    # 2. 计算 Q 和 K 的相似度，并缩放。
    # [B, H, new_len, D] @ [B, H, D, total_len]
    # → [B, H, new_len, total_len]
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(head_dim)

    # 3. Causal mask：每个 query 只能看到自己及之前的位置。
    #
    # 例如 past_len=3、new_len=2：
    # query_positions = [3, 4]
    # key_positions   = [0, 1, 2, 3, 4]
    query_positions = past_len + torch.arange(
        new_len, device=q.device
    )
    key_positions = torch.arange(total_len, device=q.device)

    # [new_len, total_len]，True 表示需要屏蔽。
    causal_mask = (
        key_positions[None, :] > query_positions[:, None]
    )

    # 自动广播到所有 batch 和 heads。
    scores = scores.masked_fill(causal_mask, float("-inf"))

    # 4. 沿 key 维度归一化，得到 attention 权重。
    # 使用 float32 计算 softmax，再转回 V 的 dtype。
    weights = torch.softmax(
        scores, dim=-1, dtype=torch.float32
    ).to(v.dtype)

    # 5. 对 V 加权求和。
    # [B, H, new_len, total_len] @ [B, H, total_len, D]
    # → [B, H, new_len, D]
    output = weights @ v

    return output, present_kv


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()

        if hidden_size <= 0 or num_heads <= 0:
            raise ValueError("hidden_size 和 num_heads 必须大于 0")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size 必须能被 num_heads 整除")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # 创建可学习的权重矩阵。
        # 随机初始化；推理实际模型时需要加载预训练权重。
        self.q_proj = nn.Linear(
            hidden_size, hidden_size, bias=False
        )
        self.k_proj = nn.Linear(
            hidden_size, hidden_size, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, hidden_size, bias=False
        )
        self.out_proj = nn.Linear(
            hidden_size, hidden_size, bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        x:
            [batch_size, new_len, hidden_size]

        past_kv:
            本层的历史 K/V cache。

        返回：
            output:
                [batch_size, new_len, hidden_size]
            present_kv:
                本层更新后的 K/V cache。
        """
        batch_size, new_len, _ = x.shape

        # 1. 用不同的权重矩阵，把 x 转换成 Q、K、V。
        # 每个结果都是 [B, new_len, hidden_size]。
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. 拆分 heads。
        # [B, T, hidden_size]
        # → [B, T, num_heads, head_dim]
        # → [B, num_heads, T, head_dim]
        q = q.reshape(
            batch_size, new_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = k.reshape(
            batch_size, new_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        v = v.reshape(
            batch_size, new_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # 3. 计算各个 head 的 attention。
        output, present_kv = scaled_dot_product_attention(
            q, k, v, past_kv=past_kv
        )

        # 4. 合并 heads。
        # [B, num_heads, T, head_dim]
        # → [B, T, num_heads, head_dim]
        # → [B, T, hidden_size]
        output = output.transpose(1, 2).contiguous()
        output = output.view(
            batch_size, new_len, self.hidden_size
        )

        # 5. 输出投影，融合不同 heads 的信息。
        output = self.out_proj(output)

        return output, present_kv


if __name__ == "__main__":
    torch.manual_seed(42)

    attention = MultiHeadAttention(
        hidden_size=128,
        num_heads=4,
    )
    attention.eval()

    with torch.inference_mode():
        # 模拟 6 个 token 的 hidden states。
        x = torch.randn(1, 6, 128)

        # Prefill：先处理前 5 个 token。
        prefill_output, kv_cache = attention(x[:, :5])

        print("Prefill output:", prefill_output.shape)
        # torch.Size([1, 5, 128])

        print("Prefill K cache:", kv_cache[0].shape)
        # torch.Size([1, 4, 5, 32])

        # Decode：利用 cache，处理第 6 个 token。
        decode_output, kv_cache = attention(
            x[:, 5:],
            past_kv=kv_cache,
        )

        print("Decode output:", decode_output.shape)
        # torch.Size([1, 1, 128])

        print("Decode K cache:", kv_cache[0].shape)
        # torch.Size([1, 4, 6, 32])

        # 验证：使用 cache 的结果应与一次处理完整序列一致。
        full_output, _ = attention(x)

        torch.testing.assert_close(
            decode_output,
            full_output[:, 5:],
            rtol=1e-4,
            atol=1e-5,
        )

        print("KV cache 一致性验证通过")