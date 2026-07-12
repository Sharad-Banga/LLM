import torch

class MultiHeadAttentionWrapper(nn.Module):
  def __init__(self, d_in, d_out, context_length,
    dropout, num_heads, qkv_bias=False):
    super().__init__()
    self.heads = torch.nn.ModuleList(
      [CausalAttention(
          d_in, d_out, context_length, dropout, qkv_bias
      )
      for _ in range(num_heads)]
    )

  def forward(self, x):
    # concating all contect_vectors
    return torch.cat([head(x) for head in self.heads], dim=-1)