import torch


class CausalAttention(nn.Module):
  # d_in -> dimension of each word
  # d_out -> dimension of output word
  # context length -> how much words it can remember (memory size)
  # dropout -> making any attention weight 0 , to make context of that word zero
  def __init__(self, d_in, d_out, context_length,
    dropout, qkv_bias=False):
    # 
    super().__init__()
    # this are layers that will take word in dim d_in
    # and give Q,K,V as output in dim d_out
    self.W_query =  torch.nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key   =  torch.nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value =  torch.nn.Linear(d_in, d_out, bias=qkv_bias)
    self.dropout =  torch.nn.Dropout(dropout) 

    # creates matrix[cont_len,cont_len] , diagnol=1-> below tri = 0
    self.register_buffer(
      'mask',
      torch.triu(torch.ones(context_length, context_length),
                            diagonal=1)
    )

  def forward(self, x):
    b, num_tokens, d_in = x.shape
    keys = self.W_key(x)
    queries = self.W_query(x)
    values = self.W_value(x)

    attn_scores = queries @ keys.transpose(1, 2)
    
    attn_scores.masked_fill_(
      self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
    
    # self.mask.bool()[:num_tokens, :num_tokens] -> above triangle true
    # and below false
    # 
    # masked_fill will replace true places in attn_score with -inf
     
    
    attn_weights = torch.softmax(
      attn_scores / keys.shape[-1]**0.5, dim=-1)
    
    attn_weights = self.dropout(attn_weights)

    context_vec = attn_weights @ values
    return context_vec