import torch


#! token id
inputs = torch.tensor([
  [40, 367, 2885, 1464],
  [1807, 3619, 402, 271]
])

#! create token embedding layer

vocab_size = 50257
embedding_dim = 256

token_embedding_layer = torch.nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim
)

# convert token to embedding
token_embeddings = token_embedding_layer(inputs)



#! Step 3: Create Position Embedding Layer
context_length = inputs.shape[1]      # = 4

# Create position IDs
# positon ids are made from length
position_ids = torch.arange(context_length)
# we its positon embeddings, we need to know total tokens in single batch , no need of all batches


pos_embedding_layer = torch.nn.Embedding(
    num_embeddings=context_length,
    embedding_dim=embedding_dim
)


#! get position embedding
position_embeddings = pos_embedding_layer(position_ids)



#! token embedding + positional embedding
input_embeddings = token_embeddings + position_embeddings


