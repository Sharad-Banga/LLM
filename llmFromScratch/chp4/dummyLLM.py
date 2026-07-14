import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		
		# create a lookup table for input embedding
		self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
		
		# create a lookup table for positional embedding
		self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
		
		self.drop_emb = nn.Dropout(cfg["drop_rate"])
		
		# create n_layer number of transformer layers
		self.trf_blocks = nn.Sequential(
		    # *[
		    #     DummyTransformerBlock(cfg)
		    #     for _ in range(cfg["n_layers"])
		    # ]
		)
		
		# normalize 
		# self.final_norm = DummyLayerNorm(cfg["emb_dim"])
		
		# nn.liner do -> output = input × weight + bias
		# means it transforms one vector into another.
		# we got context here of emb_dim size
		# as we have vocab_size size of vocab , we need score for each vocab
		# thts why we convert it into vocab_size
		# vocab with highest score is next prediction
		self.out_head = nn.Linear(
			 cfg["emb_dim"], cfg["vocab_size"], bias=False
		)
		
		
	def forward(self, in_idx):
		batch_size, seq_len = in_idx.shape
		# batch , seqlen
		# 2 sentences , 4 tokens each
		
		# create token embedding
		tok_embeds = self.tok_emb(in_idx)
		
		# create positional embedding
		pos_embeds = self.pos_emb(
		 torch.arange(seq_len, device=in_idx.device)
		)
		
		# final embedding
		x = tok_embeds + pos_embeds
		
		# drop 
		x = self.drop_emb(x)
		
		# pass embedding thru transformer
		x = self.trf_blocks(x)
		# we got cntxt vctr from transformer
		
		# normalization
		x = self.final_norm(x)
		
		# resize that into vocab size
		logits = self.out_head(x)
		return logits
		