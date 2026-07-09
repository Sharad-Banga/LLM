import torch


attention_score = torch.tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])

#softmax fxn
def softmax_naive(x):
 return torch.exp(x) / torch.exp(x).sum(dim=0)



attention_weights = softmax_naive(attention_score)

# this weights are acc to the word selected -> journey
print(attention_weights)