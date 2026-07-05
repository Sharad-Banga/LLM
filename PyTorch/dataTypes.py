
import torch
# tensor1d = torch.tensor([1,2,3]);

tensor2d = torch.tensor([[1,2],
                         [3,4]])

f = tensor2d.to(torch.float32);

print(f.dtype)