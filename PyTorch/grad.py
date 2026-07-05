import torch
x = torch.tensor(2.0, requires_grad=True)

y = x * 3
z = y + 4

z.backward();

print(x.grad)