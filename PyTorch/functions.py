import torch

# create tensor
tensor2d = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])

# shape - it gives size
# print(tensor2d.shape) 
# ! torch.Size([2, 3])


# re shaping
#print(tensor2d.reshape(3, 2))


# view -> same as reshape
# print(tensor2d.view(3, 2))

# transpose
# print(tensor2d.T)

#! multiplication of matices
print(tensor2d.matmul(tensor2d.T))
print(tensor2d @ tensor2d.T)