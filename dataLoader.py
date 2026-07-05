import torch
 
from torch.utils.data import DataLoader

torch.manual_seed(123)

train_loader = DataLoader(
  dataset=train_ds,
  batch_size=2,
  shuffle=true,
  num_workers=0
)

train_loader = DataLoader(
  dataset=test_ds,
  batch_size=2,
  shuffle=true,
  num_workers=0
)


# training loader that drops the last batch
train_loader = DataLoader(
 dataset=train_ds,
 batch_size=2,
 shuffle=True,
 num_workers=0,
 drop_last=True
)