import torch
from torch.utils.data import Dataset

class ToyDataset(Dataset):
  def __init__(self , X,Y):
    self.features = X
    self.labels = Y

  def __getitem__(self , index):
    one_x = self.features[index]
    one_y = self.labels[index]
    return one_x, one_y 
  
  def __len__(self):
    return self.labels.shape[0]

# train_ds = ToyDataset(X_train,Y_train)
# test_ds = ToyDataset(X_test,Y_test)