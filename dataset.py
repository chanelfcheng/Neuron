import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, KFold

class EEGDataset(Dataset):
    def __init__(self, data, target_col):
        self.data = data.drop(columns=[target_col]).to_numpy(dtype=np.float32)
        self.targets = data[target_col].to_numpy(dtype=np.int64)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y
    
    def __len__(self):
        return len(self.data)
    
def k_fold_validation_split(data, target_col, k=5):
    data = data.sample(frac=1).reset_index(drop=True)
    X = data.drop(columns=[target_col]).to_numpy(dtype=np.float32)
    y = data[target_col].to_numpy(dtype=np.int64)
    kf = KFold(n_splits=k, shuffle=True)
    splits = kf.split(X, y)
    return splits