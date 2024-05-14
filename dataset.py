import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold

class EEGDataset(Dataset):
    def __init__(self, data, target_col=-1):
        if isinstance(data, pd.DataFrame):
            self.data = data.drop(columns=data.columns[target_col]).to_numpy(dtype=np.float32)
            self.targets = data.iloc[:,target_col].to_numpy(dtype=np.int64)
        elif isinstance(data, torch.Tensor):
            self.data = data[:, :target_col]
            self.targets = data[:, target_col]
        elif isinstance(data, np.ndarray):
            self.data = data[:, :target_col]
            self.targets = data[:, target_col]
        else:
            raise TypeError("Data must be a pandas dataframe, torch tensor, or numpy array")

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y
    
    def __len__(self):
        return len(self.data)

def k_fold_validation_split(df, batch_size=32, k_folds=2):
    n = len(df)
    fold_size = n // k_folds

    # Create folds
    folds = []
    for i in range(k_folds):
        start = i * fold_size
        end = start + fold_size
        folds += [df.iloc[start:end]]

    # Create dataset
    train_datasets = []
    val_datasets = []

    for i in range(k_folds):
        val_datasets += [EEGDataset(folds[i], target_col=-1)]
        temp = []
        for j in range(k_folds):
            if i != j:
                # Add everything other than the ith fold to the training set
                temp += [folds[j]]
        train_datasets += [EEGDataset(pd.concat(temp), target_col=-1)]

    # Create dataloader
    train_loaders = []
    val_loaders = []

    for i in range(k_folds):
        train_loaders += [DataLoader(train_datasets[i], batch_size=batch_size, shuffle=False)]
        val_loaders += [DataLoader(val_datasets[i], batch_size=batch_size, shuffle=False)]

    return train_loaders, val_loaders