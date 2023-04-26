import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold

class EEGDataset(Dataset):
    def __init__(self, data, target_col):
        if isinstance(data, pd.DataFrame):
            self.data = data.drop(columns=[target_col]).to_numpy(dtype=np.float32)
            self.targets = data[target_col].to_numpy(dtype=np.int64)
        elif isinstance(data, torch.Tensor):
            self.data = data[:, :-1]
            self.targets = data[:, -1]
        else:
            raise TypeError("Data must be a pandas dataframe or a torch tensor")

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y
    
    def __len__(self):
        return len(self.data)
    
def epoch_data(filtered, time_window=0.5, overlap=0):
    # Epoch data into 2 second windows
    events = mne.make_fixed_length_events(filtered, duration=time_window, overlap=overlap)
    epochs = mne.Epochs(filtered, events, tmin=0, tmax=time_window, baseline=None, preload=True)

    return epochs

def k_fold_validation_split(df, batch_size=32, k_folds=2):
    n = len(df)
    k_folds = 2
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
        val_datasets += [EEGDataset(folds[i], target_col='focus')]
        temp = []
        for j in range(k_folds):
            if i != j:
                # Add everything other than the ith fold to the training set
                temp += [folds[j]]
        train_datasets += [EEGDataset(pd.concat(temp), target_col='focus')]

    # Create dataloader
    train_loaders = []
    val_loaders = []

    for i in range(k_folds):
        train_loaders += [DataLoader(train_datasets[i], batch_size=batch_size, shuffle=True)]
        val_loaders += [DataLoader(val_datasets[i], batch_size=batch_size, shuffle=True)]

    return train_loaders, val_loaders