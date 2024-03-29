from dlmi.data.CustomDataset import CustomDataset
from torch.utils.data import Subset
import torch

import numpy as np
from sklearn.model_selection import StratifiedKFold


def load_kfolds(dataset: CustomDataset, k=5):
    """
    Create k-fold splits from a dataset
    """
    
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    
    X = dataset.patients
    y = dataset.data['LABEL'].values
    sets = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        print(np.count_nonzero(y[train_index] == 1), np.count_nonzero(y[train_index] == 0))
        print(np.count_nonzero(y[test_index] == 1), np.count_nonzero(y[test_index] == 0))
        sets.append([
            Subset(dataset, train_index),
            Subset(dataset, test_index),
        ])
    return sets


def load_dataset(dataset: CustomDataset, train_proportion=0.8):
    """
    Deprecated:
    Please use load_kfolds instead
    """

    train_size = int(train_proportion * len(dataset.patients))
    val_size   = len(dataset.patients) - train_size

    # train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    mask_train = dataset.get_balanced_mask(train_size)

    subset_train_indices = dataset.get_indices_from_patient_mask(mask_train)
    
    train_set = Subset(dataset, subset_train_indices)

    subset_val_indices = dataset.get_indices_from_patient_mask(~mask_train)
    
    val_set = Subset(dataset, subset_val_indices)


    return train_set, val_set, mask_train
