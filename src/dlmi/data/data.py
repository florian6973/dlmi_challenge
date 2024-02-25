from dlmi.data.CustomDataset import CustomDataset
from torch.utils.data import Subset
import torch

def load_dataset(dataset: CustomDataset, train_proportion=0.8, seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_size = int(train_proportion * len(dataset.patients))
    val_size   = len(dataset.patients) - train_size

    # train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    mask_train = dataset.get_balanced_mask(train_size, seed)
    # print(mask_train)

    subset_train_indices = dataset.get_indices_from_patient_mask(mask_train)
    # print(subset_train_indices)
    train_set = Subset(dataset, subset_train_indices)

    subset_val_indices = dataset.get_indices_from_patient_mask(~mask_train)
    # print(subset_val_indices)
    val_set = Subset(dataset, subset_val_indices)


    return train_set, val_set, mask_train