import torch
import torchvision
from dlip.data.usps import download_usps
from torch.utils.data import random_split
from torchvision import transforms

def load_dataset(dataset, train_proportion=0.8, seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_size = int(train_proportion * len(dataset))
    val_size   = len(dataset) - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    return train_set, val_set