from dlmi.utils.utils import read_data_csv, read_image, plot_tensor_as_image
from dlmi.data.CustomDataset import CustomDataset
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import glob
import os
import torch.nn as nn

class FinalDataset(Dataset, CustomDataset):
    def __init__(self, root_dir, device,split="train"):
        if split == "train":
            self.data_dir = os.path.join(root_dir, "trainset")
        else:
            self.data_dir = os.path.join(root_dir, "testset")

        self.device = device
        self.split = split
        self.patients = [p for p in os.listdir(self.data_dir) \
                         if os.path.isdir(os.path.join(self.data_dir, p))]

        self.data = read_data_csv(self.data_dir)
        # self.image_paths = glob.glob(self.data_dir + '/**/*.jpg')
        # self.images = defaultdict(list)

        self.data['GENDER'] = self.data['GENDER'].map({'M': 0, 'F': 1}) # 1 nan value
        self.data.loc[self.data.GENDER.isna(), "GENDER"] = 0.5
        self.data['DOB'] = (pd.to_datetime('2021-01-01') - pd.to_datetime(self.data['DOB'], format='mixed')).dt.days / (100*365) # scaling normalization
        self.data['LYMPH_COUNT'] = self.data['LYMPH_COUNT'].astype(np.float32) / 200


    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        cur_patient       = self.patients[idx]
        label             = self.data.loc[self.data['ID'] == cur_patient, 'LABEL'].values[0]
        clinical_features = self.data.loc[self.data['ID'] == cur_patient, ['GENDER', 'DOB', 'LYMPH_COUNT']].values[0].astype(np.float32)
        
        # patient_images_paths = [p for p in self.image_paths if cur_patient in p]
        # patient_images = torch.zeros([len(patient_images_paths), 3, 224, 224])
        
      

        # print("New shape", features.shape)
        return [None, clinical_features], label