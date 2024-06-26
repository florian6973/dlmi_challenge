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

class MILDataset(Dataset, CustomDataset):
    """MIL dataset.
    One item in the dataset is a patient with multiple images.
    """
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
        self.image_paths = glob.glob(self.data_dir + '/**/*.jpg')

        self.data['GENDER'] = self.data['GENDER'].map({'M': 0, 'F': 1})
        self.data.loc[self.data.GENDER.isna(), "GENDER"] = 0.5   # 1 nan value in the dataset
        self.data['AGE'] = (pd.to_datetime('2021-01-01') - pd.to_datetime(self.data['DOB'], format='mixed')).dt.days / (100*365) # scaling normalization
        self.data['LYMPH_COUNT'] = self.data['LYMPH_COUNT'].astype(np.float32) / 200 # scaling normalization

        self.images = {}

        # Uncomment the following lines to preload all images in the memory
        # for path in tqdm(self.image_paths):
        #     self.images[path] = read_image(path).to(self.device)

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        cur_patient       = self.patients[idx]
        label             = self.data.loc[self.data['ID'] == cur_patient, 'LABEL'].values[0]
        clinical_features = self.data.loc[self.data['ID'] == cur_patient, ['GENDER', 'AGE', 'LYMPH_COUNT']].values[0].astype(np.float32)

        patient_images_paths = [p for p in self.image_paths if cur_patient in p]
        
        patient_images = []
        for path in patient_images_paths:
            if path not in self.images:
                self.images[path] = read_image(path).to(self.device)
            patient_images.append(self.images[path])
            
        image_features = torch.stack(patient_images)

        # we make sure that there are no more than 300 images per patient
        image_features = image_features[:300]
        
        # we pad the images with zeros if there are less than 300 images
        # to get a fixed size tensor for PyTorch
        # this tensor will be unpad later in the model
        image_features = nn.functional.pad(image_features, (0, 0, 0, 0, 0, 0, 0, 301 - image_features.shape[0]))

        return [image_features, clinical_features], label
    
    def get_balanced_mask(self, train_size):
        """deprecated since we are using load_kfolds"""
        patients_labels = self.data.loc[self.data['ID'].isin(self.patients), 'LABEL']

        from sklearn.model_selection import train_test_split

        train_patients, _ = train_test_split(self.patients, stratify=patients_labels, train_size=train_size)
        mask = np.array([p in train_patients for p in self.patients])

        # check if the mask is balanced
        print(f"Train set balanced: {np.mean(patients_labels[mask])} with {np.count_nonzero(mask)} samples")
        print(f"Test set balanced: {np.mean(patients_labels[~mask])} with {np.count_nonzero(~mask)} samples")
        
        return mask
    
    def get_indices_from_patient_mask(self, mask):
        """convert a mask of patients to a mask of indices"""
        return np.where(mask)[0]

    def get_patient_labels(self, preds, mask=None, dataset="test", fold=0):
        """
        Write the predictions to a csv file (function should be renamed)
        Can be done for the test set or the validation set (for a specific fold)
        """
        patient_labels = []
        counters = {}
        k = 0
        for i, p in enumerate(self.patients):
            patient = p
            if mask is not None and not mask[np.where(np.array(self.patients) == patient)[0]]:
                continue
            if patient not in counters:
                counters[patient] = [0, 0]
            counters[patient][preds[k].item()] += 1
            k += 1
        import json
        with open(f'counters_{dataset}_{fold}.json', 'w') as f:
            json.dump(counters, f)
        df = pd.DataFrame(columns=["ID", "LABEL"])
        for p in counters:
            df.loc[len(df)] = [p, np.argmax(counters[p])]
        df.rename(columns={"ID":"Id", "LABEL":"Predicted"}).to_csv(f"submission_{dataset}_{fold}.csv", index=False)
        if dataset != "test":
            true_labels = self.data.loc[self.data['ID'].isin(df['ID'])]

            # join the two dataframes
            df = df.merge(true_labels, on="ID")
            df.to_csv(f"submission_{dataset}_{fold}_with_true_labels.csv", index=False)

            # balanced accuracy between LABEL_x and LABEL_y
            from sklearn.metrics import balanced_accuracy_score
            bac = balanced_accuracy_score(df['LABEL_y'], df['LABEL_x'])
            print(f"Balanced accuracy for {dataset}: {bac}")
            return bac
        else:
            return 0