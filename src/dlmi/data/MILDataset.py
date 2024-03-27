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
    def __init__(self, root_dir, device,split="train", load_images=False):
        if split == "train":
            self.data_dir = os.path.join(root_dir, "trainset")
        else:
            self.data_dir = os.path.join(root_dir, "testset")

        self.load_images = load_images
        self.device = device
        self.split = split
        self.patients = [p for p in os.listdir(self.data_dir) \
                         if os.path.isdir(os.path.join(self.data_dir, p))]

        self.data = read_data_csv(self.data_dir)
        self.image_paths = glob.glob(self.data_dir + '/**/*.jpg')
        self.images = defaultdict(list)

        if load_images:
            for path in tqdm(self.image_paths, desc=f"Loading images"):
                patient = os.path.basename(os.path.dirname(path))
                self.images[patient].append(read_image(path))

        self.data['GENDER'] = self.data['GENDER'].map({'M': 0, 'F': 1}) # 1 nan value
        self.data.loc[self.data.GENDER.isna(), "GENDER"] = 0.5
        self.data['AGE'] = (pd.to_datetime('2021-01-01') - pd.to_datetime(self.data['DOB'], format='mixed')).dt.days / (100*365) # scaling normalization
        self.data['LYMPH_COUNT'] = self.data['LYMPH_COUNT'].astype(np.float32) / 200 # scaling normalization

        # print(self.data['GENDER'].isna().sum())
        # print(self.data['DOB'].isna().sum())
        # print(self.data['LYMPH_COUNT'].isna().sum())
        # exit()

        self.images = {}

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        cur_patient       = self.patients[idx]
        label             = self.data.loc[self.data['ID'] == cur_patient, 'LABEL'].values[0]
        clinical_features = self.data.loc[self.data['ID'] == cur_patient, ['GENDER', 'AGE', 'LYMPH_COUNT']].values[0].astype(np.float32)

        if self.load_images:
            
            patient_images_paths = [p for p in self.image_paths if cur_patient in p]
            
            patient_images = []
            for path in patient_images_paths:
                if path not in self.images:
                    self.images[path] = read_image(path).to("cuda")
                patient_images.append(self.images[path])
            
            

            image_features = torch.stack(patient_images)
            # print("Returning patient images, clinical_features, label", torch.stack(patient_images).shape, clinical_features.shape, label)
            image_features = nn.functional.pad(image_features, (0, 0, 0, 0, 0, 0, 0, 250 - image_features.shape[0]))

        else:
            # patient_images_paths = [p for p in self.image_paths if cur_patient in p]

            for idx, path in enumerate(patient_images_paths):
                patient_images[idx] = (read_image(path))
                

            # patient_images = self.images[cur_patient]

        # image_features = torch.stack(patient_images)
        
        image_features = nn.functional.pad(patient_images, (0, 0, 0, 0, 0, 0, 0, 250 - patient_images.shape[0]))

        # print("New shape", features.shape)
        return [image_features, clinical_features], label
    
    def get_balanced_mask(self, train_size):
        patients_labels = self.data.loc[self.data['ID'].isin(self.patients), 'LABEL']

        # sklearn balanced split
        from sklearn.model_selection import train_test_split

        train_patients, _ = train_test_split(self.patients, stratify=patients_labels, train_size=train_size)
        mask = np.array([p in train_patients for p in self.patients])

        # print(np.count_nonzero(mask), np.count_nonzero(~mask))

        # check if the mask is balanced
        print(f"Train set balanced: {np.mean(patients_labels[mask])} with {np.count_nonzero(mask)} samples")
        print(f"Test set balanced: {np.mean(patients_labels[~mask])} with {np.count_nonzero(~mask)} samples")
        
        return mask
    
    def get_indices_from_patient_mask(self, mask):
        return np.where(mask)[0]
        # indices = []
        # patients = []
        # for i, p in enumerate(self.image_paths):
        #     patient = os.path.basename(os.path.dirname(p))
        #     patients.append(patient)
        #     if mask[np.where(np.array(self.patients) == patient)[0]] and patient in self.patients:
        #         indices.append(i)
        # # print(patients)
        # return indices

    def get_patient_labels(self, preds, mask=None, dataset="test"):
        patient_labels = []
        counters = {}
        k = 0
        for i, p in enumerate(self.patients):
            # patient = os.path.basename(os.path.dirname(p))
            patient = p
            if mask is not None and not mask[np.where(np.array(self.patients) == patient)[0]]:
                continue
            if patient not in counters:
                counters[patient] = [0, 0]
            counters[patient][preds[k].item()] += 1
            k += 1
            # patient_labels.append([patient, preds[i].item()])
        import json
        with open(f'counters_{dataset}.json', 'w') as f:
            json.dump(counters, f)
        df = pd.DataFrame(columns=["ID", "LABEL"])
        for p in counters:
            df.loc[len(df)] = [p, np.argmax(counters[p])]
        df.rename(columns={"ID":"Id", "LABEL":"Predicted"}).to_csv(f"submission_{dataset}.csv", index=False)
        if dataset != "test":
            true_labels = self.data.loc[self.data['ID'].isin(df['ID'])]

            # join the two dataframes
            df = df.merge(true_labels, on="ID")
            df.to_csv(f"submission_{dataset}_with_true_labels.csv", index=False)

            # balanced accuracy between LABEL_x and LABEL_y
            from sklearn.metrics import balanced_accuracy_score
            print(f"Balanced accuracy for {dataset}: {balanced_accuracy_score(df['LABEL_y'], df['LABEL_x'])}")

        return df