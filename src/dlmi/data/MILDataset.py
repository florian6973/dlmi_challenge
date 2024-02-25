from dlmi.utils.utils import read_data_csv, read_image, plot_tensor_as_image
from dlmi.data.CustomDataset import CustomDataset
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import glob
import os


class MILDataset(Dataset, CustomDataset):
    def __init__(self, root_dir, split="train"):
        if split == "train":
            self.data_dir = os.path.join(root_dir, "trainset")
        else:
            self.data_dir = os.path.join(root_dir, "testset")

        self.split = split
        self.patients = [p for p in os.listdir(self.data_dir) \
                         if os.path.isdir(os.path.join(self.data_dir, p))]

        self.data = read_data_csv(self.data_dir)
        self.image_paths = glob.glob(self.data_dir + '/**/*.jpg')

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        cur_patient          = self.patients[idx]
        patient_images_paths = [p for p in self.image_paths if cur_patient in p]
        patient_images       = [read_image(path) for path in patient_images_paths]
        
        label = self.data.loc[self.data['ID'] == cur_patient, 'LABEL'].values[0]

        return torch.stack(patient_images), label
    
    def get_balanced_mask(self, train_size, seed=0):
        patients_labels = self.data.loc[self.data['ID'].isin(self.patients), 'LABEL']

        # sklearn balanced split
        from sklearn.model_selection import train_test_split

        train_patients, _ = train_test_split(self.patients, stratify=patients_labels, train_size=train_size, random_state=seed)
        mask = np.array([p in train_patients for p in self.patients])
        
        return mask
    
    def get_indices_from_patient_mask(self, mask):
        indices = []
        # print(self.patients)
        patients = []
        for i, p in enumerate(self.image_paths):
            patient = os.path.basename(os.path.dirname(p))
            patients.append(patient)
            if mask[np.where(np.array(self.patients) == patient)[0]] and patient in self.patients:
                indices.append(i)
        # print(patients)
        return indices

    def get_patient_labels(self, preds, mask=None, dataset="test"):
        patient_labels = []
        counters = {}
        k = 0
        for i, p in enumerate(self.image_paths):
            patient = os.path.basename(os.path.dirname(p))
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
            print(f"Balanced accuracy for {dataset}: {balanced_accuracy_score(df['LABEL_x'], df['LABEL_y'])}")

        return df
    
    
if __name__ == "__main__":

    path = r"C:\Users\gaumu\GitHub\dlmi_challenge\data\raw"

    data = MILDataset(path)

    img = iter(data).__next__()[0]

    import matplotlib.pyplot as plt

    
    plot_tensor_as_image(img[0])
    plot_tensor_as_image(img[1])