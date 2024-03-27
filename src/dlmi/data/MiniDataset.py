import random
from dlmi.utils.utils import read_data_csv, read_image, plot_tensor_as_image
from dlmi.data.CustomDataset import CustomDataset
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import glob
import os
import torch.nn as nn
import random

class Sampler():
    def __init__(self, classes, class_per_batch, batch_size):
        self.classes = classes
        self.n_batches = sum([len(x) for x in classes]) // batch_size
        self.class_per_batch = class_per_batch
        self.batch_size = batch_size

    def __iter__(self):
        
        batches = []
        for _ in range(self.n_batches):
            classes = random.sample(range(len(self.classes)), self.class_per_batch)
            batch = []
            for i in range(self.batch_size):
                klass = random.choice(classes)
                batch.append(random.choice(self.classes[klass]))
            batches.append(batch)
        return iter(batches)
    
    # classes list of list
# https://stackoverflow.com/questions/66065272/customizing-the-batch-with-specific-elements

class MiniDataset(Dataset, CustomDataset):
    def __init__(self, root_dir, device, split="train"):
        if split == "train":
            self.data_dir = os.path.join(root_dir, "trainset")
        else:
            self.data_dir = os.path.join(root_dir, "testset")

        self.split = split
        self.device = device
        self.patients = [p for p in os.listdir(self.data_dir) \
                         if os.path.isdir(os.path.join(self.data_dir, p))]

        self.data = read_data_csv(self.data_dir)
        self.image_paths = glob.glob(self.data_dir + '/**/*.jpg')#[0:1000]

        self.data['GENDER'] = self.data['GENDER'].map({'M': 0, 'F': 1}) # 1 nan value
        self.data.loc[self.data.GENDER.isna(), "GENDER"] = 0.5
        self.data['AGE'] = (pd.to_datetime('2021-01-01') - pd.to_datetime(self.data['DOB'], format='mixed')).dt.days / (100*365) # scaling normalization
        self.data['LYMPH_COUNT'] = self.data['LYMPH_COUNT'].astype(np.float32) / 200 # scaling normalization

        # print(self.data['GENDER'].isna().sum())
        # print(self.data['DOB'].isna().sum())
        # print(self.data['LYMPH_COUNT'].isna().sum())
        # exit()

        self.images = {}
        self.build_classes()


    def __len__(self):
        return len(self.image_paths)
    
    def build_classes(self):
        self.train_classes = []
        self.test_classes = []
        current = []
        last_patient = self.get_patient(self.image_paths[0])
        is_train = random.random() < 0.8
        for i, image_path in enumerate(self.image_paths):
            patient = self.get_patient(image_path)
            if patient != last_patient:
                # self.classes.append(current)
                if is_train:
                    self.train_classes.append(current)
                else:
                    self.test_classes.append(current)
                current = []
                last_patient = patient
                is_train = random.random() < 0.8
            current.append(i)
        
        
    def get_patient(self, image_path):
        return os.path.basename(os.path.dirname(image_path))
            

    # https://stackoverflow.com/questions/66065272/customizing-the-batch-with-specific-elements
    def __getitem__(self, idx):
        # cur_patient          = self.patients[idx]
        # patient_images_paths = [p for p in self.image_paths if cur_patient in p]
        # # patient_images       = [read_image(path) for path in patient_images_paths]
        # patient_images = []
        
        path = self.image_paths[idx]
        if path not in self.images:
            self.images[path] = read_image(path).to(self.device)
        cur_patient = self.get_patient(path)
        current_image = self.images[path]
  
        label = self.data.loc[self.data['ID'] == cur_patient, 'LABEL'].values[0]
        age_count_features = self.data.loc[self.data['ID'] == cur_patient, ['GENDER', 'LYMPH_COUNT', 'AGE']].values[0].astype(np.float32)


        features = current_image #torch.stack(patient_images)
        # print("Returning patient images, age_count_features, label", torch.stack(patient_images).shape, age_count_features.shape, label)
        # features = nn.functional.pad(features, (0, 0, 0, 0, 0, 0, 0, 250 - features.shape[0]))
        # print("New shape", features.shape)
        # print(age_count_features)
        return [features, age_count_features], label
    
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
    
    
if __name__ == "__main__":
    pass
    # path = r"C:\Users\gaumu\GitHub\dlmi_challenge\data\raw"

    # data = MILDataset(path)

    # img = iter(data).__next__()[0]

    # import matplotlib.pyplot as plt

    
    # plot_tensor_as_image(img[0])
    # plot_tensor_as_image(img[1])