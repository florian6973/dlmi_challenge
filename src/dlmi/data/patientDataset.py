from torch.utils.data import Dataset
import torchvision.io as io
import pandas as pd
import glob
import os
import numpy as np

class PatientDataset(Dataset):
    def __init__(self, root_dir, split="train"):

        if split == "train":
            self.data_dir = os.path.join(root_dir, "trainset")
        else:
            self.data_dir = os.path.join(root_dir, "testset")

        self.split = split
        self.patients = [p for p in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, p))]
        self.data = self.read_data_csv()

        self.image_paths = glob.glob(self.data_dir + '/**/*.jpg', recursive=True)#[:100]
        self.images = {}

    def get_balanced_mask(self, train_size, seed=0):
        # np.random.seed(seed)
        # mask = np.zeros(len(self.patients), dtype=bool)
        patients_labels = self.data.loc[self.data['ID'].isin(self.patients), 'LABEL']

        # sklearn balanced split
        from sklearn.model_selection import train_test_split

        train_patients, _ = train_test_split(self.patients, stratify=patients_labels, train_size=train_size, random_state=seed)
        mask = np.array([p in train_patients for p in self.patients])
        # print(mask)
        # exit()

        # mask[np.random.choice(len(self.patients), train_size, replace=False)] = True
        return mask

    def get_indices_from_patient_mask(self, mask):
        indices = []
        # print(self.patients)
        patients = []
        for i, p in enumerate(self.image_paths):
            patient = os.path.basename(os.path.dirname(p))
            patients.append(patient)
            # print(patient, np.where(np.array(self.patients) == patient)[0])
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
            # df = df.append({"ID": p, "LABEL": np.argmax(counters[p])}, ignore_index=True)
            df.loc[len(df)] = [p, np.argmax(counters[p])]
        df.rename_columns({"ID":"Id", "LABEL":"Predicted"}).to_csv(f"submission_{dataset}.csv", index=False)
        if dataset != "test":
            true_labels = self.data.loc[self.data['ID'].isin(df['ID'])]

            # join the two dataframes
            df = df.merge(true_labels, on="ID")
            df.to_csv(f"submission_{dataset}_with_true_labels.csv", index=False)

            # balanced accuracy between LABEL_x and LABEL_y
            from sklearn.metrics import balanced_accuracy_score
            print(f"Balanced accuracy for {dataset}: {balanced_accuracy_score(df['LABEL_x'], df['LABEL_y'])}")

        return df

    def __len__(self):
        return len(self.image_paths)

    def read_image(self, index):
        cur_img_path = self.image_paths[index]
        cur_patient  = os.path.basename(os.path.dirname(cur_img_path))
        
        image = io.read_image(cur_img_path, io.ImageReadMode.RGB) / 255

        return image
        
    def __getitem__(self, index):
        cur_img_path = self.image_paths[index]
        cur_patient  = os.path.basename(os.path.dirname(cur_img_path))

        if index not in self.images:
            self.images[index] = self.read_image(index)

        image = self.images[index]
        label  = self.data.loc[self.data['ID'] == cur_patient, 'LABEL'].values[0]

        return image, label

        
    def read_data_csv(self):
        data_csv_path = os.path.join(self.data_dir, r'*.csv')
        csv_files = glob.glob(data_csv_path)
            
        if len(csv_files) == 0:
            raise FileNotFoundError(f'Could not find the dataset\'s csv ("{data_csv_path}")')
        elif len(csv_files) > 1:
            raise Exception("Multiple dataset csv files found")

        return pd.read_csv(csv_files[0])