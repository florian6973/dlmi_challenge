from torch.utils.data import Dataset
import torchvision.io as io
import pandas as pd
import glob
import os

class PatientDataset(Dataset):
    def __init__(self, root_dir, split="train"):

        if split == "train":
            self.data_dir = os.path.join(root_dir, "trainset")
        else:
            self.data_dir = os.path.join(root_dir, "testset")

        self.split = split
        self.patients = [p for p in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, p))]
        self.data = self.read_data_csv()

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, index):

        def load_and_preprocess_images(image_paths):
            processed_images = []
            for path in image_paths:
                image = io.read_image(path, io.ImageReadMode.RGB)
                image = image.permute(1, 2, 0).numpy()
                processed_images.append(image)

            return processed_images

        patient_dir = os.path.join(self.data_dir, self.patients[index])
        image_paths = [os.path.join(patient_dir, image) for image in os.listdir(patient_dir)]
        cur_patient = os.path.basename(patient_dir)
        
        images = load_and_preprocess_images(image_paths)
        label  = self.data.loc[self.data['ID'] == cur_patient, 'LABEL'].values[0]

        return images, label

        
    def read_data_csv(self):
        data_csv_path = os.path.join(self.data_dir, r'*.csv')
        csv_files = glob.glob(data_csv_path)
            
        if len(csv_files) == 0:
            raise FileNotFoundError(f'Could not find the dataset\'s csv ("{data_csv_path}")')
        elif len(csv_files) > 1:
            raise Exception("Multiple dataset csv files found")

        return pd.read_csv(csv_files[0])