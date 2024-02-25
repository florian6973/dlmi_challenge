import os
import glob
import pandas as pd

def read_data_csv(data_dir):
    data_csv_path = os.path.join(data_dir, r'*.csv')
    csv_files = glob.glob(data_csv_path)
        
    if len(csv_files) == 0:
        raise FileNotFoundError(f'Could not find the dataset\'s csv ("{data_csv_path}")')
    elif len(csv_files) > 1:
        raise Exception("Multiple dataset csv files found")

    return pd.read_csv(csv_files[0])


def read_image(image_path):
    import torchvision.io as io

    return io.read_image(image_path, io.ImageReadMode.RGB) / 255

def plot_tensor_as_image(tensor):

    import matplotlib.pyplot as plt

    plt.imshow(tensor.permute(1, 2, 0))
    plt.show()


