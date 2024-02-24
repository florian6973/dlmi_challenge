import subprocess

import logging
import os

import hydra
import mlflow
import torch
from dlmi.data.data import load_dataset
from dlmi.models.models import LinearModel, save_model
from dlmi.data.patientDataset import PatientDataset
from dlmi.utils.mlflow import log_params_from_omegaconf_dict
from hydra import utils
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dlmi.models.train_model import BasicModel

from tqdm import tqdm

def run_local_mlflow():
    subprocess.run('mlflow server --host 127.0.0.1 --port 5001'.split())

    

@hydra.main(config_path="dlmi/conf", config_name="train_resnet", version_base="1.1")
def launch(cfg: DictConfig):
    working_dir = os.getcwd()
    train_set_path = utils.get_original_cwd() + cfg.exp.data_path
    
    print("Working dir: ", working_dir)
    complete_train_set = PatientDataset(train_set_path)
    print("Train set loaded")
    train_set, val_set, mask_train = load_dataset(complete_train_set)

    model     = hydra.utils.instantiate(cfg.model)
    
    criterion = hydra.utils.instantiate(cfg.criterion)

    optimizer = hydra.utils.instantiate(
                    cfg.optimizer,
                    *[model.parameters()], 
                    **{"lr":cfg.train.lr}
                )

    # mlflow.set_tracking_uri("file://" + utils.get_original_cwd() + "/mlruns")
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")
    mlflow.set_experiment(cfg.mlflow.runname)
    # start new run
    with mlflow.start_run():
        log_params_from_omegaconf_dict(cfg) 

        batch_size = cfg.train.batch_size

        basic_model = BasicModel(model, criterion, optimizer)
        train_dataset = DataLoader(train_set, batch_size, shuffle=True)
        val_dataset = DataLoader(val_set, batch_size, shuffle=False)

        trainer = pl.Trainer(
            max_epochs=cfg.train.num_epochs
        )
        trainer.fit(basic_model, train_dataset, val_dataset)

        # train(cfg.train.num_epochs, cfg.train.batch_size, criterion, optimizer, model, train_set)

    save_model(working_dir + "/checkpoint.pt", model)
    logging.info(f"Checkpoint saved at {working_dir}")

    model = model.to("cuda")
    complete_test_set = PatientDataset(train_set_path, split="test")
    test_dataset = DataLoader(complete_test_set, batch_size, shuffle=False)
    run_infer(test_dataset, complete_test_set, model, "test", None)

    run_infer(val_dataset, complete_train_set, model, "val", (~mask_train))
    

def run_infer(dataset, main_dataset, model, name, mask=None):
    preds = []
    for batch in tqdm(dataset, desc=f"Predicting {name} set"):
        images, labels = batch
        images = images.to("cuda")
        y_pre = model(images)
        # print(y_pre)
        # print(labels)
        selected_class = torch.argmax(y_pre, dim=1)
        # print(selected_class)
        preds.append(selected_class)
    
    preds = torch.cat(preds)
    main_dataset.get_patient_labels(preds, mask, name)

    

    # TODO Maybe improve the logging of the training loop ?
    # TODO Vizualisation methods ?

# def pred():
#     import argparse
#     import os

#     args = argparse.ArgumentParser()
#     args.add_argument("dataset", help="Dataset type (train or test)")
#     args = args.parse_args()
#     train_set_path = os.getcwd() + "/data/raw"
    
#     print("Working dir: ", working_dir)
#     complete_train_set = PatientDataset(train_set_path)
    


if __name__ == "__main__":
    launch()
