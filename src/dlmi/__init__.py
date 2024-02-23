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

from dlmi.models.train_model import train

def run_local_mlflow():
    subprocess.run('mlflow server --host 127.0.0.1 --port 5001'.split())

    

@hydra.main(config_path="dlmi/conf", config_name="train_lenet", version_base="1.1")
def launch(cfg: DictConfig):
    working_dir = os.getcwd()
    train_set_path = utils.get_original_cwd() + cfg.exp.data_path
    
    complete_train_set = PatientDataset(train_set_path)
    train_set, val_set = load_dataset(complete_train_set)

    model     = hydra.utils.instantiate(cfg.model)
    
    criterion = hydra.utils.instantiate(cfg.criterion)

    optimizer = hydra.utils.instantiate(
                    cfg.optimizer,
                    *[model.parameters()], 
                    **{"lr":cfg.train.lr}
                )

    mlflow.set_tracking_uri("file://" + utils.get_original_cwd() + "/mlruns")
    mlflow.set_experiment(cfg.mlflow.runname)
    # start new run
    with mlflow.start_run():
        log_params_from_omegaconf_dict(cfg) 

        train(cfg.train.num_epochs, cfg.train.batch_size, criterion, optimizer, model, train_set)

    save_model(working_dir + "/checkpoint.pt", model)
    logging.info(f"Checkpoint saved at {working_dir}")
    

    # TODO Maybe improve the logging of the training loop ?
    # TODO Vizualisation methods ?


if __name__ == "__main__":
    launch()
