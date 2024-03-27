import subprocess

import gc
import os

import numpy as np

import hydra
import mlflow
import torch
from dlmi.data.data import load_dataset
from dlmi.data.MILDataset import MILDataset
from dlmi.data.MiniDataset import MiniDataset, Sampler
from dlmi.utils.mlflow import log_params_from_omegaconf_dict
from dlmi.data.MiniDataset import MiniDataset, Sampler

from hydra import utils
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint

# from ray import train, tune
import ray.tune as tune
import ray.train as train
from ray.air.integrations.mlflow import setup_mlflow
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler


from tqdm import tqdm

def run_local_mlflow():
    subprocess.run('mlflow server --host 127.0.0.1 --port 5001'.split())

    

@hydra.main(config_path="dlmi/conf", config_name="train_moe", version_base="1.1")
def launch(cfg: DictConfig):
    working_dir = os.getcwd()
    train_set_path = utils.get_original_cwd() + cfg.exp.data_path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # set_seed(cfg.exp.seed)
    
    print("Working dir: ", working_dir)
    # complete_train_set = PatientDataset(train_set_path)
    # complete_train_set = MILDataset(train_set_path)

    if cfg.dataset_type == "MiniDataset":
        complete_train_set = MiniDataset(train_set_path, device)
        print("Train set loaded")
        train_set, val_set, mask_train = None, None, None
    elif cfg.dataset_type == "MILDataset":
        complete_train_set = MILDataset(train_set_path, device)
        print("Train set loaded")
        train_set, val_set, mask_train = load_dataset(complete_train_set)
    else:
        raise ValueError("Dataset type not supported, must be MiniDataset or MILDataset")


    # mlflow.set_tracking_uri("file://" + utils.get_original_cwd() + "/mlruns")
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")
    mlflow.set_experiment(cfg.mlflow.runname)
    # start new run
    # with mlflow.start_run():
    #     log_params_from_omegaconf_dict(cfg) 
        
    #     model = hydra.utils.instantiate(cfg.model, cfg)

    #     batch_size = cfg.train.batch_size

    #     # basic_model = BasicModel(model, criterion, optimizer)
    #     # basic_model   = MILModel(model, criterion, optimizer)
    #     train_dataset = DataLoader(train_set, batch_size, shuffle=True, num_workers=0)
    #     val_dataset   = DataLoader(val_set,   batch_size, shuffle=False, num_workers=0)

    #     checkpoint_callback = ModelCheckpoint(monitor="val_negacc")

    #     trainer = pl.Trainer(
    #         max_epochs=cfg.train.num_epochs,
    #         accelerator="gpu",
    #         precision=16,
    #         log_every_n_steps=10,
    #         callbacks=[checkpoint_callback]
    #     )
    #     trainer.fit(model, train_dataset, val_dataset)

    #     model.load_state_dict(torch.load(checkpoint_callback.best_model_path)["state_dict"])

        # train(cfg.train.num_epochs, cfg.train.batch_size, criterion, optimizer, model, train_set)


    if "tune" in cfg.exp.keys() and cfg.exp.tune.enabled:
        model = tune_cfg(cfg, train_set, val_set)
    else:
        # model = train_dlmi(None, cfg, complete_train_set, train_set, val_set)
        model = train_dlmi(None, cfg, complete_train_set, train_set, val_set)


    # save_model(working_dir + "/checkpoint.pt", model)
    # logging.info(f"Checkpoint saved at {working_dir}")

    gc.collect()
    model = model.to(device)
    complete_test_set = MILDataset(train_set_path, split="test")
    test_dataset = DataLoader(complete_test_set, 1, shuffle=False, num_workers=0)
    # device = "cpu"
    run_infer(test_dataset, complete_test_set, model, "test", device, None)

    val_dataset   = DataLoader(val_set,  cfg.train.batch_size, shuffle=False, num_workers=0)
    run_infer(val_dataset, complete_train_set, model, "val", device, (~mask_train))

def update_config(cfg, config):
    cfg.train.batch_size = config["batch_size"]
    cfg.train.lr = config["lr"]
    cfg.train.weight_decay = config["weight_decay"]
    cfg.train.num_epochs = config["num_epochs"]
    cfg.exp.seed = config["seed"]
    return cfg

def train_dlmi(config, cfg, complete_train_set, train_set, val_set):
    metrics = ["val_negacc"]
    tune_callback = TuneReportCheckpointCallback(metrics, on="validation_end")
    # mlflow.pytorch.autolog()
    if config is not None:
        cfg = update_config(cfg, config)
    with mlflow.start_run():
        log_params_from_omegaconf_dict(cfg) 

        pl.seed_everything(cfg.exp.seed, workers=True)    

        model = hydra.utils.instantiate(cfg.model, cfg)

        batch_size = cfg.train.batch_size
        print("Batch size: ", batch_size)

        # basic_model = BasicModel(model, criterion, optimizer)
        # basic_model   = MILModel(model, criterion, optimizer)
        # train_dataset = DataLoader(train_set, batch_size, shuffle=True, num_workers=0)
        # val_dataset   = DataLoader(val_set,   batch_size, shuffle=False, num_workers=0)

        if cfg.dataset_type == "MiniDataset":
            s_train = Sampler(complete_train_set.train_classes, class_per_batch=1, batch_size=batch_size)
            s_val = Sampler(complete_train_set.test_classes, class_per_batch=1, batch_size=batch_size)
            train_dataset = DataLoader(complete_train_set, batch_sampler=s_train)
            val_dataset = DataLoader(complete_train_set, batch_sampler=s_val)
        elif cfg.dataset_type == "MILDataset":
            train_dataset = DataLoader(train_set, batch_size, shuffle=True, num_workers=0)
            val_dataset   = DataLoader(val_set,   batch_size, shuffle=False, num_workers=0)
        checkpoint_callback = ModelCheckpoint(monitor="val_negacc")

        trainer = pl.Trainer(
            max_epochs=cfg.train.num_epochs,
            accelerator="gpu",
            precision=16,
            log_every_n_steps=10,
            callbacks=[checkpoint_callback, tune_callback],
            deterministic=True
        )
        trainer.fit(model, train_dataset, val_dataset)

        # model.load_state_dict(torch.load(checkpoint_callback.best_model_path)["state_dict"])

    if config is None:
        return model


def tune_cfg(cfg, train_set, val_set):
    config = {
        # "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([1, 5]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-4, 1e-1),
        "num_epochs": tune.choice([10, 20, 30, 40]),
        "seed": tune.choice([42, 43, 44, 45, 46, 47, 48, 49, 50]),
    }

    trainable = tune.with_parameters(
        train_dlmi,
        cfg=cfg,
        train_set=train_set,
        val_set=val_set,
    )

    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"cpu": 1, "gpu": 1}),
        tune_config=tune.TuneConfig(
            metric="val_negacc",
            mode="min",
            num_samples=cfg.exp.tune.num_samples,
        ),
        run_config=train.RunConfig(
            name="tune_dlmi",
        ),
        param_space=config,

    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config, results.get_best_result())

    cfg = update_config(cfg, results.get_best_result().config)
    model = hydra.utils.instantiate(cfg.model, cfg)
    model.load_state_dict(torch.load(results.get_best_result().get_best_checkpoint("val_negacc", "min").path + "/checkpoint")["state_dict"])
    return model

    
# https://medium.com/@soumensardarintmain/manage-cuda-cores-ultimate-memory-management-strategy-with-pytorch-2bed30cab1#:~:text=The%20recommended%20way%20is%20to,first%20and%20then%20call%20torch.
def run_infer(dataset, main_dataset, model, name, device, mask=None):
    preds = []
    # model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataset, desc=f"Predicting {name} set"):
            torch.cuda.empty_cache()
            images, labels = batch
            # print(images[0].shape)
            # images[1] = images[1].to(device)
            # print(images[1].shape)
            # y_pre = model(images[1])
            y_pre = model.infer(images[0].unsqueeze(1).to(device), images[1].to(device), images[0].shape[0])
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
