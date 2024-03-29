import subprocess

import gc
import os

import numpy as np
import pandas as pd
import glob as glob

import hydra
import mlflow
import torch
from dlmi.data.data import load_dataset, load_kfolds
from dlmi.data.MILDataset import MILDataset
from dlmi.data.MiniDataset import MiniDataset, Sampler
from dlmi.utils.mlflow import log_params_from_omegaconf_dict
from dlmi.utils.utils import set_seed
from dlmi.data.MiniDataset import MiniDataset, Sampler

from hydra import utils
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint

import ray.tune as tune
import ray.train as train
from ray.air.integrations.mlflow import setup_mlflow
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler

from tqdm import tqdm

import random

def run_local_mlflow():
    """ Run a local mlflow server """
    subprocess.run('mlflow server --host 127.0.0.1 --port 5001'.split())

    
@hydra.main(config_path="dlmi/conf", config_name="train_moe", version_base="1.1")
def launch(cfg: DictConfig):
    """ Launch the training of the model, main function """
    set_seed(cfg.exp.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"    

    working_dir = os.getcwd()
    train_set_path = utils.get_original_cwd() + cfg.exp.data_path
    print("Working dir: ", working_dir, "Data path: ", train_set_path, "Device: ", device)
    
    if cfg.dataset_type == "MiniDataset":
        complete_train_set = MiniDataset(train_set_path, device)
        print("Train set loaded")
        train_set, val_set, mask_train = None, None, None
    elif cfg.dataset_type == "MILDataset":
        complete_train_set = MILDataset(train_set_path, device)
        print("Train set loaded")
        sets = load_kfolds(complete_train_set)
    else:
        raise ValueError("Dataset type not supported, must be MiniDataset or MILDataset")


    mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")
    mlflow.set_experiment(cfg.mlflow.runname)

    val_accs = []
    for i, one_set in enumerate(sets):
        train_set = one_set[0]
        val_set = one_set[1]
        mask_train = np.array([True if i in train_set.indices else False for i in range(len(complete_train_set.patients))])

        print(
            f"Training fold {i + 1}/{len(sets)} with {len(train_set)} samples and validating with {len(val_set)} samples."
        )

        if "tune" in cfg.exp.keys() and cfg.exp.tune.enabled:
            model = tune_cfg(cfg, complete_train_set, train_set, val_set)
        else:
            model = train_dlmi(None, cfg, complete_train_set, train_set, val_set)

        gc.collect()
        model = model.to(device)
        complete_test_set = MILDataset(train_set_path, split="test", device=device)
        test_dataset = DataLoader(complete_test_set, 1, shuffle=False, num_workers=0)

        run_infer(test_dataset, complete_test_set, model, "test", device, None, i)

        batch_size = cfg.train.batch_size
        val_dataset   = DataLoader(val_set,   batch_size, shuffle=False, num_workers=0)
        bac = run_infer(val_dataset, complete_train_set, model, "val", device, (~mask_train), i)
        val_accs.append(bac)

    print("Validation accuracies: ", val_accs)
    print("Mean validation accuracy: ", np.mean(val_accs))
    print("Std validation accuracy: ", np.std(val_accs))

    np.savetxt("val_perf.txt", val_accs)
    np.savetxt("val_stats.txt", np.array([np.mean(val_accs), np.std(val_accs)]))

    submit_final()


def submit_final():
    """ merge the predictions of the different folds for the test set by majority voting """
    dfs = []
    for file in glob.glob('submission_test_*.csv'):
        print("Reading file: ", file)
        df = pd.read_csv(file)
        dfs.append(df)
    
    final_df = pd.DataFrame(columns=['Id','Predicted'])
    for i in range(len(dfs[0])):
        zero = 0
        one = 0
        for j in range(len(dfs)):
            if dfs[j].iloc[i,1] == 0:
                zero += 1
            else:
                one += 1
        print(dfs[0].iloc[i,0], zero, one)
        if zero > one:
            final_df.loc[len(final_df)] = [dfs[0].iloc[i,0],0]
        else:
            final_df.loc[len(final_df)] = [dfs[0].iloc[i,0],1]
    final_df.to_csv('submission_test_final.csv', index=False)


def update_config(cfg, config):
    cfg.train.batch_size = config["batch_size"]
    # cfg.train.lr = config["lr"]
    # cfg.train.weight_decay = config["weight_decay"]
    # cfg.train.num_epochs = config["num_epochs"]
    # cfg.exp.seed = config["seed"]
    return cfg


def train_dlmi(config, cfg, complete_train_set, train_set, val_set):
    """ Train the model """
    metrics = ["val_negacc"]
    tune_callback = TuneReportCheckpointCallback(metrics, on="validation_end")
    if config is not None:
        cfg = update_config(cfg, config)
    with mlflow.start_run():
        log_params_from_omegaconf_dict(cfg) 

        pl.seed_everything(cfg.exp.seed, workers=True)    

        model = hydra.utils.instantiate(cfg.model, cfg)

        batch_size = cfg.train.batch_size
        print("Batch size: ", batch_size)

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

        model.load_state_dict(torch.load(checkpoint_callback.best_model_path)["state_dict"])

    if config is None:
        return model


def tune_cfg(cfg, complete_train_set, train_set, val_set):
    """ Tune the hyperparameters of the model with ray """
    config = {
        "batch_size": tune.choice([1, 5]),
        # "lr": tune.loguniform(1e-4, 1e-1),
        # "weight_decay": tune.loguniform(1e-4, 1e-1),
        # "num_epochs": tune.choice([10, 20, 30, 40]),
        # "seed": tune.choice([42, 43, 44, 45, 46, 47, 48, 49, 50]),
    }

    trainable = tune.with_parameters(
        train_dlmi,
        cfg=cfg,
        complete_train_set=complete_train_set,
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

    
def run_infer(dataset, main_dataset, model, name, device, mask=None, fold=0):
    """ do inference on the test set """
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataset, desc=f"Predicting {name} set"):
            torch.cuda.empty_cache()
            images, labels = batch

            y_pre = model.predict(images[0].unsqueeze(1).to(device), images[1].to(device), images[0].shape[0])

            selected_class = torch.argmax(y_pre, dim=1)
            preds.append(selected_class)
        
        preds = torch.cat(preds)
        bac = main_dataset.get_patient_labels(preds, mask, name, fold)

        return bac



if __name__ == "__main__":
    launch()
