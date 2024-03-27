import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics
import mlflow
import torch
import hydra.utils
import torchvision
from collections import defaultdict

import torchvision.transforms as transforms    

class FinalModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.automatic_optimization = False
        
        self.cfg = cfg

        self.mlp = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            nn.LogSoftmax(dim=1)
        )

        self.criterion = hydra.utils.instantiate(cfg.criterion)

        self.outputs = {
            "train": {
                "output": defaultdict(list),
                "losses": defaultdict(list),
                "labels": list()
            },
            "val"  : {
                "output": defaultdict(list),
                "losses": defaultdict(list),
                "labels": list()
            }
        }

    def forward(self, x_mlp):
        
        return self.mlp(x_mlp)
    
    def training_step(self, batch, batch_idx):    
        
        torch.cuda.empty_cache()
        all_features, labels = batch
        mlp_features = all_features[1]
        cnn_features = all_features[0]

        pred_mlp = self.forward(mlp_features)
        loss_mlp = self.criterion(pred_mlp, labels)

        self.optimizer_mlp.zero_grad()
        self.manual_backward(loss_mlp)
        self.optimizer_mlp.step()

        self.outputs["train"]["losses"]["mlp"].append(loss_mlp.detach().item())
        self.outputs["train"]["output"]["mlp"].append(pred_mlp.detach())
        self.outputs["train"]["labels"].append(labels)

        return loss_mlp

    def validation_step(self, batch, batch_idx):   
  
        all_features, labels = batch
        mlp_features = all_features[1]
        cnn_features = all_features[0]

        pred_mlp = self(mlp_features)

        loss_mlp = self.criterion(pred_mlp, labels)

        self.outputs["val"]["losses"]["mlp"].append(loss_mlp.detach().item())
        self.outputs["val"]["output"]["mlp"].append(pred_mlp.detach())
        self.outputs["val"]["labels"].append(labels)

        return loss_mlp

    def on_train_epoch_end(self):
        y_pre_mlp  = torch.cat(self.outputs["train"]["output"]["mlp"])
        labels_all = torch.cat(self.outputs["train"]["labels"])

        acc_mlp = torchmetrics.functional.classification.accuracy(
            y_pre_mlp, labels_all, task='multiclass', num_classes=2, average='macro'
        )
        
        mlflow.log_metric("train_acc", acc_mlp, step=self.current_epoch)
        
        # log the training error to mlflow
        train_error = sum(self.outputs["train"]["losses"]["mlp"]) / len(self.outputs["train"]["losses"]["mlp"])
        mlflow.log_metric("train_error", train_error, step=self.current_epoch)
        print(f"\nEpoch {self.current_epoch} train_error: {train_error:5g} - train_acc_mlp: {acc_mlp:5g}")

        self.outputs["train"]["losses"]["mlp"] = []
        self.outputs["train"]["output"]["mlp"] = []
        self.outputs["train"]["labels"]        = []

    def on_validation_epoch_end(self):
        # log the validation error to mlflow
        y_pre_mlp  = torch.cat(self.outputs["val"]["output"]["mlp"])
        labels_all = torch.cat(self.outputs["val"]["labels"])

        acc_mlp = torchmetrics.functional.classification.accuracy(
            y_pre_mlp, labels_all, task='multiclass', num_classes=2, average='macro'
        )
        # print(acc)
        mlflow.log_metric("val_acc", acc_mlp, step=self.current_epoch)
        self.val_acc_output = []

        val_error = sum(self.outputs["val"]["losses"]["mlp"]) / len(self.outputs["val"]["losses"]["mlp"])
        self.log("val_negacc", -acc_mlp)
        mlflow.log_metric("val_error", val_error, step=self.current_epoch)
        print(f"\nEpoch {self.current_epoch} val_error: {val_error:5g} - val_acc_mlp: {acc_mlp:5g}")
        
        self.outputs["val"]["losses"]["mlp"] = []
        self.outputs["val"]["output"]["mlp"] = []
        self.outputs["val"]["labels"]        = []

    def configure_optimizers(self):

        self.optimizer_mlp = hydra.utils.instantiate(
            self.cfg.optimizer,
            *[self.mlp.parameters()], 
            **{"lr":self.cfg.train.lr}
        )

        return [self.optimizer_mlp]