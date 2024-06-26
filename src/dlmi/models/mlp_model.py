import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics
import mlflow
import torch
import hydra.utils

class MLPModel(pl.LightningModule):
    """ Simple MLP model for the DLMI challenge
    It only uses the clinical features to predict the target
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model     = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            nn.LogSoftmax(dim=1)
        )
        self.criterion = hydra.utils.instantiate(cfg.criterion)
        self.cfg = cfg

        self.train_steps_output = []
        self.train_acc_output   = []
        self.val_steps_output   = []
        self.val_acc_output     = []

    def forward(self, x):
        return self.model(x)

    def predict(self, cnn_features, mlp_features, batch_size, augment=False):
        return self.forward(mlp_features)

    def training_step(self, batch, batch_idx):     
        all_features, labels = batch
        mlp_features = all_features[1]
        preds = self.forward(mlp_features)
        loss = self.criterion(preds, labels)

        self.train_steps_output.append(loss.item())
        self.train_acc_output.append(
            [preds, labels]
        )
        return loss

    def validation_step(self, batch, batch_idx):        
        all_features, labels = batch
        mlp_features = all_features[1]
        preds = self.forward(mlp_features)
        loss = self.criterion(preds, labels)
        
        self.val_acc_output.append(
            [preds, labels]
        )

        self.val_steps_output.append(loss.item())
        return loss

    def stats(self, preds_list, labels_list, cat="val"):
        pass

    def on_train_epoch_end(self):
        y_pre_all  = torch.cat([x[0] for x in self.train_acc_output])
        labels_all = torch.cat([x[1] for x in self.train_acc_output])

        acc = torchmetrics.functional.classification.accuracy(
            y_pre_all, labels_all, task='multiclass', num_classes=2, average='macro'
        )

        mlflow.log_metric("train_acc", acc, step=self.current_epoch)
        self.train_acc_output = []
        # log the training error to mlflow
        train_error = sum(self.train_steps_output) / len(self.train_steps_output)
        mlflow.log_metric("train_error", train_error, step=self.current_epoch)
        print(f"Epoch {self.current_epoch} train_error: {train_error} - train_acc: {acc}")
        self.train_steps_output = []

    def on_validation_epoch_end(self):
        # log the validation error to mlflow
        y_pre_all = torch.cat([x[0] for x in self.val_acc_output])
        labels_all = torch.cat([x[1] for x in self.val_acc_output])
        acc = torchmetrics.functional.classification.accuracy(
            y_pre_all, labels_all, task='multiclass', num_classes=2, average='macro'
        )
        mlflow.log_metric("val_acc", acc, step=self.current_epoch)
        self.val_acc_output = []

        self.log("val_negacc", -acc)
        val_error = sum(self.val_steps_output) / len(self.val_steps_output)
        mlflow.log_metric("val_error", val_error, step=self.current_epoch)
        print(f"Epoch {self.current_epoch} val_error: {val_error} - val_acc: {acc}")
        self.val_steps_output = []

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
                    self.cfg.optimizer,
                    *[self.model.parameters()], 
                    **{"lr":self.cfg.train.lr, "weight_decay":self.cfg.train.weight_decay}
                )
        return optimizer