import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics
import mlflow
import torch


class MLPModel(pl.LightningModule):
    def __init__(self, cfg, criterion, optimizer):
        super().__init__()
        self.cfg = cfg
        self.model     = nn.Sequential(
            nn.Linear(3*224*224, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Softmax(dim=1)
        )
        self.criterion = criterion

        self.train_steps_output = []
        self.train_acc_output   = []
        self.val_steps_output   = []
        self.val_acc_output     = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        _, mlp_features, labels = batch
        preds = self.forward(mlp_features)
        loss = self.criterion(preds, labels)

        self.train_steps_output.append(loss.item())
        self.train_acc_output.append(
            [y_pre, labels]
        )
        return loss

    def validation_step(self, batch, batch_idx):        
        _, mlp_features, labels = batch
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
        # print(acc)
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
        # print(acc)
        mlflow.log_metric("val_acc", acc, step=self.current_epoch)
        self.val_acc_output = []

        val_error = sum(self.val_steps_output) / len(self.val_steps_output)
        mlflow.log_metric("val_error", val_error, step=self.current_epoch)
        print(f"Epoch {self.current_epoch} val_error: {val_error} - val_acc: {acc}")
        self.val_steps_output = []

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer