import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics
import mlflow
import torch
import hydra.utils
import torchvision

class CNNModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = torchvision.models.resnet18()
        # self.model.fc = nn.Linear(512, 2)
        self.model.fc = nn.Identity()
        self.final = nn.Sequential(nn.Linear(512*3, 2), nn.LogSoftmax(dim=1))
        # print(self.model)
        # exit()

        self.criterion = hydra.utils.instantiate(cfg.criterion)
        self.cfg = cfg

        self.train_steps_output = []
        self.train_acc_output   = []
        self.val_steps_output   = []
        self.val_acc_output     = []

    def forward(self, x):
        # print(x.shape)
        # print(x)
        return self.model(x)

    def training_step(self, batch, batch_idx):    
        # print("train", torch.cuda.memory_allocated())
        torch.cuda.empty_cache()
        all_features, labels = batch
        # mlp_features = all_features[1]
        cnn_features = all_features[0]
        # print(cnn_features.shape)
        # exit()
        # for i in range(len(cnn_features)):
        #     cnn_features[i] = self.forward(cnn_features[i])
        # preds = self.forward(mlp_features)
        # print(preds.shape)
        # print(preds)
        preds = self.infer(cnn_features, cnn_features.shape[0])
        loss = self.criterion(preds, labels)

        self.train_steps_output.append(loss.detach().item())
        # print(loss)
        self.train_acc_output.append(
            [preds.detach(), labels.detach()]
        )
        return loss

    def infer(self, cnn_features, batch_size):
        
        cnn_features = cnn_features.view(-1, cnn_features.shape[-3], cnn_features.shape[-2], cnn_features.shape[-1])
        # print(cnn_features.shape)
        # get indices where last three dimensions are zero
        indices = torch.where(torch.sum(cnn_features, dim=(1, 2, 3)) == 0)[0]
        # indices = torch.unique(indices)
        mask = torch.ones(cnn_features.shape[0], dtype=torch.bool)
        # print(indices)
        result = self.forward(cnn_features[mask])
        means = torch.zeros((batch_size, result.shape[1]))
        mins = torch.zeros((batch_size, result.shape[1]))
        maxs = torch.zeros((batch_size, result.shape[1]))
        # print(result.shape)
        last_label = 0
        i_means = 0
        for i in range(indices.shape[0]):
            current_label = indices[i]
            if current_label - last_label > 1:
                means[i_means] = torch.mean(result[last_label:current_label], dim=0)
                mins[i_means] = torch.min(result[last_label:current_label], dim=0).values
                maxs[i_means] = torch.max(result[last_label:current_label], dim=0).values
                # print(i_means, last_label, current_label)
                i_means += 1
            last_label = current_label
        # print(means.shape)
        # print(means)
        result = torch.concat([means, mins, maxs], dim=1).to(device=cnn_features.device, dtype=cnn_features.dtype)
        # print(result.shape)
        result_final = self.final(result)
        # print(result_final.shape)
        # print(result_final)

        return result_final

    def validation_step(self, batch, batch_idx):   
        # print("Val", torch.cuda.memory_allocated())      
        all_features, labels = batch
        # mlp_features = all_features[1]
        cnn_features = all_features[0]

        # merge first and second dimension
        
        # exit()
        # exit()

        preds = self.infer(cnn_features, cnn_features.shape[0])
        loss = self.criterion(preds, labels)
        
        self.val_acc_output.append(
            [preds.detach(), labels.detach()]
        )
        # print(loss)

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
        # optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # return optimizer
        # return self.optimizer
        optimizer = hydra.utils.instantiate(
                    self.cfg.optimizer,
                    *[self.model.parameters()], 
                    **{"lr":self.cfg.train.lr, "weight_decay":self.cfg.train.weight_decay}
                )
        return optimizer