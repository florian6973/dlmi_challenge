import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics
import mlflow
import torch
import hydra.utils
import torchvision

import torchvision.transforms as transforms    

class MOEModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.automatic_optimization = False
        
        self.cfg = cfg

        self.cnn = torchvision.models.resnet34(weights="DEFAULT")
        self.cnn.fc = nn.Identity()

        if cfg.get('train', {}).get('freeze_cnn', False):
            for param in self.cnn.parameters():
                param.requires_grad = False
        

        self.classifier_cnn = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2), 
            nn.LogSoftmax(dim=1)
        )

        # self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 2)

        # for param in self.cnn.fc.parameters():
        #     param.requires_grad = True

        # self.cnn = nn.Sequential(
        #     nn.Conv2d(3, 6, 5, padding=1),
        #     nn.MaxPool2d(2, 2, padding=1),
        #     nn.Conv2d(6, 16, 5, padding=1),
        #     nn.MaxPool2d(2, 2, padding=1),
        #     nn.Flatten(),
        #     nn.Linear(16 * 224//4 * 224//4, 512),
        #     nn.ReLU(),
        #     # nn.Linear(120, 84),
        #     # nn.ReLU(),
        #     # nn.Linear(84, 2),
        #     # nn.LogSoftmax(dim=1)
        # )

        # self.final_cnn = nn.Sequential(nn.Linear(512, 2), nn.LogSoftmax(dim=1))

        self.mlp = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            # nn.LogSoftmax(dim=1)
            nn.LogSoftmax(dim=1)
        )

        self.criterion = hydra.utils.instantiate(cfg.criterion)
        self.cfg = cfg

        self.train_steps_output = []
        self.train_steps_output_mlp = []
        self.train_acc_output   = []
        self.val_steps_output   = []
        self.val_steps_output_mlp   = []
        self.val_acc_output     = []

        # Define the augmentation pipeline
        # self.transform = transforms.Compose([
        #     transforms.RandomRotation(90),  # Rotate the image by a random angle up to 30 degrees
        #     transforms.ColorJitter(brightness=0.3) # Change the brightness by a random factor up to 0.3
        # ])
        
        self.transform = transforms.Compose([
            transforms.RandomRotation(180),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # self.transform_normalize = transforms.Compose([

        #     transforms.Normalize([209.15147887/255,  178.78958125/255,  179.65400146/255], [1.,1.,1.])
        # ])

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.trainer.max_epochs - 1:
            pass
            # Workaround to always save the last epoch until the bug is fixed in lightning (https://github.com/Lightning-AI/lightning/issues/4539)
            # self.trainer.check_val_every_n_epoch = 1

            # Disable backward pass for SWA until the bug is fixed in lightning (https://github.com/Lightning-AI/lightning/issues/17245)
            # self.automatic_optimization = False

    def forward(self, x_cnn, x_mlp):        
        return self.cnn(x_cnn), self.mlp(x_mlp)

    def training_step(self, batch, batch_idx):   
        torch.cuda.empty_cache()
        all_features, labels = batch
        mlp_features = all_features[1]
        cnn_features = all_features[0]

        cnn_outputs, mlp_outputs = self.infer(cnn_features, mlp_features, cnn_features.shape[0], True)
        loss_cnn = self.criterion(cnn_outputs, labels)
        loss_mlp = self.criterion(mlp_outputs, labels)
        
        self.update_optimizers(loss_cnn, loss_mlp)

        self.train_steps_output.append(loss_cnn.detach().item())
        self.train_steps_output_mlp.append(loss_mlp.detach().item())
        self.train_acc_output.append(
            [cnn_outputs.detach(), mlp_outputs.detach(), labels.detach()]
        )

    def infer(self, cnn_features, mlp_features, batch_size, augment=False):        
        # print(cnn_features.shape)
        # print(cnn_features)
        cnn_features = cnn_features.view(-1, cnn_features.shape[-3], cnn_features.shape[-2], cnn_features.shape[-1])
        indices = torch.where(torch.sum(cnn_features, dim=(1, 2, 3)).long() == 0)[0]
        # print(indices)
        mask = torch.ones(cnn_features.shape[0], dtype=torch.bool)
        mask[indices] = False
        # print(mask)
        cnn_features = cnn_features[mask]

        augment = False
        if augment:
            cnn_features = self.transform(cnn_features)
        
        # print(cnn_features.shape)
        # exit()

        pred_cnn, pred_mlp = self(cnn_features, mlp_features)
        # print(pred_cnn.shape, pred_mlp.shape)
        # print(pred_cnn, pred_mlp)

        # check if nan
        if torch.isnan(pred_cnn).any():
            print("Nan in pred_cnn")
            print(pred_cnn)
            exit()
        if torch.isnan(pred_mlp).any():
            print("Nan in pred_mlp")
            print(pred_mlp)
            exit()

        means = torch.zeros((batch_size, pred_cnn.shape[1]))

        last_label = 0
        i_means = 0
        idx_pred_cnn = 0
        for i in range(indices.shape[0]):
            current_label = indices[i]
            if current_label - last_label > 1:
                diff = current_label - last_label
                # print("average from", last_label, "to", current_label, "diff", diff, "idx_pred_cnn", idx_pred_cnn)
                means[i_means] = torch.mean(pred_cnn[idx_pred_cnn:idx_pred_cnn+diff], dim=0)
                i_means += 1
                idx_pred_cnn += diff

            last_label = current_label

        # print(means.shape)
        # print(means)
        # print(torch.isnan(means).any()) 


        pred_cnn = means.to(device=cnn_features.device, dtype=cnn_features.dtype)
        pred_cnn = self.classifier_cnn(pred_cnn)

        if torch.isnan(pred_cnn).any():
            print("Nan in pred_cnn")
            print(pred_cnn)
            exit()

        # exit()

        return pred_cnn, pred_mlp

    def validation_step(self, batch, batch_idx):      
        all_features, labels = batch
        mlp_features = all_features[1]
        cnn_features = all_features[0]

        pred_cnn, pred_mlp = self.infer(cnn_features, mlp_features, cnn_features.shape[0])

        loss_cnn = self.criterion(pred_cnn, labels)
        loss_mlp = self.criterion(pred_mlp, labels)

        self.val_acc_output.append(
            [pred_cnn.detach(), pred_mlp.detach(), labels.detach()]
        )

        self.val_steps_output.append(loss_cnn.item())
        self.val_steps_output_mlp.append(loss_mlp.item())

    def stats(self, preds_list, labels_list, cat="val"):
        pass

    def on_train_epoch_end(self):
        y_pre_cnn  = torch.cat([x[0] for x in self.train_acc_output])
        y_pre_mlp  = torch.cat([x[1] for x in self.train_acc_output])
        labels_all = torch.cat([x[2] for x in self.train_acc_output])

        acc_cnn = torchmetrics.functional.classification.accuracy(
            y_pre_cnn, labels_all, task='multiclass', num_classes=2, average='macro'
        )

        acc_mlp = torchmetrics.functional.classification.accuracy(
            y_pre_mlp, labels_all, task='multiclass', num_classes=2, average='macro'
        )
        # print(acc)
        mlflow.log_metric("train_acc", acc_cnn, step=self.current_epoch)
        self.train_acc_output = []
        # log the training error to mlflow
        train_error = sum(self.train_steps_output) / len(self.train_steps_output)
        train_error_mlp = sum(self.train_steps_output_mlp) / len(self.train_steps_output_mlp)
        mlflow.log_metric("train_error", train_error, step=self.current_epoch)
        print(f"\nEpoch {self.current_epoch} train_error: {train_error:5g} - train_error_mlp: {train_error_mlp:5g} - train_acc_cnn: {acc_cnn:5g} - train_acc_mlp: {acc_mlp:5g}")
        self.train_steps_output = []

    def on_validation_epoch_end(self):
        # log the validation error to mlflow
        y_pre_cnn  = torch.cat([x[0] for x in self.val_acc_output])
        y_pre_mlp  = torch.cat([x[1] for x in self.val_acc_output])
        labels_all = torch.cat([x[2] for x in self.val_acc_output])
        acc_cnn = torchmetrics.functional.classification.accuracy(
            y_pre_cnn, labels_all, task='multiclass', num_classes=2, average='macro'
        )
        acc_mlp = torchmetrics.functional.classification.accuracy(
            y_pre_mlp, labels_all, task='multiclass', num_classes=2, average='macro'
        )
        # print(acc)
        mlflow.log_metric("val_acc", acc_cnn, step=self.current_epoch)
        self.val_acc_output = []

        val_error = sum(self.val_steps_output) / len(self.val_steps_output)
        val_error_mlp = sum(self.val_steps_output_mlp) / len(self.val_steps_output_mlp)
        self.log("val_negacc", -acc_cnn)
        mlflow.log_metric("val_error", val_error, step=self.current_epoch)
        print(f"\nEpoch {self.current_epoch} val_error: {val_error:5g} - val_error_mlp: {val_error_mlp:5g} - val_acc_cnn: {acc_cnn:5g} - val_acc_mlp: {acc_mlp:5g}")
        self.val_steps_output = []
    
    def update_optimizers(self, loss_cnn, loss_mlp):
        optimizer_cnn, optimizer_mlp = self.optimizers()

        optimizer_cnn.zero_grad()
        self.manual_backward(loss_cnn)
        optimizer_cnn.step()
        
        optimizer_mlp.zero_grad()
        self.manual_backward(loss_mlp)
        optimizer_mlp.step()

    def configure_optimizers(self):

        # if not self.cfg.train.freeze_cnn:
        #     optimizer_cnn = hydra.utils.instantiate(
        #         self.cfg.optimizer,
        #         *[self.cnn.parameters()], 
        #         **{"lr":self.cfg.train.lr}
        #     )

        optimizer_cnn = hydra.utils.instantiate(
            self.cfg.optimizer,
            *[self.classifier_cnn.parameters()], 
            **{"lr":1}#self.cfg.train.lr}
        )


        optimizer_mlp = hydra.utils.instantiate(
            self.cfg.optimizer,
            *[self.mlp.parameters()], 
            **{"lr":0.01}#self.cfg.train.lr}
        )

        return [optimizer_cnn, optimizer_mlp]