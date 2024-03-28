import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics
import mlflow
import torch
import hydra.utils
import torchvision

import torchvision.transforms as transforms    
from torch.optim.lr_scheduler import StepLR

class MOEModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.automatic_optimization = False
        
        self.cfg = cfg

        self.cnn = torchvision.models.resnet34(weights="DEFAULT")
        # self.cnn = torchvision.models.vit_b_16(weights="DEFAULT")

        if cfg.get('train', {}).get('freeze_cnn', False):
            for param in self.cnn.parameters():
                param.requires_grad = False

        self.cnn.fc = nn.Sequential(
        #self.cnn.heads.head = nn.Sequential(
            # nn.Linear(self.cnn.heads.head.in_features, 512),
            nn.Linear(self.cnn.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.LogSoftmax(dim=1)
        )
       
        self.final_classifier = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            nn.LogSoftmax(dim=1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            nn.LogSoftmax(dim=1)
        )

        self.criterion = hydra.utils.instantiate(cfg.criterion)
        self.cfg = cfg

        self.train_steps_output = []
        self.train_steps_output_mlp = []
        self.train_steps_output_total = []
        self.train_acc_output   = []
        self.val_steps_output   = []
        self.val_steps_output_mlp   = []
        self.val_steps_output_total   = []
        self.val_acc_output     = []
        
        self.transform = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.trainer.max_epochs - 1:
            pass
            # Workaround to always save the last epoch until the bug is fixed in lightning (https://github.com/Lightning-AI/lightning/issues/4539)
            # self.trainer.check_val_every_n_epoch = 1

            # Disable backward pass for SWA until the bug is fixed in lightning (https://github.com/Lightning-AI/lightning/issues/17245)
            # self.automatic_optimization = False

    def on_val_epoch_start(self) -> None:
        torch.cuda.empty_cache()

    def forward(self, x_cnn, x_mlp):        
        return self.cnn(x_cnn), self.mlp(x_mlp)

    def predict(self, cnn_features, mlp_features, batch_size, augment=False):
        return self.infer(cnn_features, mlp_features, batch_size, augment)[2]

    def training_step(self, batch, batch_idx):   
        torch.cuda.empty_cache()
        all_features, labels = batch
        mlp_features = all_features[1]
        cnn_features = all_features[0]

        cnn_outputs, mlp_outputs, final_outputs = self.infer(cnn_features, mlp_features, cnn_features.shape[0], True)
        loss_cnn = self.criterion(cnn_outputs, labels)
        loss_mlp = self.criterion(mlp_outputs, labels)
        loss_final = self.criterion(final_outputs, labels)
        
        self.update_optimizers(loss_cnn, loss_mlp, loss_final)

        self.train_steps_output.append(loss_cnn.detach().item())
        self.train_steps_output_mlp.append(loss_mlp.detach().item())
        self.train_steps_output_total.append(loss_final.detach().item())
        self.train_acc_output.append(
            [cnn_outputs.detach(), mlp_outputs.detach(), final_outputs.detach(), labels.detach()]
        )

    def update_lr(self):
        scheduler_cnn, scheduler_mlp, scheduler_total = self.lr_schedulers()

        scheduler_cnn.step()
        scheduler_mlp.step()
        scheduler_total.step()

    def infer(self, cnn_features, mlp_features, batch_size, augment=False):
        cnn_features = cnn_features.view(-1, cnn_features.shape[-3], cnn_features.shape[-2], cnn_features.shape[-1])
        indices = torch.where(torch.sum(cnn_features, dim=(1, 2, 3)).long() == 0)[0]
        mask = torch.ones(cnn_features.shape[0], dtype=torch.bool)
        mask[indices] = False
        cnn_features = cnn_features[mask]

        if self.cfg.get('train', {}).get('augment', False):
            cnn_features = self.transform(cnn_features)

        pred_cnn, pred_mlp = self(cnn_features, mlp_features)

        if torch.isnan(pred_cnn).any():
            print("Nan in pred_cnn")
            print(pred_cnn)
            exit()
        if torch.isnan(pred_mlp).any():
            print("Nan in pred_mlp")
            print(pred_mlp)
            exit()

        aggreg = []

        last_label = 0
        i_aggreg = 0
        idx_pred_cnn = 0
        # replace this with RNN? that can learn an arbitrary number of vectors
        for i in range(indices.shape[0]):
            current_label = indices[i]
            if current_label - last_label > 1:
                diff = current_label - last_label
                aggreg.append(torch.mean(pred_cnn[idx_pred_cnn:idx_pred_cnn+diff], dim=0))
                i_aggreg += 1
                idx_pred_cnn += diff

            last_label = current_label

        aggreg = torch.stack(aggreg)

        pred_cnn = aggreg.to(device=cnn_features.device, dtype=cnn_features.dtype)

        if torch.isnan(pred_cnn).any():
            print("Nan in pred_cnn")
            print(pred_cnn)
            exit()

        pred_final = self.final_classifier(torch.cat([pred_cnn.clone().detach(), pred_mlp.clone().detach()], dim=1).float())
        
        return pred_cnn, pred_mlp, pred_final

    def validation_step(self, batch, batch_idx):      
        all_features, labels = batch
        mlp_features = all_features[1]
        cnn_features = all_features[0]

        pred_cnn, pred_mlp, pred_total = self.infer(cnn_features, mlp_features, cnn_features.shape[0])

        loss_cnn = self.criterion(pred_cnn, labels)
        loss_mlp = self.criterion(pred_mlp, labels)
        loss_final = self.criterion(pred_total, labels)

        self.val_acc_output.append(
            [pred_cnn.detach(), pred_mlp.detach(), pred_total.detach(), labels.detach()]
        )

        self.val_steps_output.append(loss_cnn.item())
        self.val_steps_output_mlp.append(loss_mlp.item())
        self.val_steps_output_total.append(loss_final.item())

    def stats(self, preds_list, labels_list, cat="val"):
        pass

    def on_train_epoch_end(self):
        y_pre_cnn  = torch.cat([x[0] for x in self.train_acc_output])
        y_pre_mlp  = torch.cat([x[1] for x in self.train_acc_output])
        y_pre_total = torch.cat([x[2] for x in self.train_acc_output])
        labels_all = torch.cat([x[3] for x in self.train_acc_output])

        acc_cnn = torchmetrics.functional.classification.accuracy(
            y_pre_cnn, labels_all, task='multiclass', num_classes=2, average='macro'
        )

        acc_mlp = torchmetrics.functional.classification.accuracy(
            y_pre_mlp, labels_all, task='multiclass', num_classes=2, average='macro'
        )

        acc_final = torchmetrics.functional.classification.accuracy(
            y_pre_total, labels_all, task='multiclass', num_classes=2, average='macro'
        )

        mlflow.log_metric("train_acc", acc_final, step=self.current_epoch)
        self.train_acc_output = []
        train_error = sum(self.train_steps_output) / len(self.train_steps_output)
        train_error_mlp = sum(self.train_steps_output_mlp) / len(self.train_steps_output_mlp)
        train_error_final = sum(self.train_steps_output_total) / len(self.train_steps_output_total)
        mlflow.log_metric("train_error", train_error_final, step=self.current_epoch)
        print(f"\nEpoch {self.current_epoch} train_error_cnn: {train_error:5g} - train_error_mlp: {train_error_mlp:5g} - train_error_final: {train_error_final:5g} - train_acc_cnn: {acc_cnn:5g} - train_acc_mlp: {acc_mlp:5g} - train_acc_final: {acc_final:5g}")
        self.train_steps_output = []

        self.update_lr()

    def on_validation_epoch_end(self):
        # log the validation error to mlflow
        y_pre_cnn  = torch.cat([x[0] for x in self.val_acc_output])
        y_pre_mlp  = torch.cat([x[1] for x in self.val_acc_output])
        y_pre_total = torch.cat([x[2] for x in self.val_acc_output])
        labels_all = torch.cat([x[3] for x in self.val_acc_output])
        acc_cnn = torchmetrics.functional.classification.accuracy(
            y_pre_cnn, labels_all, task='multiclass', num_classes=2, average='macro'
        )
        acc_mlp = torchmetrics.functional.classification.accuracy(
            y_pre_mlp, labels_all, task='multiclass', num_classes=2, average='macro'
        )
        acc_final = torchmetrics.functional.classification.accuracy(
            y_pre_total, labels_all, task='multiclass', num_classes=2, average='macro'
        )
        # print(acc)
        mlflow.log_metric("val_acc", acc_final, step=self.current_epoch)
        self.val_acc_output = []

        val_error = sum(self.val_steps_output) / len(self.val_steps_output)
        val_error_mlp = sum(self.val_steps_output_mlp) / len(self.val_steps_output_mlp)
        val_error_final = sum(self.val_steps_output_total) / len(self.val_steps_output_total)
        self.log("val_negacc", -acc_final)
        mlflow.log_metric("val_error", val_error_final, step=self.current_epoch)
        print(f"\nEpoch {self.current_epoch} val_error_cnn: {val_error:5g} - val_error_mlp: {val_error_mlp:5g} - val_error_final: {val_error_final:5g} - val_acc_cnn: {acc_cnn:5g} - val_acc_mlp: {acc_mlp:5g} - val_acc_final: {acc_final:5g}")
        self.val_steps_output = []
    
    def update_optimizers(self, loss_cnn, loss_mlp, loss_final):
        optimizer_cnn, optimizer_mlp, optimizer_total = self.optimizers()

        optimizer_cnn.zero_grad()
        self.manual_backward(loss_cnn)
        optimizer_cnn.step()
        
        optimizer_mlp.zero_grad()
        self.manual_backward(loss_mlp)
        optimizer_mlp.step()

        optimizer_total.zero_grad()
        self.manual_backward(loss_final)
        optimizer_total.step()

    def configure_optimizers(self):
        from itertools import chain
        optimizer_cnn = hydra.utils.instantiate(
            self.cfg.optimizer,
            *[self.cnn.parameters()],
            **{"lr":self.cfg.train.lr_cnn, "weight_decay":self.cfg.train.weight_decay}
        )

        optimizer_mlp = hydra.utils.instantiate(
            self.cfg.optimizer,
            *[self.mlp.parameters()], 
            **{"lr":self.cfg.train.lr_mlp, "weight_decay":self.cfg.train.weight_decay}
        )

        optimizer_final = hydra.utils.instantiate(
            self.cfg.optimizer,
            *[self.final_classifier.parameters()], 
            **{"lr":self.cfg.train.lr_final, "weight_decay":self.cfg.train.weight_decay}
        )

        scheduler_cnn = StepLR(optimizer_cnn, step_size=40, gamma=0.75)
        scheduler_mlp = StepLR(optimizer_mlp, step_size=40, gamma=0.75)
        scheduler_final = StepLR(optimizer_final, step_size=40, gamma=0.75)

        return [optimizer_cnn, optimizer_mlp, optimizer_final], [scheduler_cnn, scheduler_mlp, scheduler_final]

        # dual scheduler

    # mlflow https://d585504d-4e5f-4236-9b90-8213c9b53792.notebook.gra.ai.cloud.ovh.net/proxy/5001/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D