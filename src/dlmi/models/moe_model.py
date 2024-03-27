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
            nn.Linear(512, 2), 
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
            nn.Linear(2, 10),
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

        # print(cnn_features.shape)
        # exit()
        # for i in range(len(cnn_features)):
        #     cnn_features[i] = self.forward(cnn_features[i])
        # preds = self.forward(mlp_features)
        # print(preds.shape)
        # print(preds)

        cnn_outputs, mlp_outputs = self.infer(cnn_features, mlp_features, cnn_features.shape[0], True)
        loss_cnn = self.criterion(cnn_outputs, labels)
        loss_mlp = self.criterion(mlp_outputs, labels)
        
        self.update_optimizers(loss_cnn, loss_mlp)

        # preds = self.infer(cnn_features, mlp_features, cnn_features.shape[0], True)
        # loss  = self.criterion(preds[0], labels)

        self.train_steps_output.append(loss_cnn.detach().item())
        # print(loss)
        self.train_acc_output.append(
            [cnn_outputs.detach(), mlp_outputs.detach(), labels.detach()]
        )
        return loss_cnn

    def infer(self, cnn_features, mlp_features, batch_size, augment=False):
        
        cnn_features = cnn_features.view(-1, cnn_features.shape[-3], cnn_features.shape[-2], cnn_features.shape[-1])
        # # print(cnn_features.shape)
        # # get indices where last three dimensions are zero
        indices = torch.where(torch.sum(cnn_features, dim=(1, 2, 3)) == 0)[0]
        # # indices = torch.unique(indices)
        mask = torch.ones(cnn_features.shape[0], dtype=torch.bool)


        # # Apply the transform to an image
        cnn_features = cnn_features[mask]
        # # features = self.transform_normalize(features)

        if augment:
            cnn_features = self.transform(cnn_features)

        # print(indices)
        pred_cnn, pred_mlp = self(cnn_features, mlp_features)
        means = torch.zeros((batch_size, pred_cnn.shape[1]))
        # mins = torch.zeros((batch_size, pred_cnn.shape[1]))
        # maxs = torch.zeros((batch_size, pred_cnn.shape[1]))
        # print(result.shape)

        last_label = 0
        i_means = 0
        for i in range(indices.shape[0]):
            current_label = indices[i]
            if current_label - last_label > 1:
                means[i_means] = torch.mean(pred_cnn[last_label:current_label], dim=0)
                # mins[i_means] = torch.min(pred_cnn[last_label:current_label], dim=0).values
                # maxs[i_means] = torch.max(pred_cnn[last_label:current_label], dim=0).values
                # print(i_means, last_label, current_label)
                i_means += 1
            last_label = current_label


        # print(means.shape)
        # print(means)
        # result = torch.concat([means, mins, maxs], dim=1).to(device=cnn_features.device, dtype=cnn_features.dtype)
        pred_cnn = means.to(device=cnn_features.device, dtype=cnn_features.dtype)
        # print(result.shape)
        pred_cnn = self.classifier_cnn(pred_cnn)

        # result_final = self.final_cnn(pred_cnn)
        pred_mlp = self.mlp(mlp_features)
        # mlp_preds = self.mlp(torch.concat([result_final, mlp_features], dim=1))
        # print(result_final.shape)
        # print(result_final)

        return pred_cnn, pred_mlp

    def validation_step(self, batch, batch_idx):   
        # print("Val", torch.cuda.memory_allocated())      
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

        return loss_cnn, loss_mlp

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
        mlflow.log_metric("train_error", train_error, step=self.current_epoch)
        print(f"\nEpoch {self.current_epoch} train_error: {train_error:5g} - train_acc_cnn: {acc_cnn:5g} - train_acc_mlp: {acc_mlp:5g}")
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
        self.log("val_negacc", -acc_cnn)
        mlflow.log_metric("val_error", val_error, step=self.current_epoch)
        print(f"\nEpoch {self.current_epoch} val_error: {val_error:5g} - val_acc_cnn: {acc_cnn:5g} - val_acc_mlp: {acc_mlp:5g}")
        self.val_steps_output = []
    
    def update_optimizers(self, loss_cnn, loss_mlp):
        
        self.optimizer_cnn.zero_grad()
        self.manual_backward(loss_cnn)
        self.optimizer_cnn.step()
        
        self.optimizer_mlp.zero_grad()
        self.manual_backward(loss_mlp)
        self.optimizer_mlp.step()

    def configure_optimizers(self):

        # if not self.cfg.train.freeze_cnn:
        #     optimizer_cnn = hydra.utils.instantiate(
        #         self.cfg.optimizer,
        #         *[self.cnn.parameters()], 
        #         **{"lr":self.cfg.train.lr}
        #     )

        self.optimizer_cnn = hydra.utils.instantiate(
            self.cfg.optimizer,
            *[self.classifier_cnn.parameters()], 
            **{"lr":self.cfg.train.lr}
        )


        self.optimizer_mlp = hydra.utils.instantiate(
            self.cfg.optimizer,
            *[self.mlp.parameters()], 
            **{"lr":self.cfg.train.lr}
        )

        return [self.optimizer_cnn, self.optimizer_mlp]