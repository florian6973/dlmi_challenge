import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics
import mlflow
import torch


class BasicModel(pl.LightningModule):
    def __init__(self, model, criterion, optimizer):
        super().__init__()
        self.model     = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_steps_output = []
        self.train_acc_output   = []
        self.val_steps_output   = []
        self.val_acc_output     = []

    def compute_forward_loss(self, images, labels):
        images = images.to("cuda")
        labels = labels.to("cuda")
        
        # forward pass
        y_pre = self.model(images)        

        # print(self.criterion)
        # print(y_pre)
        # print(labels)
        loss = self.criterion(y_pre, labels)
        return loss, y_pre

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        loss, y_pre = self.compute_forward_loss(*batch)

        self.train_steps_output.append(loss.item())
        self.train_acc_output.append(
            [y_pre, batch[1]]
        )
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch[1]
        loss, y_pre = self.compute_forward_loss(*batch)
        # measure balanced accuracy
        
        self.val_acc_output.append(
            [y_pre, labels]
        )

        self.val_steps_output.append(loss.item())
        return loss

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
        return self.optimizer

class MILModel(pl.LightningModule):
    def __init__(self, base_model, criterion, optimizer):
        super().__init__()

        self.base_model = base_model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.base_model = base_model

        self.fc = nn.Linear(base_model.resnet.fc.in_features, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.base_model(x)
        pooled_features = torch.mean(features, dim=0, keepdim=True)
        logits = self.fc(pooled_features)
        return self.sigmoid(logits)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, y.unsqueeze(1).float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, y.unsqueeze(1).float())
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return self.optimizer
    
# def train(num_epochs, batch_size, criterion, optimizer, model, dataset):
#     training_mo
    # train_error = []
    # train_loader = DataLoader(dataset, batch_size, shuffle=True)
    # model = model.to("cuda")
    # model.train()  # Indicates to the network we are in training mode
    # for epoch in range(num_epochs):
    #     epoch_average_loss = 0.0
    #     for images, labels in train_loader:
    #         # y_pre = model(images.view(batch_size, -1))
    #         # reshape the inputs from [N, img_shape, img_shape] to [N, img_shape*img_shape]
    #         images = images.to("cuda")
    #         labels = labels.to("cuda")
    #         y_pre = model(images)

    #         # One-hot encoding or labels so as to calculate MSE error:
    #         # labels_one_hot = torch.FloatTensor(batch_size, 10)
    #         # labels_one_hot.zero_()
    #         # labels_one_hot.scatter_(1, labels.view(-1, 1), 1)

    #         # loss = criterion(y_pre, labels_one_hot)  # Real number
    #         # print(y_pre)
    #         # print(y_pre.shape)
    #         # print(labels.shape)
    #         # print(labels)
    #         # loss = criterion(y_pre, labels)

    #         # MSE LOSS
    #         loss = criterion(y_pre, labels.float().view(-1, 1))

    #         optimizer.zero_grad()  # Set all the parameters gradient to 0
    #         loss.backward()  # Computes  dloss/da for every parameter a which has requires_grad=True
    #         optimizer.step()  # Updates the weights
    #         epoch_average_loss += loss.item() * batch_size / len(dataset)
    #     train_error.append(epoch_average_loss)
    #     # log the training error to mlflow
    #     mlflow.log_metric("train_error", epoch_average_loss, step=epoch)
    #     logging.info(
    #         "Epoch [{}/{}], Loss: {:.4f}".format(
    #             epoch + 1, num_epochs, epoch_average_loss
    #         )
    #     )
    # return train_error
