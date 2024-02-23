import logging
import os

import hydra
import mlflow
import torch
from dlmi.data.data import load_dataset
from dlmi.models.models import LinearModel, save_model
from dlmi.data.patientDataset import PatientDataset
from dlmi.utils.mlflow import log_params_from_omegaconf_dict
from hydra import utils
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def train(num_epochs, batch_size, criterion, optimizer, model, dataset):
    train_error = []
    train_loader = DataLoader(dataset, batch_size, shuffle=True)
    model = model.to("cuda")
    model.train()  # Indicates to the network we are in training mode
    for epoch in range(num_epochs):
        epoch_average_loss = 0.0
        for images, labels in train_loader:
            # y_pre = model(images.view(batch_size, -1))
            # reshape the inputs from [N, img_shape, img_shape] to [N, img_shape*img_shape]
            images = images.to("cuda")
            labels = labels.to("cuda")
            y_pre = model(images.permute(0, 3, 1, 2))

            # One-hot encoding or labels so as to calculate MSE error:
            # labels_one_hot = torch.FloatTensor(batch_size, 10)
            # labels_one_hot.zero_()
            # labels_one_hot.scatter_(1, labels.view(-1, 1), 1)

            # loss = criterion(y_pre, labels_one_hot)  # Real number
            # print(y_pre)
            # print(y_pre.shape)
            # print(labels.shape)
            # print(labels)
            # loss = criterion(y_pre, labels)

            # MSE LOSS
            loss = criterion(y_pre, labels.float().view(-1, 1))

            optimizer.zero_grad()  # Set all the parameters gradient to 0
            loss.backward()  # Computes  dloss/da for every parameter a which has requires_grad=True
            optimizer.step()  # Updates the weights
            epoch_average_loss += loss.item() * batch_size / len(dataset)
        train_error.append(epoch_average_loss)
        # log the training error to mlflow
        mlflow.log_metric("train_error", epoch_average_loss, step=epoch)
        logging.info(
            "Epoch [{}/{}], Loss: {:.4f}".format(
                epoch + 1, num_epochs, epoch_average_loss
            )
        )
    return train_error
