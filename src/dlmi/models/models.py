import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        # We allocate space for the weights
        self.l1 = nn.Linear(input_size, 100)
        self.l2 = nn.Linear(100, output_size)
        # Input size is 16*16, output size should be the same with the number of classes

    def forward(self, inputs):  # Called when we apply the network
        h = F.relu(
            self.l1(inputs)
        )  # You can put anything, as long as its Pytorch functions
        outputs = F.softmax(
            self.l2(h), dim=1
        )  # Use softmax as the activation function for the last layer
        return outputs

class LeNet(nn.Module):
    def __init__(self, img_size=224):
        super(LeNet, self).__init__()
        self.img_size = img_size
        self.conv1 = nn.Conv2d(3, 6, 5, padding=1)
        self.pool  = nn.MaxPool2d(2, 2, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=1)
        self.fc1   = nn.Linear(16 * img_size//4 * img_size//4, 120)  
        self.fc2   = nn.Linear(120, 84)  
        self.fc3   = nn.Linear(84, 1) 

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        # print(x.shape)
        x = x.reshape(-1, 16 * self.img_size//4 * self.img_size//4)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  
        return x



def load_model(path_checkpoint, modelClass: torch.nn.Module, **kwargs):
    model = modelClass(**kwargs)
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model


def save_model(path_checkpoint, model):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        path_checkpoint,
    )
