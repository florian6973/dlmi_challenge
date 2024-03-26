import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetworkAvgPool(nn.Module):
    
    def __init__(self, sz_conv, ch_conv, n_attr):
        super(GatingNetworkAvgPool, self).__init__()

        self.sz_conv                = sz_conv
        self.ch_conv                = ch_conv
        self.n_attr                 = n_attr
        self.pool                   = nn.AdaptiveAvgPool2d((1, 1))
        self.conv                   = nn.Conv2d(self.ch_conv, 1, kernel_size=1, padding=0, stride=1)
        self.linear                 = nn.Linear(1 + self.n_attr, 2)
        self.tanh                   = torch.tanh
        self.softmax                = lambda x: F.softmax(x, dim=1)

    def forward(self, fmap, attr):
        self.c_smears               = self.tanh(self.conv(self.pool(fmap)).squeeze(-1).squeeze(-1))
        self.c_attr                 = attr
        self.l_in                   = torch.cat((self.c_smears, self.c_attr), dim=-1)
        probs                       = self.softmax(self.linear(self.l_in))
        return probs[:,[0]]