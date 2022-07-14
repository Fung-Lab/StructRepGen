import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ff_net(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(ff_net, self).__init__()
        self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
        self.lin_list = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim) for i in range(layers)]
        )

        self.lin2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = torch.nn.functional.relu(self.lin1(x))
        for layer in self.lin_list:
            out = torch.nn.functional.relu(layer(out))
        out = self.lin2(out)
        return out