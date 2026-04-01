import torch
import torch.nn as nn
from typing import Optional
from collections import OrderedDict

class RecoveryMLP(nn.Module):
    def __init__(self, hidden_size=4096, mlp_dim=128):
        super().__init__()
        self.model_name = f"recovery_mlp_{mlp_dim}"
        
        self.network = nn.Sequential(OrderedDict([
            ('input_proj', nn.Linear(hidden_size, mlp_dim)),
            ('act1', nn.GELU()),
            ('dropout1', nn.Dropout(0.1)),
            ('output_layer', nn.Linear(mlp_dim, 1))
        ]))

    def forward(self, x):
        raw_output = self.network(x)
        pred_std = torch.nn.functional.softplus(raw_output[:, 0])
        
        return pred_std