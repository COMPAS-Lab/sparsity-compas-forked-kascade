import torch
import torch.nn as nn
from typing import Optional
from collections import OrderedDict

class RecoveryMLP(nn.Module):
    def __init__(self, hidden_size=4096, mlp_dim=128):
        super().__init__()
        self.model_name = f"recovery_mlp_{mlp_dim}"
        
        # A single, pure sequential pipeline
        self.network = nn.Sequential(OrderedDict([
            ('input_proj', nn.Linear(hidden_size, mlp_dim)),
            ('act1', nn.GELU()),
            ('dropout1', nn.Dropout(0.1)),
            # The final layer outputs 2 values: [mean, std_pre_softplus]
            ('output_layer', nn.Linear(mlp_dim, 2))
        ]))

    def forward(self, x):
        # x: [batch_size, 4096]
        
        # Raw predictions for both stats
        raw_output = self.network(x) # [batch_size, 2]
        
        # Split the outputs
        pred_mean = raw_output[:, 0:1]
        
        # We still need to ensure standard deviation is positive.
        # Even in a sequential model, we apply Softplus to the second channel.
        pred_std = torch.nn.functional.softplus(raw_output[:, 1:2])
        
        return pred_mean, pred_std