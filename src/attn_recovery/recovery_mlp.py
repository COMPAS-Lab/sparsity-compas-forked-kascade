import torch
import torch.nn as nn
from typing import Optional
from collections import OrderedDict

def preprocess_means(data, eps=1e-9):
    """
    pre-processing for distribution means spanning 10^-2 to 10^-5.
    """
    # Capture the sign (polarity)
    sign = torch.sign(data)
    # Apply log10 to the absolute magnitude
    # We add eps to ensure the log is defined even for 0.0 values
    log_magnitude = torch.log10(torch.abs(data) + eps)
    return sign * log_magnitude

def post_process_means(data, eps=1e-9):
    """
    post-processing for distribution means spanning 10^-2 to 10^-5.
    """
    # Capture the sign (polarity)
    sign = torch.sign(data)
    linear_magnitude = torch.pow(10, torch.abs(data))
    return sign * linear_magnitude

class RecoveryMLP(nn.Module):
    def __init__(self, name: str, hidden_size=4096, mlp_dim=128):
        super().__init__()
        self.model_name = name
        
        # A single, pure sequential pipeline
        self.network = nn.Sequential(OrderedDict([
            ('input_proj', nn.Linear(hidden_size, mlp_dim)),
            ('act1', nn.GELU()),
            ('dropout1', nn.Dropout(0.05)),
            # The final layer output 1 value: either mean or std
            ('hidden_layer', nn.Linear(mlp_dim, 1))
        ]))

    def forward(self, x):
        # x: [batch_size, 4096]
        return self.network(x)