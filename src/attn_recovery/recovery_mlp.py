import torch
import torch.nn as nn
from typing import Optional

import torch
import torch.nn as nn


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
    def __init__(self, hidden_dim=16):
        super().__init__()
        
        # Input: [mean(X), std(X)] = 2 features
        self.mlp = nn.Sequential(OrderedDict([
            ('input_proj', nn.Linear(2, hidden_dim)),
            ('act1', nn.GELU()),
            ('dropout1', nn.Dropout(0.05)),
            ('hidden_layer', nn.Linear(hidden_dim, 1))
        ]))

    def forward(self, x_stats):
        # x_stats: [batch * seq, 2]
        return self.mlp(x_stats)
