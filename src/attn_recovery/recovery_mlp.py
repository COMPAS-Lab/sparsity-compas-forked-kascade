import torch
import torch.nn as nn
from typing import Optional

import torch
import torch.nn as nn

class RecoveryMLP(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        
        # Input: [mean(X), std(X)] = 2 features
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2) # Outputs: [target_mean, target_std]
        )

    def forward(self, x_stats):
        # x_stats: [batch * seq, 2]
        
        out = self.mlp(x_stats)
        # Apply Softplus to the predicted Std (second column)
        target_mean = out[:, 0:1]
        target_std = torch.nn.functional.softplus(out[:, 1:2])
        
        return target_mean, target_std
