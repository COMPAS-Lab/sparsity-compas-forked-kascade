import torch
import torch.nn as nn
from typing import Optional

import torch
import torch.nn as nn

class RecoveryMLP(nn.Module):
    def __init__(self, num_layers=32, hidden_dim=16):
        super().__init__()
        # Layer Context: Embed the layer_id into a small vector
        self.layer_emb = nn.Embedding(num_layers, 4) 
        
        # Input: [mean(X), std(X), layer_emb(4)] = 6 features
        self.net = nn.Sequential(
            nn.Linear(2 + 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2) # Outputs: [target_mean, target_std]
        )

    def forward(self, x_stats, layer_id):
        # x_stats: [batch * seq, 2]
        # layer_id: [batch * seq] (long tensor)
        
        embs = self.layer_emb(layer_id)
        combined = torch.cat([x_stats, embs], dim=-1)
        
        out = self.net(combined)
        # Apply Softplus to the predicted Std (second column)
        target_mean = out[:, 0:1]
        target_std = torch.nn.functional.softplus(out[:, 1:2])
        
        return target_mean, target_std