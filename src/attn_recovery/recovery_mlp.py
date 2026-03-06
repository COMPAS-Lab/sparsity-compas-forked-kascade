import torch
import torch.nn as nn
from typing import Optional

import torch
import torch.nn as nn

class RecoveryMLP(nn.Module):
    def __init__(self, num_layers=32, hidden_dim=32):
        super().__init__()
        self.layer_emb = nn.Embedding(num_layers, 8)
        
        # Shared trunk: Learns general "energy" of the token
        self.shared_trunk = nn.Sequential(
            nn.Linear(2 + 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Separate "Heads" for Mean and Std
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.std_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def forward(self, x_stats, layer_id):
        emb = self.layer_emb(layer_id)
        feat = self.shared_trunk(torch.cat([x_stats, emb], dim=-1))
        
        return self.mean_head(feat), self.std_head(feat)

class RecoveryDualMLP(nn.Module):
    def __init__(self, num_layers=32, hidden_dim=16):
        super().__init__()
        # Shared Embedding
        self.layer_emb = nn.Embedding(num_layers, 8)
        
        # MLP for Mean Prediction
        self.mean_mlp = nn.Sequential(
            nn.Linear(2 + 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # MLP for Std Prediction
        self.std_mlp = nn.Sequential(
            nn.Linear(2 + 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def forward(self, x_stats, layer_id):
        # x_stats: [batch, 2], layer_id: [batch]
        emb = self.layer_emb(layer_id)
        input_feat = torch.cat([x_stats, emb], dim=-1)
        
        pred_mean = self.mean_mlp(input_feat)
        pred_std = self.std_mlp(input_feat)
        
        return pred_mean, pred_std