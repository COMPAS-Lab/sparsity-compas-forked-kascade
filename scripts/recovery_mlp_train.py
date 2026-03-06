from kascade.attn_recovery import RecoveryMLP, RecoveryDualMLP
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd

def load_training_data(base_path: Path, model_name: str, layer_id=-1):
    # load training files
    input_mean_path = base_path / f"{model_name}_input_mean.npy"
    input_std_path = base_path / f"{model_name}_input_std.npy"
    output_mean_path = base_path / f"{model_name}_output_mean.npy"
    output_std_path = base_path / f"{model_name}_output_std.npy"
    input_mean = np.load(input_mean_path, allow_pickle=True).item()
    input_std = np.load(input_std_path, allow_pickle=True).item()
    output_mean = np.load(output_mean_path, allow_pickle=True).item()
    output_std = np.load(output_std_path, allow_pickle=True).item()

    n_layers = input_mean.keys()
    # n_layers has the format "layer_<layer number>"
    print(f"detect {len(n_layers)} layers from model {model_name}")

    attn_ins_data, attn_outs_data = [], []
    attn_data_layer_ids = []
    for l in n_layers:
        if layer_id != -1 and int(l.split("_")[-1]) != layer_id:
            continue
        # each layer contains many iterations, prefill and decode iters 
        # are not distinguished here
        curr_input_mean = np.concatenate(input_mean[l], axis=0)
        curr_input_std = np.concatenate(input_std[l], axis=0)
        curr_output_mean = np.concatenate(output_mean[l], axis=0)
        curr_output_std = np.concatenate(output_std[l], axis=0)

        curr_batch_size = curr_input_mean.shape[0]
        layer_ids = int(l.split("_")[-1])
        layer_ids = np.array([layer_ids] * curr_batch_size)

        curr_attn_ins = np.concatenate([curr_input_mean, curr_input_std], axis=-1)
        curr_attn_outs = np.concatenate([curr_output_mean, curr_output_std], axis=-1)

        # check dimensions
        assert(len(curr_attn_ins) == len(curr_attn_outs) and \
                len(curr_attn_ins) == len(layer_ids))
        attn_ins_data.append(curr_attn_ins)
        attn_outs_data.append(curr_attn_outs)
        attn_data_layer_ids.append(layer_ids)

    # concat all generated data
    attn_ins_data = torch.tensor(np.concatenate(attn_ins_data, axis=0), dtype=float)
    attn_outs_data = torch.tensor(np.concatenate(attn_outs_data, axis=0), dtype=float)
    attn_layer_ids = torch.tensor(np.concatenate(attn_data_layer_ids, axis=0), dtype=int)
    print(f"data loader: attn ins size: {attn_ins_data.size()}, attn outs size: {attn_outs_data.size()}, layer id size: {attn_layer_ids.size()}")

    return attn_ins_data, attn_outs_data, attn_layer_ids, len(n_layers)

def prepare_recovery_dataloaders(attn_ins, layer_ids, expected_outputs, 
                                    batch_size=1024, train_ratio=0.9, seed=42):
    """
    Transforms raw tensors into Train and Test DataLoaders.
    
    Args:
        attn_ins: [N, 2] - FloatTensor (mean/std of X)
        layer_ids: [N, 1] or [N] - LongTensor (the layer index)
        expected_outputs: [N, 2] - FloatTensor (target mean/std of O)
    """
    
    # 1. Ensure correct dtypes and shapes
    # layer_ids must be Long for the Embedding layer; squeeze to [N]
    attn_ins = attn_ins.to(torch.float32)
    layer_ids = layer_ids.to(torch.long).squeeze()
    expected_outputs = expected_outputs.to(torch.float32)

    # 2. Create the unified Dataset
    full_dataset = TensorDataset(attn_ins, layer_ids, expected_outputs)

    # 3. Calculate split lengths
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size

    # 4. Perform the split
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed) 
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True
    )

    print(f"Dataset Split: {train_size} train samples, {test_size} test samples.")
    return train_loader, test_loader

# metric
def r2_score(y_true, y_pred):
    """
    Calculate the R^2 score.
    
    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        
    Returns:
        R^2 score.
    """
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def avg_l2_dist(pred_mu, pred_std, target_mu, target_std):
    # Inputs are tensors of shape [N, 1]
    
    # 1. Compute absolute errors
    mu_err = (target_mu - pred_mu)**2
    std_err = (target_std - pred_std)**2
    
    # 2. Treat as a 2D vector distance per token
    dist = torch.sqrt(mu_err + std_err) # L2 distance in mu-std space
    
    # 3. Calculate Mean Distance
    avg_dist = torch.mean(dist)
    
    # 4. (Optional) Relative distance compared to the target magnitude
    target_mag = torch.sqrt(target_mu**2 + target_std**2 + 1e-6)
    rel_dist = torch.mean(dist / target_mag)
    
    return avg_dist.item(), rel_dist.item()

def avg_dist(pred_mu, pred_std, target_mu, target_std):
    # Inputs are tensors of shape [N, 1]
    
    # 1. Compute absolute errors
    mu_err = abs(target_mu - pred_mu)
    std_err = abs(target_std - pred_std)
    
    # 2. Calculate Mean Distance
    avg_dist_mu = torch.mean(mu_err)
    avg_dist_std = torch.mean(std_err)
    
    return avg_dist_mu.item(), avg_dist_std.item()
    
def vector_distance_loss(pred_mean, pred_std, target_mean, target_std):
    # Treat (mean, std) as a 2D coordinate
    # Use torch.sqrt(sum_of_squares)
    diff_sq = (target_mean - pred_mean)**2 + (target_std - pred_std)**2
    
    # We add 1e-8 for numerical stability of the sqrt gradient
    dist = torch.sqrt(diff_sq + 1e-8)
    
    return torch.mean(dist)
    
def train_recovery_mlp(
    model, 
    train_loader, 
    test_loader, 
    epochs=100, 
    lr=1e-3, 
    patience=5,
    min_delta=1e-5,
    device="cuda",
    model_save_path=Path(""),
):
    """
    Trains the RecoveryOracle to predict unpruned attention moments.
    """
    model.to(device)
    # Weight decay helps prevent overfitting to specific token patterns
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Scheduler: Reduces LR when the distance plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_dist = float('inf')
    epochs_no_improve = 0

    # early stopping logic
    best_val_loss = float('inf')
    epochs_no_improve = False
    history = {
        'train': [], 'val': [], 
        'avg_mu_err': [], 'avg_std_err': [], 
    }
    
    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # 1. Training Phase
        for batch_x, batch_layers, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_x = batch_x.to(device)
            batch_layers = batch_layers.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass: pred is (mean, std)
            pred_mean, pred_std = model(batch_x, batch_layers)
            
            # Calculate combined loss
            loss = vector_distance_loss(pred_mean, pred_std, batch_y[:, 0:1], batch_y[:, 1:2])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # 2. Validation Phase
        model.eval()
        val_loss = 0.0
        # separately consider R^2 of mean and std
        all_preds_mean = []
        all_targets_mean = []
        all_preds_std = []
        all_targets_std = []
        
        with torch.no_grad():
            for batch_x, batch_layers, batch_y in test_loader:
                batch_x, batch_layers, batch_y = batch_x.to(device), batch_layers.to(device), batch_y.to(device)
                
                p_mean, p_std = model(batch_x, batch_layers)
                v_loss = vector_distance_loss(p_mean, p_std, batch_y[:, 0:1], batch_y[:, 1:2])
                val_loss += v_loss.item()
                
                # Store for R^2 calculation (flattening for simplicity)
                all_preds_mean.append(p_mean)
                all_targets_mean.append(batch_y[:, 0:1])
                all_preds_std.append(p_std)
                all_targets_std.append(batch_y[:, 1:2])

        # 3. Calculate Metrics
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(test_loader)
        
        # Simple R^2 calculation: 1 - (SS_res / SS_tot)
        all_preds_mean = torch.cat(all_preds_mean, dim=0)
        all_targets_mean = torch.cat(all_targets_mean, dim=0)

        all_preds_std = torch.cat(all_preds_std, dim=0)
        all_targets_std = torch.cat(all_targets_std, dim=0)

        avg_dist_mu, avg_dist_std = \
            avg_dist(all_preds_mean, all_preds_std, all_targets_mean, all_targets_std)

        history['train'].append(avg_train)
        history['val'].append(avg_val)
        history['avg_mu_err'].append(avg_dist_mu)
        history['avg_std_err'].append(avg_dist_std)

        print(f'''Epoch {epoch+1}: 
                Train Loss: {avg_train:.6f} | 
                Val Loss: {avg_val:.6f} | 
                avg dist mu: {avg_dist_mu:.4f} | 
                avg dist std: {avg_dist_std:.4f}''')

        # Check if the improvement is greater than min_delta
        if avg_val < (best_val_loss - min_delta):
            best_val_loss = avg_val
            epochs_no_improve = 0
            # Save the best model state
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered! No significant improvement for {patience} epochs.")
            break

    # Reload best weights before returning
    model.load_state_dict(torch.load(model_save_path))
    # save history to csv file
    history_df = pd.DataFrame(history)
    model_train_log_path = model_save_path.name.replace(".pt", "_train_log.csv")
    history_df.to_csv(model_save_path.parent / model_train_log_path, index=False)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Recovery MLP")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--data_base_path", type=str, required=True, help="Path to the training data directory")
    parser.add_argument("--model_path", type=str, required=True, help="Directory to save the trained model")
    args = parser.parse_args()

    data_base_path = Path(args.data_base_path)
    model_path = Path(args.model_path)
    hidden_dim=16
    
    attn_ins, attn_outs, layer_ids, n_layers = \
        load_training_data(data_base_path, args.model_name, layer_id=2)
    train_loader, test_loader = \
        prepare_recovery_dataloaders(attn_ins, layer_ids, attn_outs)

    # model = RecoveryMLP(num_layers=n_layers, hidden_dim=hidden_dim).cuda()
    model = RecoveryDualMLP(num_layers=n_layers, hidden_dim=hidden_dim).cuda()

    model_path.mkdir(parents=True, exist_ok=True)
    save_path = model_path / f"{args.model_name}_recovery_dual_mlp_{hidden_dim}.pt"
    trained_model = \
        train_recovery_mlp(model, train_loader, test_loader, model_save_path=save_path, patience=10)

    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
