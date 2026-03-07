from kascade.attn_recovery import RecoveryMLP
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd

def load_training_data(base_path: Path, model_name: str):
    # load training files
    input_mean_path = base_path / f"{model_name}_input_mean.npy"
    input_std_path = base_path / f"{model_name}_input_std.npy"
    output_mean_path = base_path / f"{model_name}_output_mean.npy"
    output_std_path = base_path / f"{model_name}_output_std.npy"
    input_mean = np.load(input_mean_path, allow_pickle=True).item()
    input_std = np.load(input_std_path, allow_pickle=True).item()
    output_mean = np.load(output_mean_path, allow_pickle=True).item()
    output_std = np.load(output_std_path, allow_pickle=True).item()

    n_layers = list(input_mean.keys())
    # n_layers has the format "layer_<layer number>"
    print(f"detect {len(n_layers)} layers from model {model_name}")

    attn_ins_data, attn_outs_data = {}, {}
    attn_ins_test, attn_outs_test = {}, {}

    for l in n_layers:
        # each layer contains many iterations, prefill and decode iters 
        # are not distinguished here
        # find the last elements in input_mean[l] that has shape[0] > 1, that
        # is the start of the last instances. 
        # Use it as test set
        inst_dim_0 = np.array([iter_dat.shape[0] for iter_dat in input_mean[l]])
        last_inst_start_id = np.where(inst_dim_0 > 1)[0][-1]

        curr_input_mean_test = np.concatenate(input_mean[l][last_inst_start_id:], axis=0)
        curr_input_std_test = np.concatenate(input_std[l][last_inst_start_id:], axis=0)
        curr_output_mean_test = np.concatenate(output_mean[l][last_inst_start_id:], axis=0)
        curr_output_std_test = np.concatenate(output_std[l][last_inst_start_id:], axis=0)

        curr_input_mean = np.concatenate(input_mean[l][:last_inst_start_id], axis=0)
        curr_input_std = np.concatenate(input_std[l][:last_inst_start_id], axis=0)
        curr_output_mean = np.concatenate(output_mean[l][:last_inst_start_id], axis=0)
        curr_output_std = np.concatenate(output_std[l][:last_inst_start_id], axis=0)

        curr_attn_ins_test = np.concatenate([curr_input_mean_test, curr_input_std_test], axis=-1)
        curr_attn_outs_test = np.concatenate([curr_output_mean_test, curr_output_std_test], axis=-1)
        curr_attn_ins = np.concatenate([curr_input_mean, curr_input_std], axis=-1)
        curr_attn_outs = np.concatenate([curr_output_mean, curr_output_std], axis=-1)

        # check dimensions
        assert(len(curr_attn_ins) == len(curr_attn_outs))
        assert(len(curr_attn_ins_test) == len(curr_attn_outs_test))

        # attach curr iter data
        attn_ins_test[l] = torch.tensor(curr_attn_ins_test, dtype=float)
        attn_outs_test[l] = torch.tensor(curr_attn_outs_test, dtype=float)
        attn_ins_data[l] = torch.tensor(curr_attn_ins, dtype=float)
        attn_outs_data[l] = torch.tensor(curr_attn_outs, dtype=float)

        print(f"train+validation set at {l}: attn ins size: {attn_ins_data[l].size()}, attn outs size: {attn_outs_data[l].size()}")
        print(f"test set at {l}: attn ins size: {attn_ins_test[l].size()}, attn outs size: {attn_outs_test[l].size()}")
        
        del input_mean[l]
        del input_std[l]
        del output_mean[l]
        del output_std[l]

    ret = {
        "train": (attn_ins_data, attn_outs_data),
        "test": (attn_ins_test, attn_outs_test), 
        "n_layers": n_layers
    }

    return ret

def prepare_recovery_dataloaders(attn_ins, expected_outputs, 
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
    expected_outputs = expected_outputs.to(torch.float32)

    # 2. Create the unified Dataset
    full_dataset = TensorDataset(attn_ins, expected_outputs)

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
    if len(train_dataset) > 0:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    else:
        train_loader = None
    
    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            pin_memory=True
        )
    else:
        test_loader = None

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

def avg_norm_dist(pred_mu, pred_std, target_mu, target_std):
    # Inputs are tensors of shape [N, 1]
    
    # 1. Compute absolute errors
    mu_err = abs(target_mu - pred_mu) / (abs(target_mu) + 1e-8)
    std_err = abs(target_std - pred_std) / target_std
    
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
    validate_loader, 
    epochs=100, 
    lr=1e-3, 
    patience=5,
    min_delta=1e-5,
    device="cuda",
    model_save_path=Path(""),
):
    """
    Trains the RecoveryMLP to predict unpruned attention moments.
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
        'train_l2_dist': [], 'val_l2_dist': [], 
        'avg_mu_err': [], 'avg_std_err': [], 
    }
    
    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # 1. Training Phase
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass: pred is (mean, std)
            pred_mean, pred_std = model(batch_x)
            
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
            for batch_x, batch_y in validate_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                p_mean, p_std = model(batch_x)
                v_loss = vector_distance_loss(p_mean, p_std, batch_y[:, 0:1], batch_y[:, 1:2])
                val_loss += v_loss.item()
                
                # Store for R^2 calculation (flattening for simplicity)
                all_preds_mean.append(p_mean)
                all_targets_mean.append(batch_y[:, 0:1])
                all_preds_std.append(p_std)
                all_targets_std.append(batch_y[:, 1:2])

        # 3. Calculate Metrics
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(validate_loader)
        
        # Simple R^2 calculation: 1 - (SS_res / SS_tot)
        all_preds_mean = torch.cat(all_preds_mean, dim=0)
        all_targets_mean = torch.cat(all_targets_mean, dim=0)

        all_preds_std = torch.cat(all_preds_std, dim=0)
        all_targets_std = torch.cat(all_targets_std, dim=0)

        avg_dist_mu, avg_dist_std = \
            avg_norm_dist(all_preds_mean, all_preds_std, all_targets_mean, all_targets_std)

        history['train_l2_dist'].append(avg_train)
        history['val_l2_dist'].append(avg_val)
        history['avg_mu_err'].append(avg_dist_mu)
        history['avg_std_err'].append(avg_dist_std)

        print(f"Epoch {epoch+1}: " + \
                f"Train Loss: {avg_train:.6f} | " + \
                f"Val Loss: {avg_val:.6f} | " + \
                f"avg mu err: {avg_dist_mu:.4f} | " + \
                f"avg std err: {avg_dist_std:.4f}")

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

def test_recovery_mlp(
    model, 
    test_loader, 
    device="cuda",
    result_path=Path(""),
):
    model.eval()
    val_loss = 0.0
    # separately consider R^2 of mean and std
    all_preds_mean = []
    all_targets_mean = []
    all_preds_std = []
    all_targets_std = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            p_mean, p_std = model(batch_x)
            v_loss = vector_distance_loss(p_mean, p_std, batch_y[:, 0:1], batch_y[:, 1:2])
            val_loss += v_loss.item()
            
            all_preds_mean.append(p_mean)
            all_targets_mean.append(batch_y[:, 0:1])
            all_preds_std.append(p_std)
            all_targets_std.append(batch_y[:, 1:2])

    avg_val = val_loss / len(test_loader)
    # use avg_dist as final result
    all_preds_mean = torch.cat(all_preds_mean, dim=0)
    all_targets_mean = torch.cat(all_targets_mean, dim=0)

    all_preds_std = torch.cat(all_preds_std, dim=0)
    all_targets_std = torch.cat(all_targets_std, dim=0)

    avg_dist_mu, avg_dist_std = \
        avg_norm_dist(all_preds_mean, all_preds_std, all_targets_mean, all_targets_std)

    print(f"test results: avg mu dist: {avg_dist_mu:.4f}, avg std dist: {avg_dist_std:.4f}")
    res_df = pd.DataFrame({
        "val_l2_dist": [avg_val],
        "avg_mu_err": [avg_dist_mu],
        "avg_std_err": [avg_dist_std],    
    })
    res_df.to_csv(result_path, index=False)
    

def main():
    parser = argparse.ArgumentParser(description="Train Recovery MLP")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--data_base_path", type=str, required=True, help="Path to the training data directory")
    parser.add_argument("--model_path", type=str, required=True, help="Directory to save the trained model")
    args = parser.parse_args()

    data_base_path = Path(args.data_base_path)
    model_path = Path(args.model_path)
    hidden_dim=16
    
    print(f"loading model files...")
    loaded_raw_data = \
        load_training_data(data_base_path, args.model_name)

    # disassemble data for data loader preparation
    n_layers = loaded_raw_data["n_layers"]
    attn_ins, attn_outs = loaded_raw_data["train"]
    attn_ins_test, attn_outs_test = loaded_raw_data["test"]

    # train each mlp for each layer, save them separately
    for l in n_layers:
        print(f"training {l}...")
        train_loader, validate_loader = \
            prepare_recovery_dataloaders(attn_ins[l], attn_outs[l])
        _, test_loader = \
            prepare_recovery_dataloaders(attn_ins_test[l], attn_outs_test[l], train_ratio=0.0)

        # start training
        model = RecoveryMLP(hidden_dim=hidden_dim).cuda()

        model_path.mkdir(parents=True, exist_ok=True)
        save_path = model_path / f"{args.model_name}_recovery_mlp_{hidden_dim}_{l}.pt"
        trained_model = \
            train_recovery_mlp(model, train_loader, validate_loader, model_save_path=save_path, patience=10)
        print(f"Model saved to {save_path}")

        # start testing
        test_res_path = model_path / f"{args.model_name}_recovery_mlp_{hidden_dim}_{l}_test_log.csv"
        test_recovery_mlp(trained_model, test_loader, result_path=test_res_path)

if __name__ == "__main__":
    main()
