from kascade.attn_recovery import RecoveryMLP, preprocess_means, post_process_means
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def load_training_data(base_path: Path, model_name: str):
    # load training files
    input_path = base_path / f"{model_name}_input.npy"
    output_mean_path = base_path / f"{model_name}_output_mean.npy"
    output_std_path = base_path / f"{model_name}_output_std.npy"
    input_hstates = np.load(input_path, allow_pickle=True).item()
    output_mean = np.load(output_mean_path, allow_pickle=True).item()
    output_std = np.load(output_std_path, allow_pickle=True).item()

    n_layers = list(input_hstates.keys())
    # n_layers has the format "layer_<layer number>"
    print(f"detect {len(n_layers)} layers from model {model_name}")
    hidden_dim = input_hstates[n_layers[0]][0].shape[-1]
    print(f"detect hidden dimension: {hidden_dim}")

    attn_ins_data, attn_outs_data_mean, attn_outs_data_std = {}, {}, {}
    attn_ins_test, attn_outs_test_mean, attn_outs_test_std = {}, {}, {}

    for l in n_layers:
        # each layer contains many iterations, prefill and decode iters 
        # are not distinguished here
        # find the last elements in input_mean[l] that has shape[0] > 1, that
        # is the start of the last instances. 
        # Use it as test set
        inst_dim_0 = np.array([iter_dat.shape[0] for iter_dat in input_hstates[l]])
        last_inst_start_id = np.where(inst_dim_0 > 1)[0][-1]

        curr_attn_ins_test = np.concatenate(input_hstates[l][last_inst_start_id:], axis=0)
        curr_attn_ins = np.concatenate(input_hstates[l][:last_inst_start_id], axis=0)

        curr_output_mean_test = np.concatenate(output_mean[l][last_inst_start_id:], axis=0)
        curr_output_std_test = np.concatenate(output_std[l][last_inst_start_id:], axis=0)
        curr_output_mean = np.concatenate(output_mean[l][:last_inst_start_id], axis=0)
        curr_output_std = np.concatenate(output_std[l][:last_inst_start_id], axis=0)

        # check dimensions
        assert(len(curr_attn_ins) == len(curr_output_mean))
        assert(len(curr_attn_ins) == len(curr_output_std))
        assert(len(curr_attn_ins_test) == len(curr_output_mean_test))
        assert(len(curr_attn_ins_test) == len(curr_output_std_test))

        # attach curr iter data
        attn_ins_test[l] = torch.tensor(curr_attn_ins_test, dtype=float)
        attn_outs_test_mean[l] = torch.tensor(curr_output_mean_test, dtype=float)
        attn_outs_test_std[l] = torch.tensor(curr_output_std_test, dtype=float)
        attn_ins_data[l] = torch.tensor(curr_attn_ins, dtype=float)
        attn_outs_data_mean[l] = torch.tensor(curr_output_mean, dtype=float)
        attn_outs_data_std[l] = torch.tensor(curr_output_std, dtype=float)

        print(f"train+validation set at {l}: attn ins size: {attn_ins_data[l].size()}, attn outs mean size: {attn_outs_data_mean[l].size()}, attn outs std size: {attn_outs_data_std[l].size()}")
        print(f"test set at {l}: attn ins size: {attn_ins_test[l].size()}, attn outs mean size: {attn_outs_test_mean[l].size()}, attn outs std size: {attn_outs_test_std[l].size()}")
        
        del input_hstates[l]
        del output_mean[l]
        del output_std[l]

    ret_mean = {
        "train": (attn_ins_data, attn_outs_data_mean),
        "test": (attn_ins_test, attn_outs_test_mean), 
        "n_layers": n_layers,
        "hidden_dim": hidden_dim
    }

    ret_std = {
        "train": (attn_ins_data, attn_outs_data_std),
        "test": (attn_ins_test, attn_outs_test_std), 
        "n_layers": n_layers,
        "hidden_dim": hidden_dim
    }

    return ret_mean, ret_std

def prepare_recovery_dataloaders(attn_ins, expected_outputs, 
                                    batch_size=256, train_ratio=0.9, seed=42):
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
    print(f"data loader: total size: {total_size}, train ratio: {train_ratio}, batch size: {batch_size}")
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
def avg_norm_dist(pred_data, target_data):
    err = abs(pred_data - target_data)
    avg_dist = torch.mean(err)
    
    return avg_dist.item()
    

def vector_distance_loss(pred_data, target_data, **kwargs):
    diff_sq = (target_data - pred_data)**2
    dist = torch.sqrt(diff_sq + 1e-9)
    
    return torch.mean(dist)


# def gaussian_kl_loss(p_mu, p_sigma, gt_mu, gt_sigma, eps=1e-8):
#     """
#     Kullback-Leibler Divergence between two Gaussians.
#     P: Predicted (p_mu, p_sigma)
#     Q: Ground Truth (gt_mu, gt_sigma)
#     """
#     # Term 1: log(sigma_gt / sigma_p)
#     term1 = torch.log(gt_sigma + eps) - torch.log(p_sigma + eps)
    
#     # Term 2: (sigma_p^2 + (mu_p - mu_gt)^2) / (2 * sigma_gt^2)
#     # This is the 'Normalized L2' you were looking for!
#     term2 = (p_sigma**2 + (p_mu - gt_mu)**2) / (2 * gt_sigma**2 + eps)
    
#     # KL Formula
#     kl = term1 + term2 - 0.5
    
#     return torch.mean(kl)


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
    loss_func=torch.nn.functional.huber_loss,
    preprocess_func=lambda x: x,
    post_process_func=lambda x: x,
):
    """
    Trains the RecoveryMLP to predict unpruned attention moments.
    """
    model.to(device)
    # Weight decay helps prevent overfitting to specific token patterns
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_dist = float('inf')
    epochs_no_improve = 0

    # early stopping logic
    best_val_loss = float('inf')
    epochs_no_improve = False
    history = {
        'train_l2_dist': [], 
        'val_l2_dist': [], 
        'avg_err': [],
    }
    
    print(f"Starting training on {device}...")
    initial_delta = 0.5
    loss_func_kwargs = {}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        if loss_func == torch.nn.functional.huber_loss:
            # reduce curr_delta by 0.1 for each 5 epoches, until 0.1
            curr_delta = max(initial_delta - 0.1 * epoch // 5, 0.1)
            loss_func_kwargs["delta"] = curr_delta
        
        # 1. Training Phase
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_y = preprocess_func(batch_y)
            
            # Forward pass: pred is (mean, std)
            pred_data = model(batch_x)
            
            # Calculate combined loss
            loss = loss_func(pred_data, batch_y, **loss_func_kwargs)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # 2. Validation Phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in validate_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_y = preprocess_func(batch_y)
                
                p_data = model(batch_x)
                v_loss = loss_func(p_data, batch_y, **loss_func_kwargs)
                val_loss += v_loss.item()
                
                # Store for R^2 calculation (flattening for simplicity)
                all_preds.append(p_data)
                all_targets.append(batch_y)

        # 3. Calculate Metrics
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(validate_loader)
        
        # Simple R^2 calculation: 1 - (SS_res / SS_tot)
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        all_preds = post_process_func(all_preds)
        all_targets = post_process_func(all_targets)
        avg_dist = avg_norm_dist(all_preds, all_targets)

        history['train_l2_dist'].append(avg_train)
        history['val_l2_dist'].append(avg_val)
        history['avg_err'].append(avg_dist)

        print(f"Epoch {epoch+1}: " + \
                f"Train Loss: {avg_train:.6f} | " + \
                f"Val Loss: {avg_val:.6f} | " + \
                f"avg err: {avg_dist:.4f}")

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
    preprocess_func=lambda x: x,
    post_process_func=lambda x: x,
):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_y = preprocess_func(batch_y)
            
            p_data = model(batch_x)
            
            all_preds.append(p_data)
            all_targets.append(batch_y)

    avg_val = val_loss / len(test_loader)
    # use avg_dist as final result
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    all_preds = post_process_func(all_preds)
    all_targets = post_process_func(all_targets)

    avg_dist = avg_norm_dist(all_preds, all_targets)

    print(f"test results: avg dist: {avg_dist:.4f}")
    res_df = pd.DataFrame({
        "val_l2_dist": [avg_val],
        "avg_err": [avg_dist],    
    })
    res_df.to_csv(result_path, index=False)
    

def visualize_manifold_property_heat(
    hidden_states: torch.Tensor,
    log_mean_targets: torch.Tensor,
    layer_name: str = "layer",
    save_path: Path = Path("manifold_heat.png"),
    n_samples: int = 100,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    random_state: int = 42,
    use_tsne_fallback: bool = False,
):
    """
    2-D Manifold Projection with Property Heat.

    Projects a batch of high-dimensional hidden states (e.g. 4096-dim) down to
    2-D via UMAP (or t-SNE as a fallback) and colours each point by its Ground
    Truth Log-Mean target value.

    Diagnostic interpretation
    -------------------------
    * Smooth colour gradients  → the input space encodes a learnable manifold
      for the mean signal.  The MLP *should* be able to fit it.
    * Randomly mixed colours  → the 4096-dim space is too noisy; the target
      signal is not spatially organised and the MLP will struggle.

    Args:
        hidden_states     : Tensor [N, D] – raw input vectors (CPU or GPU).
        log_mean_targets  : Tensor [N] or [N, 1] – ground-truth mean values
                            (will be log-transformed internally if > 0).
        layer_name        : String label used in the plot title and file name.
        save_path         : Where to write the PNG.
        n_samples         : How many points to project (default 100).
        umap_n_neighbors  : UMAP ``n_neighbors`` hyper-parameter.
        umap_min_dist     : UMAP ``min_dist`` hyper-parameter.
        random_state      : Seed for reproducibility.
        use_tsne_fallback : Use sklearn t-SNE when umap-learn is unavailable.

    Returns:
        Path to the saved figure.
    """
    # ------------------------------------------------------------------ #
    # 1. Sample n_samples points                                           #
    # ------------------------------------------------------------------ #
    N = hidden_states.shape[0]
    n_samples = min(n_samples, N)
    rng = np.random.default_rng(random_state)
    idx = rng.choice(N, size=n_samples, replace=False)

    X = hidden_states[idx].detach().cpu().float().numpy()          # [n, D]
    y_raw = preprocess_means(log_mean_targets[idx])
    y_raw = y_raw.reshape(-1)                                       # [n]

    # ------------------------------------------------------------------ #
    # 2. Log-transform the colour property                                 #
    # ------------------------------------------------------------------ #
    # Shift so all values are positive before taking log, then normalise.
    # y_shifted = y_raw - y_raw.min() + 1e-8
    # y_log = np.log(y_shifted)
    # y_norm = (y_log - y_log.min()) / (y_log.max() - y_log.min() + 1e-12)
    y_norm = y_raw

    # print y_norm stat features
    print(f"y_norm stat features: min={y_norm.min()}, max={y_norm.max()}, mean={y_norm.mean()}, std={y_norm.std()}")

    # ------------------------------------------------------------------ #
    # 3. Dimensionality reduction → 2D                                    #
    # ------------------------------------------------------------------ #
    method_label = ""
    if not use_tsne_fallback:
        try:
            import umap  # noqa: PLC0415
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                random_state=random_state,
                low_memory=False,
            )
            embedding = reducer.fit_transform(X)   # [n, 2]
            method_label = "UMAP"
        except ImportError:
            print("[visualize_manifold] umap-learn not found – falling back to t-SNE.")
            use_tsne_fallback = True

    if use_tsne_fallback:
        from sklearn.manifold import TSNE  # noqa: PLC0415
        reducer = TSNE(
            n_components=2,
            perplexity=min(30, n_samples - 1),
            random_state=random_state,
            n_iter=1000,
        )
        embedding = reducer.fit_transform(X)
        method_label = "t-SNE"

    # ------------------------------------------------------------------ #
    # 4. Plot                                                              #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = cm.get_cmap("RdYlBu_r")   # blue=low, red=high
    sc = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=y_norm,
        cmap=cmap,
        s=40,
        alpha=0.85,
        edgecolors="none",
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Normalised log(Ground Truth Mean)", fontsize=10)

    ax.set_title(
        f"{layer_name}  |  {method_label} 2-D Manifold  |  "
        f"Property Heat: GT Log-Mean  (n={n_samples})",
        fontsize=11,
    )
    ax.set_xlabel(f"{method_label} dim-1", fontsize=9)
    ax.set_ylabel(f"{method_label} dim-2", fontsize=9)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[visualize_manifold] Saved → {save_path}")
    return save_path


def main():

    parser = argparse.ArgumentParser(description="Train Recovery MLP")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--data_base_path", type=str, required=True, help="Path to the training data directory")
    parser.add_argument("--model_path", type=str, required=True, help="Directory to save the trained model")
    args = parser.parse_args()

    data_base_path = Path(args.data_base_path)
    model_path = Path(args.model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    mlp_dim=256
    batch_size=1024
    
    print(f"loading model files...")
    loaded_raw_data_mean, loaded_raw_data_std = \
        load_training_data(data_base_path, args.model_name)

    # try to train for mean prediction first
    # disassemble data for data loader preparation
    n_layers = loaded_raw_data_mean["n_layers"]
    hidden_dim = loaded_raw_data_mean["hidden_dim"]
    attn_ins, attn_outs = loaded_raw_data_mean["train"]
    attn_ins_test, attn_outs_test = loaded_raw_data_mean["test"]


    # data visualization for all layers
    for l in n_layers:
        layer_name = l
        save_path = Path("results") / f"{args.model_name}_recovery_mlp_{mlp_dim}_{l}_manifold.png"
        visualize_manifold_property_heat(
            attn_ins[layer_name],
            attn_outs[layer_name],
            layer_name=layer_name,
            save_path=save_path,
            n_samples=1000,
        )

    exit()

    mlp_name = f"{args.model_name}_recovery_mlp_{mlp_dim}_mean"
    loss_func = torch.nn.functional.huber_loss
    pre_process_func = preprocess_means
    post_process_func = lambda x: x

    # train each mlp for each layer, save them separately
    for l in n_layers:
        print(f"training {l} layer for mean...")
        train_loader, validate_loader = \
            prepare_recovery_dataloaders(attn_ins[l], attn_outs[l], batch_size=batch_size)
        _, test_loader = \
            prepare_recovery_dataloaders(attn_ins_test[l], attn_outs_test[l], train_ratio=0.0, batch_size=batch_size)

        # start training
        model = RecoveryMLP(mlp_name, hidden_size=hidden_dim, mlp_dim=mlp_dim).cuda()
        save_path = model_path / "mean" / f"{args.model_name}_recovery_mlp_{mlp_dim}_{l}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        trained_model = train_recovery_mlp(
            model, 
            train_loader, 
            validate_loader, 
            epochs=300, 
            model_save_path=save_path, 
            patience=15, 
            loss_func=loss_func,
            preprocess_func=pre_process_func,
            post_process_func=post_process_func,
            )
        print(f"Model saved to {save_path}")

        # start testing
        test_res_path = model_path / "mean" / f"{args.model_name}_recovery_mlp_{mlp_dim}_{l}_test_log.csv"
        test_recovery_mlp(trained_model, test_loader, result_path=test_res_path)

if __name__ == "__main__":
    main()
