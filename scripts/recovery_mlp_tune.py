"""
recovery_mlp_tune.py

Standalone Optuna hyperparameter search for RecoveryMLP's `mlp_dim`.

Usage:
    python recovery_mlp_tune.py \
        --model_name  llama \
        --data_base_path /path/to/data \
        --n_trials 20 \
        --tune_epochs 30

The script:
1. Loads training data (reusing helpers from recovery_mlp_train).
2. Picks the first layer as a representative layer for the search.
3. Runs an Optuna study over the categorical search space {32, 64, 128, 256, 512}.
4. Prints and saves the best `mlp_dim` to a JSON file next to the data.
"""

import argparse
import json
from pathlib import Path

import optuna
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Ensure the scripts/ directory is on sys.path so that recovery_mlp_train
# can be imported regardless of the current working directory.
import sys
sys.path.insert(0, str(Path(__file__).parent))

from recovery_mlp_train import (
    load_training_data,
    prepare_recovery_dataloaders,
    gaussian_kl_loss,
)
from kascade.attn_recovery import RecoveryMLP


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial, attn_ins, attn_outs, hidden_dim, tune_epochs, device):
    """
    Train one trial with a sampled mlp_dim and return the best validation KL loss.

    Optuna's MedianPruner can prune the trial early by calling trial.should_prune()
    after each epoch.
    """
    mlp_dim = trial.suggest_categorical("mlp_dim", [32, 64, 128, 256, 512])

    train_loader, val_loader = prepare_recovery_dataloaders(attn_ins, attn_outs)

    model = RecoveryMLP(hidden_size=hidden_dim, mlp_dim=mlp_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val = float("inf")

    for epoch in range(tune_epochs):
        # --- Training phase ---
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred_mean, pred_std = model(batch_x)
            loss = gaussian_kl_loss(
                pred_mean, pred_std, batch_y[:, 0:1], batch_y[:, 1:2]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                p_mean, p_std = model(batch_x)
                val_loss += gaussian_kl_loss(
                    p_mean, p_std, batch_y[:, 0:1], batch_y[:, 1:2]
                ).item()

        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)
        best_val = min(best_val, avg_val)

        # Report intermediate value to Optuna for pruning
        trial.report(avg_val, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def tune_mlp_dim(
    attn_ins,
    attn_outs,
    hidden_dim: int,
    n_trials: int = 20,
    tune_epochs: int = 30,
    device: str = "cuda",
) -> int:
    """
    Create and run an Optuna study to find the best mlp_dim.

    Args:
        attn_ins:    Input tensor for the representative layer.
        attn_outs:   Target tensor for the representative layer.
        hidden_dim:  Input feature dimension (passed to RecoveryMLP).
        n_trials:    Number of Optuna trials to run.
        tune_epochs: Max training epochs per trial (shorter than final training).
        device:      Torch device string.

    Returns:
        Best mlp_dim found by the study.
    """
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    study.optimize(
        lambda trial: objective(
            trial, attn_ins, attn_outs, hidden_dim, tune_epochs, device
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_mlp_dim = study.best_params["mlp_dim"]
    print(
        f"\n[Optuna] Best mlp_dim = {best_mlp_dim}  "
        f"(val KL loss = {study.best_value:.6f})"
    )

    # Print a compact summary of all trials
    print("\n[Optuna] Trial summary:")
    for t in sorted(study.trials, key=lambda t: t.value if t.value is not None else float("inf")):
        status = t.state.name
        val = f"{t.value:.6f}" if t.value is not None else "pruned"
        print(f"  Trial {t.number:3d} | mlp_dim={t.params.get('mlp_dim', '?'):>4} | val={val} | {status}")

    return best_mlp_dim, study


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for RecoveryMLP mlp_dim"
    )
    parser.add_argument("--model_name", type=str, required=True, help="Model name (used to locate data files)")
    parser.add_argument("--data_base_path", type=str, required=True, help="Directory containing the .npy training data")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials (default: 20)")
    parser.add_argument("--tune_epochs", type=int, default=30, help="Max epochs per trial (default: 30)")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device (default: cuda)")
    parser.add_argument(
        "--rep_layer", type=str, default=None,
        help="Representative layer to use for tuning (e.g. 'layer_0'). "
             "Defaults to the first layer found in the data."
    )
    args = parser.parse_args()

    data_base_path = Path(args.data_base_path)

    # 1. Load data
    print("Loading training data...")
    loaded = load_training_data(data_base_path, args.model_name)
    n_layers = loaded["n_layers"]
    hidden_dim = loaded["hidden_dim"]
    attn_ins, attn_outs = loaded["train"]

    # 2. Pick representative layer
    rep_layer = args.rep_layer if args.rep_layer else n_layers[0]
    if rep_layer not in attn_ins:
        raise ValueError(f"Layer '{rep_layer}' not found. Available: {n_layers}")
    print(f"Using layer '{rep_layer}' as representative for tuning.")

    # 3. Run Optuna study
    best_mlp_dim, study = tune_mlp_dim(
        attn_ins=attn_ins[rep_layer],
        attn_outs=attn_outs[rep_layer],
        hidden_dim=hidden_dim,
        n_trials=args.n_trials,
        tune_epochs=args.tune_epochs,
        device=args.device,
    )

    # 4. Save result
    result = {
        "best_mlp_dim": best_mlp_dim,
        "best_val_loss": study.best_value,
        "n_trials": args.n_trials,
        "tune_epochs": args.tune_epochs,
        "rep_layer": rep_layer,
    }
    out_path = f"./results/attn_recovery/{args.model_name}_optuna_best_mlp_dim.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to {out_path}")


if __name__ == "__main__":
    main()
