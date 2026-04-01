import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import torch
import argparse


def plot_attn_stat_shifting(model_name):
    attn_o_stat_path = Path(f"./results/attn_recovery")
    attn_out_origin_mean_path = attn_o_stat_path / f"train/{model_name}_output_mean.npy"
    attn_out_origin_var_path = attn_o_stat_path / f"train/{model_name}_output_std.npy"
    attn_out_pruned_mean_path = attn_o_stat_path / f"pruned/{model_name}_output_mean.npy"
    attn_out_pruned_var_path = attn_o_stat_path / f"pruned/{model_name}_output_std.npy"

    # load and squeeze the attn output stat data
    attn_o_pruned_mean = np.load(attn_out_pruned_mean_path, allow_pickle=True).item()
    attn_o_pruned_var = np.load(attn_out_pruned_var_path, allow_pickle=True).item()
    attn_o_origin_mean = np.load(attn_out_origin_mean_path, allow_pickle=True).item()
    attn_o_origin_var = np.load(attn_out_origin_var_path, allow_pickle=True).item()

    # attn output stat data shape: (layer_id, batch_id, seq_len, 1)
    layer_ids = list(attn_o_origin_mean.keys())
    num_layers = len(layer_ids)
    batch_id = 0
    num_tokens = attn_o_origin_mean[layer_ids[0]][batch_id].shape[0]
    print("num_layers: ", num_layers)
    print("num_tokens: ", num_tokens)
    num_tokens_pruned = attn_o_pruned_mean[layer_ids[0]][batch_id].shape[0]
    print("num_tokens_pruned: ", num_tokens_pruned)

    # generate a subplots for attn output stat, for mean and var respectively
    # each plot has num_insts subplots
    # each subplot is a scatter plot of attn output stat, x axis is the seq_len, 
    # y axis is the value of attn output stat
    fig_mean, axes_mean = plt.subplots(num_layers, 1, figsize=(22, 60))
    plt.subplots_adjust(hspace=0.3)
    fig_var, axes_var = plt.subplots(num_layers, 1, figsize=(22, 60))
    plt.subplots_adjust(hspace=0.3)

    scatter_configs = {
        "s": 7,
        "linewidths": 0,
        "alpha": 0.4,
    }

    for stat_name in ["mean", "std"]:
        axes = axes_mean if stat_name == "mean" else axes_var
        fig = fig_mean if stat_name == "mean" else fig_var
        attn_o_pruned_stat = attn_o_pruned_mean if stat_name == "mean" else attn_o_pruned_var
        attn_o_origin_stat = attn_o_origin_mean if stat_name == "mean" else attn_o_origin_var

        all_axes = axes.flatten()
        ymin = 0.0 if stat_name == "std" else -0.001
        ymax = 0.01 if stat_name == "std" else 0.001

        for layer_idx in range(num_layers):
            all_axes[layer_idx].grid(True)
            # draw a line with arrow, from 0.0 to the value of attn_o_stat, on y axis
            # for each x=token_ids, and the arrow width is 0.04, reduce arrow size to 0.004
            for token_id in range(num_tokens):
                ori_attn_o_stat = attn_o_origin_stat[layer_ids[layer_idx]][batch_id][token_id][0]
                pruned_attn_o_stat = attn_o_pruned_stat[layer_ids[layer_idx]][batch_id][token_id][0]
                
                if abs(ori_attn_o_stat - pruned_attn_o_stat) < 1e-3:
                    continue

                # trim the arrow to ymin and ymax if it exceeds the range
                if pruned_attn_o_stat > ymax:
                    pruned_attn_o_stat = ymax
                if pruned_attn_o_stat < ymin:
                    pruned_attn_o_stat = ymin
                if ori_attn_o_stat > ymax:
                    ori_attn_o_stat = ymax
                if ori_attn_o_stat < ymin:
                    ori_attn_o_stat = ymin

                all_axes[layer_idx].annotate(
                    "", 
                    xy=(token_id, pruned_attn_o_stat), 
                    arrowprops=dict(facecolor='black', arrowstyle='->'),
                    xytext=(token_id, ori_attn_o_stat),
                )

            all_axes[layer_idx].set_title(f"Layer {layer_idx}")
            all_axes[layer_idx].set_xlabel("Token ID")
            all_axes[layer_idx].set_ylabel(f"Attn Output {stat_name}")
            all_axes[layer_idx].set_xlim(0, num_tokens)
            all_axes[layer_idx].set_ylim(ymin, ymax)

        fig.savefig(attn_o_stat_path / f"attn_o_{stat_name}.png", dpi=260, bbox_inches='tight', pad_inches=0.1)


# ---------------------------------------------------------------------------
# Input feature extraction
# ---------------------------------------------------------------------------

def extract_row_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Given a 2-D array x of shape [N, D], compute a comprehensive set of
    per-row statistical and geometric features.

    Returns
    -------
    features : np.ndarray of shape [N, F]
    feature_names : list[str] of length F
    """
    feature_fns: list[tuple[str, callable]] = [
        ("mean",        lambda v: np.mean(v, axis=1)),
        ("std",         lambda v: np.std(v, axis=1)),
        ("variance",    lambda v: np.var(v, axis=1)),
        ("median",      lambda v: np.median(v, axis=1)),
        ("max",         lambda v: np.max(v, axis=1)),
        ("min",         lambda v: np.min(v, axis=1)),
        ("range",       lambda v: np.max(v, axis=1) - np.min(v, axis=1)),
        ("q25",         lambda v: np.percentile(v, 25, axis=1)),
        ("q75",         lambda v: np.percentile(v, 75, axis=1)),
        ("iqr",         lambda v: np.percentile(v, 75, axis=1) - np.percentile(v, 25, axis=1)),
        ("q10",         lambda v: np.percentile(v, 10, axis=1)),
        ("q90",         lambda v: np.percentile(v, 90, axis=1)),
        ("l1_norm",     lambda v: np.sum(np.abs(v), axis=1)),
        ("l2_norm",     lambda v: np.linalg.norm(v, axis=1)),
        ("linf_norm",   lambda v: np.max(np.abs(v), axis=1)),
        ("skewness",    lambda v: _safe_skewness(v)),
        ("kurtosis",    lambda v: _safe_kurtosis(v)),
        ("rms",         lambda v: np.sqrt(np.mean(v ** 2, axis=1))),
        ("abs_mean",    lambda v: np.mean(np.abs(v), axis=1)),
        ("pos_frac",    lambda v: np.mean(v > 0, axis=1)),
        ("neg_frac",    lambda v: np.mean(v < 0, axis=1)),
        ("zero_frac",   lambda v: np.mean(v == 0, axis=1)),
        ("energy",      lambda v: np.sum(v ** 2, axis=1)),
        ("mean_abs_dev",lambda v: np.mean(np.abs(v - np.mean(v, axis=1, keepdims=True)), axis=1)),
        ("cv",          lambda v: _safe_cv(v)),          # coefficient of variation
        ("norm_entropy",lambda v: _safe_norm_entropy(v)),
    ]

    feature_cols = [fn(x) for _, fn in feature_fns]
    feature_names = [name for name, _ in feature_fns]
    features = np.stack(feature_cols, axis=1)   # [N, F]
    return features, feature_names


def _safe_skewness(v: np.ndarray) -> np.ndarray:
    """Per-row skewness (Fisher's definition, unbiased)."""
    from scipy.stats import skew
    return skew(v, axis=1, bias=False)


def _safe_kurtosis(v: np.ndarray) -> np.ndarray:
    """Per-row excess kurtosis (Fisher's definition)."""
    from scipy.stats import kurtosis
    return kurtosis(v, axis=1, fisher=True, bias=False)


def _safe_cv(v: np.ndarray) -> np.ndarray:
    """Coefficient of variation = std / |mean|  (nan → 0)."""
    mu = np.mean(v, axis=1)
    sd = np.std(v, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = np.where(np.abs(mu) > 1e-12, sd / np.abs(mu), 0.0)
    return cv


def _safe_norm_entropy(v: np.ndarray) -> np.ndarray:
    """
    Normalised spectral/weight entropy of the absolute values in each row,
    treating |v_i| / sum(|v_i|) as a probability mass function.
    Rows that are all-zero get entropy 0.
    """
    abs_v = np.abs(v)
    row_sum = abs_v.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(row_sum > 0, abs_v / row_sum, 0.0)
        log_p = np.where(p > 0, np.log(p + 1e-30), 0.0)
    raw_entropy = -np.sum(p * log_p, axis=1)
    max_entropy = np.log(v.shape[1] + 1e-30)
    return raw_entropy / max_entropy


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def compute_pearson_correlation(
    features: np.ndarray,
    feature_names: list[str],
    targets: np.ndarray,   # [N, 2]  col-0 = output_mean, col-1 = output_std
) -> pd.DataFrame:
    """
    Compute Pearson r and p-value between every input feature and each target.

    Returns a tidy DataFrame with columns:
        feature, target, pearson_r, p_value
    """
    records = []
    target_names = ["attn_outs_mean", "attn_outs_std"]
    for t_idx, t_name in enumerate(target_names):
        y = targets[:, t_idx]
        for f_idx, f_name in enumerate(feature_names):
            x = features[:, f_idx]
            # Drop NaN rows for this pair
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3:
                r, p = np.nan, np.nan
            else:
                r, p = pearsonr(x[mask], y[mask])
            records.append({"feature": f_name, "target": t_name,
                            "pearson_r": r, "p_value": p})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Heatmap visualisation
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    title: str = "Pearson r – Input Features vs. Attn Output",
    save_path: Path | None = None,
):
    """
    Plot a heatmap where:
      - x-axis : input feature names
      - y-axis : [attn_outs_mean, attn_outs_std]

    Cells are annotated with the Pearson r value.
    """
    pivot = corr_df.pivot(index="target", columns="feature", values="pearson_r")
    # fix row order
    row_order = ["attn_outs_mean", "attn_outs_std"]
    pivot = pivot.reindex([r for r in row_order if r in pivot.index])

    n_features = pivot.shape[1]
    fig_w = max(14, n_features * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, 3.5))

    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

    ax.set_xticks(range(n_features))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title(title, fontsize=11, pad=10)

    # annotate each cell
    for row_i, row_label in enumerate(pivot.index):
        for col_i, col_label in enumerate(pivot.columns):
            val = pivot.loc[row_label, col_label]
            if np.isfinite(val):
                ax.text(col_i, row_i, f"{val:.2f}",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if abs(val) > 0.6 else "black")

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Heatmap saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def load_all_data(base_path: Path, model_name: str) -> dict:
    """
    Load training data (both train-split and test-split) for all layers,
    concatenating them so no samples are discarded.

    Returns
    -------
    dict with keys:
        n_layers   : list of layer key strings
        hidden_dim : int
        attn_ins   : dict[layer_key -> np.ndarray [N, D]]
        attn_outs  : dict[layer_key -> np.ndarray [N, 2]]  (mean, std)
    """
    # ── paths (mirrors load_training_data) ──────────────────────────
    # input_path       = base_path / f"{model_name}_vmatrix.npy"
    input_path       = base_path / f"{model_name}_input.npy"
    output_mean_path = base_path / f"{model_name}_output_mean.npy"
    output_std_path  = base_path / f"{model_name}_output_std.npy"

    input_hstates = np.load(input_path,       allow_pickle=True).item()
    output_mean   = np.load(output_mean_path, allow_pickle=True).item()
    output_std    = np.load(output_std_path,  allow_pickle=True).item()

    n_layers   = list(input_hstates.keys())
    hidden_dim = input_hstates[n_layers[0]][0].shape[-1]
    print(f"[load_all_data] {len(n_layers)} layers | hidden_dim={hidden_dim}")

    attn_ins_all  = {}
    attn_outs_all = {}

    for l in n_layers:
        # concatenate ALL iterations (no train/test split)
        x    = np.concatenate(input_hstates[l], axis=0)          # [N, D]
        y_mu = np.concatenate(output_mean[l],   axis=0)          # [N, 1]
        y_sd = np.concatenate(output_std[l],    axis=0)          # [N, 1]
        y    = np.concatenate([y_mu, y_sd], axis=-1)             # [N, 2]

        attn_ins_all[l]  = x
        attn_outs_all[l] = y

        print(f"  {l}: ins {x.shape}, outs {y.shape}")

    return {
        "n_layers":  n_layers,
        "hidden_dim": hidden_dim,
        "attn_ins":  attn_ins_all,
        "attn_outs": attn_outs_all,
    }


def analyze_input_output_correlation(
    base_path: str | Path,
    model_name: str,
    out_dir: str | Path | None = None,
):
    """
    Full pipeline:
      1. Load data (all samples, no train/test distinction).
      2. Extract per-row features for each layer's attn_ins.
      3. Compute Pearson r and p-value against attn_outs (mean & std).
      4. Save per-layer CSV and aggregated CSV.
      5. Plot per-layer heatmaps + an aggregated heatmap (r averaged over layers).

    Args
    ----
    base_path  : directory containing the .npy data files.
    model_name : prefix used when the .npy files were saved.
    out_dir    : where to write CSVs and PNG figures.
                 Defaults to  <base_path>/corr_analysis/
    """
    base_path = Path(base_path)
    out_dir   = Path(out_dir) if out_dir else base_path / "corr_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load ----------------------------------------------------------------
    data = load_all_data(base_path, model_name)
    n_layers  = data["n_layers"]
    attn_ins  = data["attn_ins"]
    attn_outs = data["attn_outs"]

    all_layer_dfs = []

    for l in n_layers:
        print(f"\n── {l} ──")

        x = attn_ins[l]   # [N, D]  numpy
        y = attn_outs[l]  # [N, 2]  numpy

        # 2. Feature extraction ----------------------------------------------
        features, feature_names = extract_row_features(x)
        print(f"  Extracted {len(feature_names)} features from {x.shape[0]} samples.")

        # 3. Pearson correlation ---------------------------------------------
        corr_df = compute_pearson_correlation(features, feature_names, y)
        corr_df.insert(0, "layer", l)

        # 4. Save per-layer CSV ----------------------------------------------
        csv_path = out_dir / f"{model_name}_{l}_corr.csv"
        corr_df.to_csv(csv_path, index=False)
        print(f"  Correlation table saved → {csv_path}")

        all_layer_dfs.append(corr_df)

        # 5. Per-layer heatmap -----------------------------------------------
        heatmap_path = out_dir / f"{model_name}_{l}_corr_heatmap.png"
        plot_correlation_heatmap(
            corr_df,
            title=f"Pearson r  [{model_name}  {l}]",
            save_path=heatmap_path,
        )

    # ── 6. Aggregated (mean |r| across layers) ─────────────────────────────
    agg_df_full = pd.concat(all_layer_dfs, ignore_index=True)
    agg_csv_path = out_dir / f"{model_name}_all_layers_corr.csv"
    agg_df_full.to_csv(agg_csv_path, index=False)
    print(f"\nAggregated correlation table saved → {agg_csv_path}")

    # Average |r| across layers for the aggregated heatmap
    agg_mean = (
        agg_df_full
        .groupby(["target", "feature"], sort=False)["pearson_r"]
        .mean()
        .reset_index()
        .rename(columns={"pearson_r": "pearson_r"})
    )
    agg_heatmap_path = out_dir / f"{model_name}_all_layers_corr_heatmap.png"
    plot_correlation_heatmap(
        agg_mean,
        title=f"Pearson r (avg over all layers)  [{model_name}]",
        save_path=agg_heatmap_path,
    )

    print("\nDone.")
    return agg_df_full


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse Pearson correlation between attn_ins features and attn_outs."
    )
    parser.add_argument("--model_name",     required=True,
                        help="Model name prefix used when saving .npy files.")
    parser.add_argument("--data_base_path", required=True,
                        help="Directory containing the .npy data files.")
    parser.add_argument("--out_dir",        default=None,
                        help="Output directory for CSVs and heatmap PNGs. "
                             "Defaults to <data_base_path>/corr_analysis/")
    args = parser.parse_args()

    analyze_input_output_correlation(
        base_path  = args.data_base_path,
        model_name = args.model_name,
        out_dir    = args.out_dir,
    )
