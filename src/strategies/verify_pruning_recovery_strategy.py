# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
VerifyPruningRecoveryStrategy
==============================
Oracle upper-bound verification for attention-output distribution recovery.

For each layer (layer_idx > 0):
  1. Run the original full attention (flash_attention_2 or sdpa) to get the
     ground-truth attention output and capture its row-wise mean and std.
  2. Run the pruned attention (EfficientKascadeStrategy) on the same inputs.
  3. Shift the pruned output to match the ground-truth mean/std using the same
     normalise-then-rescale formula used in apply_mlp_recovery().
  4. Print diagnostics: max / min / avg absolute difference between the
     ground-truth mean/std and the shifted output's mean/std.

This gives the theoretical upper bound on how much distribution recovery can
help, assuming a perfect mean/std predictor.

Memory notes
------------
- The full-attention first pass MUST use flash attention (FA3 > FA2) to avoid OOM.
  We fall back to "sdpa" if neither FA variant is registered.
- The pruned second pass uses the EfficientKascadeStrategy kernel path.
- transformers' _flash_attention_forward reads attn_implementation from the model
  config and has no else-branch, so we temporarily override it to the resolved FA
  name during the GT pass to avoid an UnboundLocalError.
"""

from .EfficientKascadeStrategy import EfficientKascadeStrategy
from typing import List, Optional
import torch
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def _get_full_attn_fn():
    """Return the best available full-attention function (FA2 > sdpa)."""
    for name in ("flash_attention_2", "sdpa"):
        if name in ALL_ATTENTION_FUNCTIONS:
            return ALL_ATTENTION_FUNCTIONS[name], name
    raise RuntimeError("No compatible full-attention implementation found.")


class VerifyPruningRecoveryStrategy(EfficientKascadeStrategy):
    """
    Verifies oracle upper bound of distribution recovery.

    Inherits all efficient_kascade kernel machinery.  Overrides
    ``attention_forward`` to run a full-attention pass first (for ground-truth
    statistics) and then the pruned pass, shifting the pruned output to the
    ground-truth distribution.
    """

    def __init__(
        self,
        recompute_layers: List[int],
        model_name: str,
        name: str = "verify_pruning_recovery",
        k: float = 1,
        tile_size: int = 1,
        rolling_prefill: bool = False,
        block_size: int = 12288,
    ):
        super().__init__(
            recompute_layers=recompute_layers,
            model_name=model_name,
            name=name,
            k=k,
            tile_size=tile_size,
            rolling_prefill=rolling_prefill,
            block_size=block_size,
        )
        self._full_attn_fn, self._full_attn_name = _get_full_attn_fn()
        print(f"[VerifyPruningRecoveryStrategy] Using '{self._full_attn_name}' for full-attention ground-truth pass.")

    # ------------------------------------------------------------------
    # Core override
    # ------------------------------------------------------------------

    def attention_forward(
        self,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        """
        For layer 0: behave exactly like EfficientKascadeStrategy (recompute
        pass to populate topk indices, no shifting).

        For layer > 0:
          1. Full-attention pass → ground-truth mean/std.
          2. Pruned-attention pass (EfficientKascadeStrategy).
          3. Shift pruned output → ground-truth distribution.
          4. Print diagnostics.
        """
        layer_idx = module.layer_idx

        # ---------------------------------------------------------------
        # Layer 0: just run the efficient kascade recompute pass as usual.
        # ---------------------------------------------------------------
        if layer_idx == 0:
            return super().attention_forward(
                module, query, key, value, attention_mask, scaling,
                dropout=dropout, **kwargs
            )

        # ---------------------------------------------------------------
        # Layer > 0: two-pass approach.
        # ---------------------------------------------------------------

        # --- Pass 1: full attention for ground truth (no_grad, in-place safe) ---
        # transformers' _flash_attention_forward reads `attn_implementation` from
        # the model config and uses it to select FA2 vs FA3 internals via a bare
        # if/elif with no else branch.  If the value isn't one of the three known
        # strings the local variable `_is_fa3` is never assigned, causing an
        # UnboundLocalError.  Temporarily set the config to "flash_attention_2"
        # so the correct branch fires, then restore our strategy name afterwards.
        _saved_attn_impl = module.config._attn_implementation
        module.config._attn_implementation = self._full_attn_name
        with torch.no_grad():
            gt_out, _ = self._full_attn_fn(
                module, query, key, value, attention_mask,
                scaling=scaling, dropout=0.0, **kwargs
            )
        module.config._attn_implementation = _saved_attn_impl

        # gt_out shape: (B, Lq, H, D)  or  (B, H, Lq, D) depending on impl.
        # All HF attention functions return (B, Lq, H*D) after transpose+reshape,
        # but the raw attention function output before o_proj is (B, Lq, H, D).
        # Here it's the raw before-o_proj output returned by the attention fn:
        # shape (B, Lq, H, D) → we treat it as (..., D) for row-wise stats.
        gt_flat = gt_out.reshape(-1, gt_out.shape[-1]).to(torch.float32)
        gt_mean = gt_flat.mean(dim=-1, keepdim=True)   # (N, 1)
        gt_std  = gt_flat.std(dim=-1, keepdim=True)    # (N, 1)

        # --- Pass 2: pruned attention ---
        pruned_out, attn_weights = super().attention_forward(
            module, query, key, value, attention_mask, scaling,
            dropout=dropout, **kwargs
        )

        # --- Shift pruned output to ground-truth distribution ---
        eps = 1e-8
        original_dtype = pruned_out.dtype
        pruned_flat = pruned_out.reshape(-1, pruned_out.shape[-1]).to(torch.float32)

        curr_mu  = pruned_flat.mean(dim=-1, keepdim=True)
        curr_std = pruned_flat.std(dim=-1, keepdim=True)

        # Cast ground truth to the same device/shape as pruned
        gt_mean_dev = gt_mean.to(pruned_flat.device)
        gt_std_dev  = gt_std.to(pruned_flat.device)

        # Normalise then rescale: (x - mu) / (sigma + eps) * gt_sigma + gt_mu
        pruned_flat = (pruned_flat - curr_mu) / (curr_std + eps) * gt_std_dev + gt_mean_dev

        shifted_out = pruned_flat.reshape_as(pruned_out.float()).to(original_dtype)

        # --- Diagnostics ---
        with torch.no_grad():
            shifted_flat = shifted_out.reshape(-1, shifted_out.shape[-1]).to(torch.float32)
            shifted_mean = shifted_flat.mean(dim=-1, keepdim=True)
            shifted_std  = shifted_flat.std(dim=-1, keepdim=True)

            mean_diff = (shifted_mean - gt_mean_dev).abs()
            std_diff  = (shifted_std  - gt_std_dev).abs()

            # print(
            #     f"[Layer {layer_idx:02d}] "
            #     f"mean_diff  max={mean_diff.max().item():.6f}  "
            #     f"min={mean_diff.min().item():.6f}  "
            #     f"avg={mean_diff.mean().item():.6f} | "
            #     f"std_diff  max={std_diff.max().item():.6f}  "
            #     f"min={std_diff.min().item():.6f}  "
            #     f"avg={std_diff.mean().item():.6f}"
            # )

        return shifted_out, attn_weights

    def register_attention(self):
        """
        Register under a unique name so the model config can route to this
        strategy's attention_forward.  Reuses EfficientKascadeStrategy's
        mask-attention registration for masks, but points the compute function
        to our overridden attention_forward.
        """
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

        _attention_forward = (
            lambda module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs:
            self.attention_forward(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
        )
        ALL_MASK_ATTENTION_FUNCTIONS.register(self.name, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
        ALL_ATTENTION_FUNCTIONS.register(self.name, _attention_forward)
