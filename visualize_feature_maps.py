#!/usr/bin/env python
"""visualize_feature_maps.py

Visualize intermediate feature maps of a trained U-Net checkpoint.

Example usage
-------------
python visualize_feature_maps.py \
    --model_path model_23JUNE.pt \
    --sample_idx 3 \
    --out_dir feature_maps \
    --layer_filter "encoder"

The script has two back-ends:
* torchview.FeatureMapVisualizer – preferred when the optional torchview
  dependency is available. Produces a single PNG summarising all feature maps.
* A lightweight manual hook implementation (always available) that can also be
  filtered to specific layers by name and stores each layer as a separate PNG.

This keeps heavy visualisation logic out of `load_trained_unet.py` so that file
remains focused on inference / evaluation.
"""

import argparse
import os
import math
from typing import Dict, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.UNeXt import UNet
from utils.data_processing import CellDataset

# -------------------------------------------------------------
# Optional dependency --------------------------------------------------------
# -------------------------------------------------------------
try:
    from torchview import FeatureMapVisualizer  # type: ignore

    _HAS_TORCHVIEW = True
except ImportError:  # pragma: no cover – optional
    FeatureMapVisualizer = None  # type: ignore
    _HAS_TORCHVIEW = False


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_dataset(modelinfo: Dict, root_override: Optional[str] = None, crop_size: Optional[int] = None) -> CellDataset:
    """Re-create the dataset used for training, with optional overrides."""
    dkwargs = modelinfo["dataset_kwargs"].copy()
    if root_override is not None:
        dkwargs["root"] = root_override
    if crop_size is not None and "transform_kwargs" in dkwargs:
        dkwargs["transform_kwargs"]["crop_size"] = crop_size
    return CellDataset(**dkwargs)


def _load_model(modelinfo: Dict, device: torch.device) -> UNet:
    model = UNet(**modelinfo["model_kwargs"], model_idx=0)
    model.load_state_dict(modelinfo["model"])
    model.eval()
    model.to(device)
    return model


def _collect_activations(model: torch.nn.Module, x: torch.Tensor, layer_filter: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """Forward *x* through *model* while capturing feature maps via hooks."""
    activations: Dict[str, torch.Tensor] = {}

    def save_activation(name: str):
        def hook(_, __, output):
            activations[name] = output.detach().cpu()

        return hook

    # Register hooks ----------------------------------------------------------------
    for name, module in model.named_modules():
        if layer_filter is None or layer_filter in name:
            module.register_forward_hook(save_activation(name))

    with torch.no_grad():
        _ = model(x)  # populate *activations*

    return activations


def _plot_feature_maps(
    activations: Dict[str, torch.Tensor],
    out_dir: str,
    ncols: int = 8,
    cmap: str = "viridis",
    top_k: Optional[int] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    for layer_name, fmap in activations.items():  # fmap: (B, C, H, W)
        fmap = fmap.squeeze(0)  # -> (C, H, W)

        if top_k is not None:
            # Select top-k channels based on mean activation
            mean_activations = fmap.mean(dim=[1, 2])
            top_indices = torch.topk(mean_activations, k=min(top_k, fmap.shape[0])).indices
            fmap = fmap[top_indices]
            plot_title = f"{layer_name} (Top {top_k} activations)"
        else:
            plot_title = layer_name

        n_channels = fmap.shape[0]
        nrows = math.ceil(n_channels / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
        axes = np.atleast_1d(axes).flatten()

        for ch in range(n_channels):
            axes[ch].imshow(fmap[ch], cmap=cmap)
            axes[ch].set_title(f"ch {ch}", fontsize=6)
            axes[ch].axis("off")

        # Hide any unused subplots
        for ax in axes[n_channels:]:
            ax.axis("off")

        plt.suptitle(plot_title, fontsize=10)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        fig.savefig(
            os.path.join(out_dir, f"{layer_name.replace('.', '_')}.png"), dpi=150
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize feature maps of a trained U-Net model."
    )
    parser.add_argument(
        "--model_path", default="model_23JUNE.pt", help="Path to the saved model checkpoint (.pt)"
    )
    parser.add_argument(
        "--root", default=None, help="Override dataset root directory if desired"
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index of the dataset sample to visualise",
    )
    parser.add_argument(
        "--out_dir", default="feature_maps", help="Directory to write PNGs"
    )
    parser.add_argument(
        "--device",
        default="cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="Device to run inference on",
    )
    parser.add_argument(
        "--layer_filter",
        default=None,
        help="Substring to filter layer names (e.g. 'encoder')",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=None,
        help="Override crop_size used during training",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Visualize only the top-K most activated channels for each layer",
    )
    parser.add_argument(
        "--list_layers",
        action="store_true",
        help="List all model layer names and exit.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---------------------------------------------------------------------
    # 1.  Load checkpoint and reconstruct model/dataset --------------------
    # ---------------------------------------------------------------------
    modelinfo = torch.load(args.model_path, map_location="cpu")
    model = _load_model(modelinfo, device=device)

    if args.list_layers:
        print("Available layers in the model:")
        for name, module in model.named_modules():
            print(f"- {name} ({module.__class__.__name__})")
        return

    dataset = _build_dataset(
        modelinfo, root_override=args.root, crop_size=args.crop_size
    )

    # ---------------------------------------------------------------------
    # 2.  Prepare the input sample ----------------------------------------
    # ---------------------------------------------------------------------
    sample = dataset[args.sample_idx]
    for key in sample:
        sample[key] = sample[key].unsqueeze(0).to(device)

    x = model.select_inputs(model.input_type, sample)

    # ---------------------------------------------------------------------
    # 3a. torchview back-end (if available) --------------------------------
    # ---------------------------------------------------------------------
    if _HAS_TORCHVIEW and args.top_k is None:
        print(
            "[INFO] torchview detected – generating unified feature-map visual..."
        )
        vis = FeatureMapVisualizer(
            model, input_shape=tuple(x.shape), layer_order="breadth"
        )  # type: ignore
        vis.visualize()  # internally calls forward pass
        os.makedirs(args.out_dir, exist_ok=True)
        vis.save_graph(
            os.path.join(args.out_dir, "torchview_feature_maps.png")
        )
        # Fall through – user may still want per-layer PNGs below if a filter is set
        if args.layer_filter is None:
            print(
                f"[INFO] Saved torchview visual to {args.out_dir}/torchview_feature_maps.png"
            )
            return

    # ---------------------------------------------------------------------
    # 3b. Manual hook back-end --------------------------------------------
    # ---------------------------------------------------------------------
    if args.top_k is not None:
        print(f"[INFO] Finding top {args.top_k} activations...")
    print("[INFO] Collecting activations via forward hooks...")
    activations = _collect_activations(model, x, layer_filter=args.layer_filter)
    print(f"[INFO] Saving {len(activations)} layers to '{args.out_dir}'")
    _plot_feature_maps(activations, args.out_dir, top_k=args.top_k)


if __name__ == "__main__":
    main() 