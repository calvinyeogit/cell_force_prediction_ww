#!/usr/bin/env python
"""visualize_skip_connection.py

Visualizes how a U-Net model fuses information at a skip connection by
plotting feature maps from the encoder, the upsampled decoder path, and the
fused result after concatenation.

Example usage
-------------
python visualize_skip_connection.py \
    --model_path model_23JUNE.pt \
    --sample_idx 3
"""

import argparse
import os
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from utils.UNeXt import UNet
from utils.data_processing import CellDataset

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_dataset(modelinfo: Dict, root_override: Optional[str] = None) -> CellDataset:
    """Re-create the dataset used for training, with optional overrides."""
    dkwargs = modelinfo["dataset_kwargs"].copy()
    if root_override is not None:
        dkwargs["root"] = root_override
    return CellDataset(**dkwargs)


def _load_model(modelinfo: Dict, device: torch.device) -> UNet:
    """Loads the U-Net model from a checkpoint."""
    model = UNet(**modelinfo["model_kwargs"], model_idx=0)
    model.load_state_dict(modelinfo["model"])
    model.eval()
    model.to(device)
    return model


def collect_activations(model: torch.nn.Module, x: torch.Tensor, target_layers: List[str]) -> Dict[str, torch.Tensor]:
    """
    Registers forward hooks on target layers and runs a forward pass to
    collect their activations.
    """
    activations: Dict[str, torch.Tensor] = {}
    hooks = []

    def save_activation(name: str):
        def hook(_, __, output):
            activations[name] = output.detach().cpu()
        return hook

    for name, module in model.named_modules():
        if name in target_layers:
            hooks.append(module.register_forward_hook(save_activation(name)))

    with torch.no_grad():
        _ = model(x)

    for hook in hooks:
        hook.remove()

    return activations


def plot_skip_connection(
    activations: Dict[str, torch.Tensor],
    out_path: str,
    target_layers: List[str],
    cmap: str = "viridis",
):
    """
    Generates and saves a single comprehensive plot showing feature maps
    from the three target layers involved in a skip connection.
    """
    titles = {
        target_layers[0]: "Encoder Output (All Channels from layers_encode.2.3)",
        target_layers[1]: "Upsampled Decoder Input (All Channels from layers_decode.1.0)",
        target_layers[2]: "Fused Result (All Channels from layers_decode.1.1.0)",
    }

    # Calculate the number of rows needed for each layer's feature maps
    subplot_rows = {}
    for layer_name in target_layers:
        fmap = activations[layer_name]
        n_channels = fmap.shape[1]  # (B, C, H, W) -> C
        ncols = 8  # Fixed number of columns for channel plots
        nrows = math.ceil(n_channels / ncols)
        subplot_rows[layer_name] = nrows

    total_subplot_rows = sum(subplot_rows.values())
    
    # Create a figure with enough space for all plots and 3 titles
    fig = plt.figure(figsize=(20, 2 * total_subplot_rows + 4))
    
    # Main GridSpec for the 3 sections
    main_gs = GridSpec(3, 1, figure=fig, height_ratios=[subplot_rows[ln] for ln in target_layers], hspace=0.4)

    for i, layer_name in enumerate(target_layers):
        fmap = activations[layer_name].squeeze(0) # -> (C, H, W)
        n_channels = fmap.shape[0]
        ncols = 8
        
        # Nested GridSpec for the current section (title + channels)
        nested_gs = GridSpecFromSubplotSpec(subplot_rows[layer_name] + 1, ncols, subplot_spec=main_gs[i], hspace=0.3)
        
        # Section title
        title_ax = fig.add_subplot(nested_gs[0, :])
        title_ax.set_title(titles[layer_name], fontsize=16, weight='bold')
        title_ax.axis('off')

        # Channel subplots
        for ch in range(n_channels):
            ax = fig.add_subplot(nested_gs[1 + ch // ncols, ch % ncols])
            ax.imshow(fmap[ch], cmap=cmap)
            ax.set_title(f"ch {ch}", fontsize=8)
            ax.axis("off")
        
        # Hide unused subplots in the nested grid
        for ch_idx in range(n_channels, subplot_rows[layer_name] * ncols):
             ax = fig.add_subplot(nested_gs[1 + ch_idx // ncols, ch_idx % ncols])
             ax.axis('off')


    # No tight_layout here, as GridSpec handles it.
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved skip connection visualization to {out_path}")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize feature map fusion at a U-Net skip connection."
    )
    parser.add_argument(
        "--model_path", default="model_23JUNE.pt", help="Path to the saved model checkpoint (.pt)"
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index of the dataset sample to visualise",
    )
    parser.add_argument(
        "--out_dir", default="skip_connection_visuals", help="Directory to write PNGs"
    )
    parser.add_argument(
        "--device",
        default="cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="Device to run inference on",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1.  Load checkpoint and reconstruct model/dataset
    # ---------------------------------------------------------------------
    print(f"[INFO] Loading model from {args.model_path}...")
    modelinfo = torch.load(args.model_path, map_location="cpu")
    model = _load_model(modelinfo, device=device)
    dataset = _build_dataset(modelinfo)

    # ---------------------------------------------------------------------
    # 2.  Prepare the input sample
    # ---------------------------------------------------------------------
    print(f"[INFO] Using data sample index {args.sample_idx}...")
    sample = dataset[args.sample_idx]
    for key in sample:
        sample[key] = sample[key].unsqueeze(0).to(device)

    x = model.select_inputs(model.input_type, sample)

    # ---------------------------------------------------------------------
    # 3.  Collect activations from target layers
    # ---------------------------------------------------------------------
    target_layers = [
        "layers_encode.2.3",      # Encoder output (spatial detail)
        "layers_decode.1.0",      # Upsampled output (semantic context)
        "layers_decode.1.1.0",    # Fused result after ConvNextCell
    ]
    print(f"[INFO] Collecting activations from: {target_layers}...")
    activations = collect_activations(model, x, target_layers)

    # ---------------------------------------------------------------------
    # 4.  Plot and save the results
    # ---------------------------------------------------------------------
    output_filename = f"skip_connection_fusion_all_channels_sample_{args.sample_idx}.png"
    output_path = os.path.join(args.out_dir, output_filename)
    
    # Ensure the layers are plotted in the specified order
    plot_skip_connection(activations, output_path, target_layers)

if __name__ == "__main__":
    main() 