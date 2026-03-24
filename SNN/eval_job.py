"""eval_job.py

Evaluate one trained checkpoint on the test dataset and produce:

  Metrics (stdout + JSON):
    - RMSE, MAE, mean relative error in native units (cells for spatial, log10 for energy)
    - Same in mm for spatial outputs  (1 cell = 3 mm along x/y, 12 mm along z)
    - For energy outputs: also RMSE, MAE, relative error on linear E = 10^(log10_E)

  Plots (saved as png under PLOTS_DIR/<checkpoint_stem>/):
    loss_curves.png               -- training loss (per batch) and validation loss (per epoch)
    hist2d.png                    -- 2D histogram pred vs target, native units
    residuals.png                 -- residual distribution, native units
    rel_error_profile.png         -- mean relative error vs binned target, native units
    abs_error_profile.png         -- mean absolute error vs binned target, native units
    hist2d_mm.png                 -- same 2D histogram in mm           (spatial tasks)
    residuals_mm.png              -- residuals in mm                   (spatial tasks)
    rel_error_profile_mm.png      -- relative error profile in mm      (spatial tasks)
    abs_error_profile_mm.png      -- absolute error profile in mm      (spatial tasks)
    hist2d_energy_linear.png      -- 2D histogram for E in GeV         (energy tasks)
    residuals_energy_linear.png   -- residuals for E in GeV            (energy tasks)
    rel_error_profile_energy.png  -- relative error profile for E      (energy tasks)
    abs_error_profile_energy.png  -- absolute error profile for E      (energy tasks)

Usage
-----
    python eval_job.py --ckpt_path ./checkpoints/energy_spike.pt
    python eval_job.py --ckpt_path ./checkpoints/centroid_x_membrane.pt
    python eval_job.py --ckpt_path ./checkpoints/energy_centroid_ann.pt
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Condor nodes
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from torch.utils.data import Dataset

import snntorch as snn
from snntorch import surrogate

import dataset as ds
import SNN_func as snnfn


# ── Configuration ──────────────────────────────────────────────────────────────
DATA_PATH  = "/lustre/ific.uv.es/ml/uovi123/snncalo/PhotonData/Reflections"
MAX_FILES  = 50
BATCH_SIZE = 64
PLOTS_DIR  = "./results_full/Refl"
EVAL_SEED  = 42       # fixed seed for reproducible train/val/test split

POPULATION = 20
INPUT_SIZE = ds.nSensors * 4   # 100 sensors × 4 thresholds = 400

# Cell size in mm along each axis (linear quantities: centroids)
_CELL_MM = {"x": 3.0, "y": 3.0, "z": 12.0}
# Cell size squared in mm² (quadratic quantities: dispersions = energy-weighted σ²)
_CELL_MM2 = {k: v ** 2 for k, v in _CELL_MM.items()}


# ── Task metadata ──────────────────────────────────────────────────────────────
# Tuple: (needs_E, needs_C, needs_D, comp_idx, n_tasks, output_labels, cell_sizes_mm)
#
# output_labels  : human-readable name for each output dimension
# cell_sizes_mm  : mm per cell for each output (None = not a spatial quantity)
#
# Spatial outputs are stored in cell units by the dataset; multiply by cell_sizes_mm
# to obtain mm.  Cell size: 3 mm along x/y, 12 mm along z.
# _TASK_META tuple layout:
#   (needs_E, needs_C, needs_D, comp_idx, n_tasks, output_labels, cell_sizes_mm, mm_output_labels)
#
# output_labels    : LaTeX label per network output (native units)
# cell_sizes_mm    : mm (or mm²) conversion factor per output; None for non-spatial outputs
# mm_output_labels : LaTeX labels for the mm-converted spatial outputs (same order as spatial_idx)
_TASK_META = {
    "energy":            (True,  False, False, None, 1,
                          [r"$\log(E\,/\,\mathrm{MeV})$"],
                          [None],
                          []),
    "centroid_x":        (False, True,  False, 0,    1,
                          [r"$x_c$"],
                          [_CELL_MM["x"]],
                          [r"$x_c$ [mm]"]),
    "centroid_y":        (False, True,  False, 1,    1,
                          [r"$y_c$"],
                          [_CELL_MM["y"]],
                          [r"$y_c$ [mm]"]),
    "centroid_z":        (False, True,  False, 2,    1,
                          [r"$z_c$"],
                          [_CELL_MM["z"]],
                          [r"$z_c$ [mm]"]),
    "dispersion_x":      (False, False, True,  0,    1,
                          [r"$\sigma^2_x$"],
                          [_CELL_MM2["x"]],
                          [r"$\sigma^2_x\ [\mathrm{mm}^2]$"]),
    "dispersion_y":      (False, False, True,  1,    1,
                          [r"$\sigma^2_y$"],
                          [_CELL_MM2["y"]],
                          [r"$\sigma^2_y\ [\mathrm{mm}^2]$"]),
    "dispersion_z":      (False, False, True,  2,    1,
                          [r"$\sigma^2_z$"],
                          [_CELL_MM2["z"]],
                          [r"$\sigma^2_z\ [\mathrm{mm}^2]$"]),
    "energy_centroid":   (True,  True,  False, None, 4,
                          [r"$\log(E\,/\,\mathrm{MeV})$",
                           r"$x_c$", r"$y_c$", r"$z_c$"],
                          [None, _CELL_MM["x"], _CELL_MM["y"], _CELL_MM["z"]],
                          [r"$x_c$ [mm]", r"$y_c$ [mm]", r"$z_c$ [mm]"]),
    "energy_dispersion": (True,  False, True,  None, 4,
                          [r"$\log(E\,/\,\mathrm{MeV})$",
                           r"$\sigma^2_x$", r"$\sigma^2_y$", r"$\sigma^2_z$"],
                          [None, _CELL_MM2["x"], _CELL_MM2["y"], _CELL_MM2["z"]],
                          [r"$\sigma^2_x\ [\mathrm{mm}^2]$",
                           r"$\sigma^2_y\ [\mathrm{mm}^2]$",
                           r"$\sigma^2_z\ [\mathrm{mm}^2]$"]),
}

# Labels that identify a log-energy output and centroid-in-cell outputs
_LOGE_LABEL    = r"$\log(E\,/\,\mathrm{MeV})$"
_CENTROID_LABS = {r"$x_c$", r"$y_c$", r"$z_c$"}

# Human-readable title for each output mode
_MODE_TITLE = {"spike": "Spike", "membrane": "Membrane", "ann": "ANN", "refl": "Spike"}

# Suptitle override for multi-target tasks per mode
_MULTI_SUPTITLE = {
    "refl": "Multi-target regression with reflections and attenuation",
}


# ── Spike generation ───────────────────────────────────────────────────────────
def spikegen_multi(data, multiplicity=4):
    """Encode photon counts into spikes with 4 fixed thresholds (10², 10³, 10⁴, 10⁵)."""
    og_shape   = data.shape
    spike_data = torch.zeros(og_shape[1], og_shape[0],
                             multiplicity * og_shape[2], device=data.device)
    for i in range(multiplicity):
        condition = data > np.power(10, i + 2)
        batch_idx, time_idx, sensor_idx = torch.nonzero(condition, as_tuple=True)
        spike_data[time_idx, batch_idx, multiplicity * sensor_idx + i] = 1
    return spike_data


# ── Network builders ───────────────────────────────────────────────────────────
def make_snn_net(output_size: int, output_mode: str) -> snnfn.Spiking_Net:
    leaky_params = dict(
        beta=0.9, learn_beta=True,
        threshold=1.0, learn_threshold=True,
        spike_grad=surrogate.atan(),
    )
    last_layer_params = leaky_params.copy()
    if output_mode == "membrane":
        last_layer_params.update(threshold=1e20, learn_threshold=False)
    neuron_params = {i: [snn.Leaky, leaky_params] for i in range(1, 4)}
    neuron_params[3] = [snn.Leaky, last_layer_params]
    net_desc = {
        "layers":        [INPUT_SIZE, 120, 120, output_size],
        "timesteps":     ds.timesteps,
        "output":        output_mode,
        "neuron_params": neuron_params,
    }
    return snnfn.Spiking_Net(net_desc, spikegen_multi)


class ANN(nn.Module):
    """Plain feed-forward network for calorimeter regression.

    Input: raw photon-count tensor of shape (B, T, nSensors).
    The time dimension is collapsed by summation before the dense layers.
    Output: (B, output_size).
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.net = nn.Sequential(
            nn.Linear(input_size,  hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        x = x.sum(dim=1).float()   # (B, T, nSensors) → (B, nSensors)
        x = self.bn(x)             # normalise to zero mean, unit variance
        return self.net(x)         # (B, output_size)


# ── Prediction helpers ─────────────────────────────────────────────────────────
def predict_spikefreq(output):
    """Average spike count over the population: (T, B, pop) → (B,)."""
    return output.sum(0).mean(1)


def predict_membrane(output):
    """Final-step membrane potential, averaged over population: (T, B, pop) → (B,)."""
    return output[-1].mean(1)


def absolute_error(prediction, targets):
    return torch.abs(targets - prediction)


# ── Dataset helpers ────────────────────────────────────────────────────────────
class ComponentDataset(Dataset):
    def __init__(self, base, idx):
        self.base = base
        self.idx  = idx

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        sample, target = self.base[i]
        return sample, target[self.idx]


class CombinedTargetDataset(Dataset):
    def __init__(self, ds1, ds2):
        assert len(ds1) == len(ds2)
        self.ds1 = ds1
        self.ds2 = ds2

    def __len__(self):
        return len(self.ds1)

    def __getitem__(self, i):
        sample, t1 = self.ds1[i]
        _,     t2  = self.ds2[i]
        t1 = t1.unsqueeze(0) if t1.dim() == 0 else t1
        t2 = t2.unsqueeze(0) if t2.dim() == 0 else t2
        return sample, torch.cat([t1, t2])


# ── Checkpoint name parser ─────────────────────────────────────────────────────
def parse_ckpt_name(ckpt_path: str):
    """Parse task_name and output_mode from the checkpoint filename.

    Expected patterns: <task>_spike.pt  /  <task>_membrane.pt  /  <task>_ann.pt
                       <task>_refl.pt   (spike net trained with reflections)
    A trailing _test suffix is stripped before parsing.

    Returns (task_name, output_mode) where output_mode ∈ {spike, membrane, ann, refl}.
    """
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    stem = re.sub(r'_test$', '', stem)
    idx  = stem.rfind('_')
    if idx == -1:
        raise ValueError(f"Cannot parse checkpoint name: {stem!r}")
    task_name   = stem[:idx]
    output_mode = stem[idx + 1:]
    if output_mode not in ("spike", "membrane", "ann", "refl"):
        raise ValueError(
            f"Unrecognised output mode {output_mode!r} in checkpoint: {stem!r}"
        )
    if task_name not in _TASK_META:
        raise ValueError(
            f"Unrecognised task {task_name!r} in checkpoint: {stem!r}"
        )
    return task_name, output_mode


# ── Build task dataset ─────────────────────────────────────────────────────────
def build_task_dataset(task_name: str):
    needs_E, needs_C, needs_D, comp_idx, n_tasks, *_ = _TASK_META[task_name]
    convert_to_log = lambda x: (x[0], torch.log10(x[1]))

    energy_data = ds.build_dataset(
        DATA_PATH, MAX_FILES, lazy=True, primary_only=True,
        target="energy", transform=convert_to_log,
    ) if needs_E else None

    centroid_data = ds.build_dataset(
        DATA_PATH, MAX_FILES, lazy=True, primary_only=True,
        target="centroid",
    ) if needs_C else None

    dispersion_data = ds.build_dataset(
        DATA_PATH, MAX_FILES, lazy=True, primary_only=True,
        target="dispersion",
    ) if needs_D else None

    if task_name == "energy_centroid":
        return CombinedTargetDataset(energy_data, centroid_data)
    if task_name == "energy_dispersion":
        return CombinedTargetDataset(energy_data, dispersion_data)

    base = energy_data or centroid_data or dispersion_data
    return ComponentDataset(base, comp_idx) if comp_idx is not None else base


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(net, predictor, test_loader, device):
    """Return predictions and targets as (N, n_tasks) tensors."""
    net.eval()
    all_preds, all_targets = [], []
    for data, targets in test_loader:
        data    = data.to(device)
        targets = targets.to(device)
        output  = net(data)
        pred, _ = predictor(output, targets, reduction="none")
        all_preds.append(pred.cpu())
        all_targets.append(targets.cpu())

    preds   = torch.cat(all_preds,   dim=0)
    targets = torch.cat(all_targets, dim=0)

    if preds.dim() == 1:
        preds   = preds.unsqueeze(-1)
    if targets.dim() == 1:
        targets = targets.unsqueeze(-1)

    return preds, targets   # (N, n_tasks)


# ── mm-converted spatial arrays ───────────────────────────────────────────────
def make_mm_arrays(preds, targets, task_name):
    """Return (mm_preds, mm_targets, mm_labels) for the spatial outputs of task_name.

    Returns (None, None, None) when the task has no spatial outputs.
    """
    _, _, _, _, _, _, cell_sizes, mm_labels = _TASK_META[task_name]
    spatial_idx = [i for i, cs in enumerate(cell_sizes) if cs is not None]
    if not spatial_idx:
        return None, None, None

    mm_p = preds[:, spatial_idx].clone()
    mm_t = targets[:, spatial_idx].clone()
    for k, i in enumerate(spatial_idx):
        mm_p[:, k] = mm_p[:, k] * cell_sizes[i]
        mm_t[:, k] = mm_t[:, k] * cell_sizes[i]

    return mm_p, mm_t, mm_labels


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, label: str) -> dict:
    """RMSE, MAE and mean relative error for one output column.

    Entries where |target| < 1e-1 are excluded from the relative error.
    """
    residuals = preds - targets
    rmse = float(residuals.pow(2).mean().sqrt())
    mae  = float(residuals.abs().mean())
    valid = targets.abs() >= 1e-1
    if valid.any():
        mre = float((residuals[valid].abs() / targets[valid].abs()).mean())
    else:
        mre = float("nan")
    return {"label": label, "RMSE": rmse, "MAE": mae, "mean_relative_error": mre}


# ── Plot helpers ──────────────────────────────────────────────────────────────
def _subplot_layout(n: int):
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    return nrows, ncols


# ── Plot 1: training / validation loss curves ─────────────────────────────────
def plot_loss_curves(loss_hist: dict, out_path: str, title: str) -> None:
    """Plot per-batch training loss and per-epoch validation loss vs iteration.

    loss_hist is the 'loss_hist' dict stored in the checkpoint:
        loss_hist["train"]      = {epoch_idx: [batch_loss, ...]}
        loss_hist["validation"] = {epoch_idx: scalar_loss}
    """
    train_hist = loss_hist.get("train", {})
    val_hist   = loss_hist.get("validation", {})

    if not train_hist and not val_hist:
        print("  (no loss history in checkpoint — skipping loss plot)")
        return

    # Flatten training batches to a single iteration axis
    train_iters  = []
    train_losses = []
    offset = 0
    for epoch in sorted(train_hist.keys()):
        batch_losses = train_hist[epoch]
        for b, loss in enumerate(batch_losses):
            train_iters.append(offset + b)
            train_losses.append(loss)
        offset += len(batch_losses)

    # Validation: place each epoch marker at the iteration where that epoch ended.
    # Epoch 0 in val_hist is the pre-training baseline → iteration 0.
    val_iters  = []
    val_losses = []
    cumulative = 0
    for epoch in sorted(val_hist.keys()):
        val_iters.append(cumulative)
        val_losses.append(val_hist[epoch])
        # Advance by the number of batches in this training epoch (if it exists)
        if epoch in train_hist:
            cumulative += len(train_hist[epoch])

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

    if train_losses:
        ax.plot(train_iters, train_losses,
                color="steelblue", linewidth=0.8, alpha=0.8, label="Training (per batch)")
    if val_losses:
        ax.plot(val_iters, val_losses,
                color="darkorange", linewidth=1.8, linestyle="--",
                marker="o", markersize=5, label="Validation (per epoch)")

    ax.set_xlabel("Iteration (batches)", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Plot helpers (shared) ──────────────────────────────────────────────────────
def _subplot_title(label, title_prefix, suptitle):
    """Return per-subplot title: just the label when a figure suptitle is used."""
    return label if suptitle else f"{title_prefix}\n{label}"


def _get_range(forced_ranges, i, t, p, margin_frac=0.05):
    """Return (lo, hi) for axis / histogram range."""
    if forced_ranges is not None and forced_ranges[i] is not None:
        return forced_ranges[i]
    lo = min(t.min(), p.min())
    hi = max(t.max(), p.max())
    margin = margin_frac * (hi - lo)
    return lo - margin, hi + margin


# ── Plot 2: 2D histogram pred vs target ───────────────────────────────────────
def plot_2d_histograms(preds, targets, labels, out_path, title_prefix,
                       forced_ranges=None, integer_ticks=None, suptitle=None):
    """2D histogram (SymLogNorm colour scale), one subplot per output.

    forced_ranges  : list of (lo, hi) or None per output; None → auto range.
    integer_ticks  : list of bool; True → set integer ticks on both axes.
    suptitle       : overall figure title (used for multi-target figures).
    """
    n = preds.shape[1]
    nrows, ncols = _subplot_layout(n)
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(6 * ncols, 5 * nrows),
                            constrained_layout=True)
    axs = np.atleast_1d(axs).flatten()

    for i, label in enumerate(labels):
        ax = axs[i]
        t  = targets[:, i].numpy()
        p  = preds[:, i].numpy()

        lo, hi = _get_range(forced_ranges, i, t, p)

        h = ax.hist2d(t, p, bins=60, range=[[lo, hi], [lo, hi]],
                      cmap="viridis", norm=SymLogNorm(linthresh=1))
        fig.colorbar(h[3], ax=ax, label="Counts")
        ax.plot([lo, hi], [lo, hi],
                color="white", linestyle="--", linewidth=1.5, label="Ideal")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Target",     fontsize=12)
        ax.set_ylabel("Prediction", fontsize=12)
        ax.set_title(_subplot_title(label, title_prefix, suptitle), fontsize=11)
        ax.legend(fontsize=10, loc="upper left")

        if integer_ticks is not None and integer_ticks[i]:
            ticks = list(range(int(np.floor(lo)), int(np.ceil(hi)) + 1))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

    for j in range(n, len(axs)):
        fig.delaxes(axs[j])

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Plot 3: residuals distribution ────────────────────────────────────────────
def plot_residuals(preds, targets, labels, out_path, title_prefix, suptitle=None):
    """Distribution of residuals (pred − target), one subplot per output."""
    n = preds.shape[1]
    nrows, ncols = _subplot_layout(n)
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(6 * ncols, 4 * nrows),
                            constrained_layout=True)
    axs = np.atleast_1d(axs).flatten()

    for i, label in enumerate(labels):
        ax  = axs[i]
        res = (preds[:, i] - targets[:, i]).numpy()

        ax.hist(res, bins=80, edgecolor="black", linewidth=0.4,
                alpha=0.8, color="steelblue")
        ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Residual (pred − target)", fontsize=12)
        ax.set_ylabel("Counts", fontsize=12)
        ax.set_title(_subplot_title(label, title_prefix, suptitle), fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        mu, sigma = res.mean(), res.std()
        ax.text(0.97, 0.95,
                f"μ = {mu:.4f}\nσ = {sigma:.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    for j in range(n, len(axs)):
        fig.delaxes(axs[j])

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Plot 4: relative error profile vs binned target ───────────────────────────
def plot_rel_error_profile(preds, targets, labels, out_path, title_prefix,
                           n_bins: int = 15, forced_ranges=None,
                           integer_ticks=None, suptitle=None, log_x=None):
    """Mean relative error ± std as a function of binned target value.

    Entries where |target| < 1e-1 are excluded.
    forced_ranges / integer_ticks apply to the x-axis (target range).
    log_x: list of bool per output; True → log-spaced bins and log x-axis scale.
    """
    n = preds.shape[1]
    nrows, ncols = _subplot_layout(n)
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(6 * ncols, 4 * nrows),
                            constrained_layout=True)
    axs = np.atleast_1d(axs).flatten()

    for i, label in enumerate(labels):
        ax = axs[i]
        t_all = targets[:, i].numpy()
        p_all = preds[:, i].numpy()

        # Skip entries where |target| is negligible
        valid_mask = np.abs(t_all) >= 1e-1
        t = t_all[valid_mask]
        p = p_all[valid_mask]

        if len(t) < 2:
            ax.set_title(_subplot_title(label, title_prefix, suptitle), fontsize=11)
            continue

        rel_err = np.abs(p - t) / np.abs(t)

        # Use forced range for bin edges / x-axis if provided
        if forced_ranges is not None and forced_ranges[i] is not None:
            lo, hi = forced_ranges[i]
        else:
            lo, hi = t.min(), t.max()

        use_log = log_x is not None and log_x[i]
        if use_log:
            bin_edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
        else:
            bin_edges = np.linspace(lo, hi, n_bins + 1)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_idx     = np.clip(np.digitize(t, bin_edges) - 1, 0, n_bins - 1)

        means = np.full(n_bins, np.nan)
        stds  = np.full(n_bins, np.nan)
        for k in range(n_bins):
            mask = bin_idx == k
            if mask.sum() >= 2:
                means[k] = rel_err[mask].mean()
                stds[k]  = rel_err[mask].std()

        ok = ~np.isnan(means)
        xv, yv, sv = bin_centres[ok], means[ok], stds[ok]

        ax.plot(xv, yv, marker="o", markersize=4, linewidth=1.5,
                color="steelblue", label="Mean rel. error")
        ax.fill_between(xv, yv - sv, yv + sv,
                        alpha=0.25, color="steelblue", label="±1 std")

        ax.set_xlabel("Target",         fontsize=12)
        ax.set_ylabel("Relative error", fontsize=12)
        ax.set_title(_subplot_title(label, title_prefix, suptitle), fontsize=11)
        ax.set_xlim(lo, hi)
        if use_log:
            ax.set_xscale("log")
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(bottom=0)

        if integer_ticks is not None and integer_ticks[i]:
            ax.set_xticks(range(int(np.floor(lo)), int(np.ceil(hi)) + 1))

    for j in range(n, len(axs)):
        fig.delaxes(axs[j])

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Plot 5: absolute error profile vs binned target ───────────────────────────
def plot_abs_error_profile(preds, targets, labels, out_path, title_prefix,
                           n_bins: int = 15, forced_ranges=None,
                           integer_ticks=None, suptitle=None, log_x=None):
    """Mean absolute error ± std as a function of binned target value.

    forced_ranges / integer_ticks apply to the x-axis (target range).
    log_x: list of bool per output; True → log-spaced bins and log x-axis scale.
    """
    n = preds.shape[1]
    nrows, ncols = _subplot_layout(n)
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(6 * ncols, 4 * nrows),
                            constrained_layout=True)
    axs = np.atleast_1d(axs).flatten()

    for i, label in enumerate(labels):
        ax = axs[i]
        t = targets[:, i].numpy()
        p = preds[:, i].numpy()

        if len(t) < 2:
            ax.set_title(_subplot_title(label, title_prefix, suptitle), fontsize=11)
            continue

        abs_err = np.abs(p - t)

        # Use forced range for bin edges / x-axis if provided
        if forced_ranges is not None and forced_ranges[i] is not None:
            lo, hi = forced_ranges[i]
        else:
            lo, hi = t.min(), t.max()

        use_log = log_x is not None and log_x[i]
        if use_log:
            bin_edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
        else:
            bin_edges = np.linspace(lo, hi, n_bins + 1)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_idx     = np.clip(np.digitize(t, bin_edges) - 1, 0, n_bins - 1)

        means = np.full(n_bins, np.nan)
        stds  = np.full(n_bins, np.nan)
        for k in range(n_bins):
            mask = bin_idx == k
            if mask.sum() >= 2:
                means[k] = abs_err[mask].mean()
                stds[k]  = abs_err[mask].std()

        ok = ~np.isnan(means)
        xv, yv, sv = bin_centres[ok], means[ok], stds[ok]

        ax.plot(xv, yv, marker="o", markersize=4, linewidth=1.5,
                color="darkorange", label="Mean abs. error")
        ax.fill_between(xv, yv - sv, yv + sv,
                        alpha=0.25, color="darkorange", label="±1 std")

        ax.set_xlabel("Target",          fontsize=12)
        ax.set_ylabel("Absolute error",  fontsize=12)
        ax.set_title(_subplot_title(label, title_prefix, suptitle), fontsize=11)
        ax.set_xlim(lo, hi)
        if use_log:
            ax.set_xscale("log")
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(bottom=0)

        if integer_ticks is not None and integer_ticks[i]:
            ax.set_xticks(range(int(np.floor(lo)), int(np.ceil(hi)) + 1))

    for j in range(n, len(axs)):
        fig.delaxes(axs[j])

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Convenience: run all three plots for one set of arrays ────────────────────
def _save_plot_trio(preds, targets, labels, out_dir, prefix, title_prefix,
                    forced_ranges=None, integer_ticks=None, suptitle=None,
                    log_x=None):
    """Save hist2d, residuals, rel-error profile and abs-error profile for a given set of arrays."""
    plot_2d_histograms(preds, targets, labels,
                       os.path.join(out_dir, f"hist2d{prefix}.png"),
                       title_prefix, forced_ranges, integer_ticks, suptitle)
    plot_residuals(preds, targets, labels,
                   os.path.join(out_dir, f"residuals{prefix}.png"),
                   title_prefix, suptitle)
    plot_rel_error_profile(preds, targets, labels,
                           os.path.join(out_dir, f"rel_error_profile{prefix}.png"),
                           title_prefix, forced_ranges=forced_ranges,
                           integer_ticks=integer_ticks, suptitle=suptitle,
                           log_x=log_x)
    plot_abs_error_profile(preds, targets, labels,
                           os.path.join(out_dir, f"abs_error_profile{prefix}.png"),
                           title_prefix, forced_ranges=forced_ranges,
                           integer_ticks=integer_ticks, suptitle=suptitle,
                           log_x=log_x)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate one trained checkpoint.")
    parser.add_argument("--ckpt_path", required=True,
                        help="Path to the completed .pt checkpoint file.")
    parser.add_argument("--device", default=None,
                        help="Torch device string, e.g. 'cpu' or 'cuda'. "
                             "Defaults to 'cuda' if available, else 'cpu'.")
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt_path):
        sys.exit(f"ERROR: checkpoint not found: {args.ckpt_path}")

    # ── Parse checkpoint filename ─────────────────────────────────────────────
    task_name, output_mode = parse_ckpt_name(args.ckpt_path)
    needs_E, needs_C, needs_D, comp_idx, n_tasks, output_labels, cell_sizes, _ = \
        _TASK_META[task_name]

    stem = os.path.splitext(os.path.basename(args.ckpt_path))[0]
    print(f"\n{'='*64}")
    print(f"  Checkpoint : {stem}")
    print(f"  Task       : {task_name}")
    print(f"  Mode       : {output_mode}")
    print(f"  Outputs    : {n_tasks}  ({', '.join(output_labels)})")
    print(f"{'='*64}\n")

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = os.path.join(PLOTS_DIR, stem)
    os.makedirs(out_dir, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Build network + predictor ─────────────────────────────────────────────
    if output_mode in ("spike", "membrane", "refl"):
        snn_mode    = "spike" if output_mode in ("spike", "refl") else "membrane"
        output_size = POPULATION * n_tasks
        net         = make_snn_net(output_size, snn_mode)
        pred_fn     = predict_spikefreq if snn_mode == "spike" else predict_membrane
        predictor   = snnfn.Predictor(pred_fn, absolute_error, population_sizes=POPULATION)
    else:   # ann
        net       = ANN(ds.nSensors, hidden_size=120, output_size=n_tasks)
        pred_fn   = lambda x: x.squeeze(-1)
        predictor = snnfn.Predictor(pred_fn, absolute_error, population_sizes=1)

    # ── Load weights ──────────────────────────────────────────────────────────
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.to(device)
    print(f"Loaded checkpoint  (epoch {ckpt.get('epoch', '?')})\n")

    # ── Dataset + test loader ─────────────────────────────────────────────────
    print("Indexing dataset …")
    dataset = build_task_dataset(task_name)
    print(f"  Total samples : {len(dataset)}")

    torch.manual_seed(EVAL_SEED)
    _, test_loader, _ = ds.build_loaders(
        dataset, split=(0.7, 0.15), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0,
    )
    print(f"  Test samples  : {len(test_loader.dataset)}\n")

    # ── Inference ─────────────────────────────────────────────────────────────
    print("Running inference …")
    preds, targets = run_inference(net, predictor, test_loader, device)
    # preds / targets shape: (N, n_tasks)

    # ── Metrics — native output space ────────────────────────────────────────
    all_metrics = []
    print("\n── Metrics (native units) ──")
    for i, label in enumerate(output_labels):
        m = compute_metrics(preds[:, i], targets[:, i], label)
        all_metrics.append(m)
        print(f"  [{label:35s}]  RMSE={m['RMSE']:.6f}  "
              f"MAE={m['MAE']:.6f}  MRE={m['mean_relative_error']:.6f}")

    # ── Metrics — mm units for spatial outputs ────────────────────────────────
    mm_preds, mm_targets, mm_labels = make_mm_arrays(preds, targets, task_name)
    if mm_preds is not None:
        print("\n── Metrics (mm units) ──")
        for i, label in enumerate(mm_labels):
            m = compute_metrics(mm_preds[:, i], mm_targets[:, i], label)
            all_metrics.append(m)
            print(f"  [{label:35s}]  RMSE={m['RMSE']:.4f}  "
                  f"MAE={m['MAE']:.4f}  MRE={m['mean_relative_error']:.6f}")

    # ── Metrics — linear energy scale (MeV → GeV) ────────────────────────────
    E_pred_GeV = E_target_GeV = None
    if needs_E:
        E_pred_GeV   = torch.pow(10.0, preds[:, 0])   / 1000.0
        E_target_GeV = torch.pow(10.0, targets[:, 0]) / 1000.0
        m_lin = compute_metrics(E_pred_GeV, E_target_GeV, r"$E$ [GeV]")
        all_metrics.append(m_lin)
        print(f"\n── Metrics (linear energy: E [GeV]) ──")
        print(f"  [{'E [GeV]':35s}]  RMSE={m_lin['RMSE']:.4f}  "
              f"MAE={m_lin['MAE']:.4f}  MRE={m_lin['mean_relative_error']:.6f}")

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump({
            "checkpoint":     stem,
            "task":           task_name,
            "mode":           output_mode,
            "n_test_samples": len(test_loader.dataset),
            "metrics":        all_metrics,
        }, fh, indent=2)
    print(f"\nMetrics saved → {metrics_path}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PLOTS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\nSaving plots …")

    title_prefix  = _MODE_TITLE[output_mode]
    is_multi      = n_tasks > 1
    multi_suptitle = (_MULTI_SUPTITLE.get(output_mode, "Multi-target regression")
                      if is_multi else None)

    # Per-output axis ranges and integer-tick flags for native-unit plots
    forced_native    = [(1.5, 5.0) if l == _LOGE_LABEL
                        else (0.0, 9.0) if l in _CENTROID_LABS
                        else None
                        for l in output_labels]
    int_ticks_native = [l in _CENTROID_LABS for l in output_labels]

    # ── 1. Training / validation loss curves ──────────────────────────────────
    loss_hist = ckpt.get("loss_hist", {})
    plot_loss_curves(
        loss_hist,
        os.path.join(out_dir, "loss_curves.png"),
        title=f"{title_prefix} — {stem}",
    )

    # ── 2-4. Native units: 2D hist + residuals + rel-error profile ────────────
    _save_plot_trio(preds, targets, output_labels, out_dir, "",
                    title_prefix, forced_native, int_ticks_native, multi_suptitle)

    # ── 5-10. Physical units (mm spatial + GeV energy) ────────────────────────
    if is_multi:
        # For multi-target tasks: combine linear-E [GeV] and mm spatial in one set
        if E_pred_GeV is not None and mm_preds is not None:
            phys_p      = torch.cat([E_pred_GeV.unsqueeze(-1),   mm_preds], dim=1)
            phys_t      = torch.cat([E_target_GeV.unsqueeze(-1), mm_targets], dim=1)
            phys_labels = [r"$E$ [GeV]"] + mm_labels
            phys_ranges = [(0.03, 100)] + [None] * len(mm_labels)
            phys_log_x  = [True]        + [False] * len(mm_labels)
            _save_plot_trio(phys_p, phys_t, phys_labels, out_dir, "_physical",
                            title_prefix, phys_ranges, [False] * len(phys_labels),
                            multi_suptitle, log_x=phys_log_x)
    else:
        # Single-target: separate mm and linear-energy plots
        if mm_preds is not None:
            _save_plot_trio(mm_preds, mm_targets, mm_labels, out_dir, "_mm",
                            title_prefix)
        if E_pred_GeV is not None:
            _save_plot_trio(E_pred_GeV.unsqueeze(-1), E_target_GeV.unsqueeze(-1),
                            [r"$E$ [GeV]"], out_dir, "_energy_linear",
                            title_prefix, forced_ranges=[(0.03, 100)],
                            log_x=[True])

    print(f"\nAll outputs → {out_dir}/")
    print(f"Done: {stem}\n")


if __name__ == "__main__":
    main()
