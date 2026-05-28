"""Visualization utilities for training and evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_sample(
    interferogram: np.ndarray,
    true_coeffs: np.ndarray,
    pred_coeffs: np.ndarray,
    true_phase: np.ndarray,
    pred_phase: np.ndarray,
    mse: float,
    rmse: float,
    max_error: float,
    per_coeff_rmse: np.ndarray,
    save_path: str | Path,
    wavefront_rms: float | None = None,
) -> None:
    """Generate a 2x2 visualization of training progress.

    Panels:
        Top-left: Interferogram image
        Top-right: Bar plot of true vs predicted coefficients
        Bottom-left: True phase map (no tilt)
        Bottom-right: Reconstructed phase map (no tilt)

    Args:
        interferogram: 2D array, the input image.
        true_coeffs: True predicted coefficients (num_zernike - 3 values).
        pred_coeffs: Predicted coefficients (num_zernike - 3 values).
        true_phase: True phase map (generated without tilts).
        pred_phase: Predicted phase map.
        mse: Mean squared error.
        rmse: Root mean squared error.
        max_error: Maximum absolute error.
        per_coeff_rmse: Per-coefficient RMSE.
        save_path: Where to save the figure.
        wavefront_rms: RMS of wavefront error (no tilt), in radians.
    """
    num_coeffs = len(true_coeffs)
    zernike_labels = [f"Z{i}" for i in range(4, 4 + num_coeffs)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    title = f"MSE: {mse:.6f}  |  RMSE: {rmse:.4f}  |  Max Error: {max_error:.4f}"
    if wavefront_rms is not None:
        title += f"  |  WF RMS: {wavefront_rms:.4f} rad"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Top-left: Interferogram
    ax = axes[0, 0]
    im = ax.imshow(interferogram, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Input Interferogram")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Top-right: Coefficient comparison bar plot
    ax = axes[0, 1]
    x = np.arange(num_coeffs)
    width = 0.35
    ax.bar(x - width / 2, true_coeffs, width, label="True", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, pred_coeffs, width, label="Predicted", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(zernike_labels, fontsize=9, rotation=45 if num_coeffs > 16 else 0)
    ax.set_ylabel("Coefficient Value")
    ax.set_title(f"Zernike Coefficients (Z4–Z{3 + num_coeffs})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Bottom-left: True phase
    ax = axes[1, 0]
    im = ax.imshow(true_phase, cmap="RdBu_r")
    ax.set_title("True Phase (no tilt)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Bottom-right: Predicted phase
    ax = axes[1, 1]
    im = ax.imshow(pred_phase, cmap="RdBu_r")
    ax.set_title("Predicted Phase (no tilt)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_inference_result(
    interferogram: np.ndarray,
    pred_coeffs: np.ndarray,
    pred_phase: np.ndarray,
    save_path: str | Path,
    num_zernike: int = 15,
) -> None:
    """Visualize inference on a real image (no ground truth).

    Panels:
        Left: Input interferogram
        Center: Predicted Zernike coefficients bar chart
        Right: Reconstructed phase map (no tilt)

    Args:
        interferogram: 2D array, the input image.
        pred_coeffs: Predicted coefficients Z4–Z{num_zernike}.
        pred_phase: Predicted phase map.
        save_path: Where to save the figure.
        num_zernike: Total number of Zernike modes (for labeling).
    """
    num_coeffs = len(pred_coeffs)
    zernike_labels = [f"Z{i}" for i in range(4, 4 + num_coeffs)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Inference Result", fontsize=14, fontweight="bold")

    # Left: Interferogram
    ax = axes[0]
    im = ax.imshow(interferogram, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Input Interferogram")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Center: Predicted coefficients
    ax = axes[1]
    x = np.arange(num_coeffs)
    ax.bar(x, pred_coeffs, color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(zernike_labels, fontsize=9, rotation=45 if num_coeffs > 16 else 0)
    ax.set_ylabel("Coefficient Value")
    ax.set_title(f"Predicted Coefficients (Z4–Z{3 + num_coeffs})")
    ax.grid(axis="y", alpha=0.3)

    # Right: Predicted phase
    ax = axes[2]
    im = ax.imshow(pred_phase, cmap="RdBu_r")
    ax.set_title("Predicted Phase (no tilt)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
