import time
import os
import json

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models import ResNetRegressor
from plots import plot_training_sample

from sklearn.metrics import r2_score



# Paths to data, history and model

par_dir = "meta_output/zernike/Zs_A_mag_40_60_NonedB_464x464"
folder = "resnet_10k_464_zernike_pretrained_normy_weighted_combined_wd_1_lr_3_hd_0.1_BEST"

model_path = par_dir + "/" + folder + "/" + "output/model.pth"
history_path = par_dir + "/" + folder + "/" + "output/history.npy"
test_data_path = par_dir + "/" + folder + "/" + "output/test_data.npz"

##########################################################################
# First load history

history = np.load(history_path, allow_pickle=True)[()]


loss = np.array(history["loss"])
loss = loss[loss < 1.0]

val_loss = np.array(history["val_loss"])
val_loss = val_loss[val_loss < 1.0]

print(history)

fig, ax = plt.subplots()
plt.plot(loss[:], label='loss')
plt.plot(val_loss[:], label='val_loss')
ax.set_xlabel('iterations')
ax.set_ylabel('loss')
ax.legend()
plt.show()

#########################################################################
# Load data

data = np.load(test_data_path)

X_test = data['X_test'][:250]
y_test = data['y_test'][:250]
tilt_test = data["tilt_test"][:250]

normalize_y = "normy" in folder
print(f"Normy:{normalize_y}" )

if normalize_y:
    y_norm_stats_path = par_dir + "/" + folder + "/" + "output/y_norm_stats.npz"
    y_norm = np.load(y_norm_stats_path)
    y_mean = y_norm["mean"]
    y_std = y_norm["std"]


# Just for case print already all the shapes of loaded data
print(X_test.shape)
print(y_test.shape)
print(tilt_test.shape)
#print(phases_test.shape)

#########################################################################
# Load model and config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config to determine input channels
config_path = os.path.join(par_dir, folder, "output", "train_config.json")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    model_type = config.get("model", "resnet")
    # Check if the model was trained with mask (2 channels) or without (1 channel)
    # For resnet, we typically use 2 channels (image + mask), but let's check the actual checkpoint
    in_channels = 2 if model_type == "resnet" else 1
else:
    in_channels = 1  # default fallback

# Try to infer input channels from checkpoint
try:
    temp_checkpoint = torch.load(model_path, map_location=device)
    if isinstance(temp_checkpoint, dict):
        if "model_state_dict" in temp_checkpoint:
            state_dict = temp_checkpoint["model_state_dict"]
        elif "state_dict" in temp_checkpoint:
            state_dict = temp_checkpoint["state_dict"]
        else:
            state_dict = temp_checkpoint
    else:
        state_dict = temp_checkpoint

    # Check conv1 weight shape to determine input channels
    if "conv1.weight" in state_dict:
        conv1_shape = state_dict["conv1.weight"].shape
        in_channels = conv1_shape[1]  # second dimension is input channels
        print(f"Detected {in_channels} input channels from checkpoint")
except Exception as e:
    print(f"Could not detect input channels from checkpoint: {e}, using default {in_channels}")

model = ResNetRegressor(num_outputs=y_test.shape[1], pretrained=False, in_channels=in_channels)

checkpoint = torch.load(model_path, map_location=device)

# Handle different checkpoint formats
if isinstance(checkpoint, dict):
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Assume the checkpoint is directly the state dict
        model.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

# Prepare input data
if X_test.shape[-1] == 1:  # single channel, need to add mask for resnet
    X_test_torch = torch.from_numpy(X_test).permute(0, 3, 1, 2).float()
else:
    X_test_torch = torch.from_numpy(X_test).permute(0, 3, 1, 2).float()

tilt_torch = torch.from_numpy(tilt_test).float()

# Start inference
start = time.time()

with torch.no_grad():
    X_test_device = X_test_torch.to(device)
    tilt_device = tilt_torch.to(device)
    koefs_pred = model(X_test_device, tilt_device).cpu().numpy()

if normalize_y:
    koefs_pred_denorm = koefs_pred * y_std + y_mean

else:
    koefs_pred_denorm = koefs_pred


end = time.time()

print(f"Prediction time: {(end-start):.3f} s")
print(f"Prediction time per one interferogram:{(100*(end-start)/(koefs_pred.shape[0])):.2f} ms")


###############################################################################
# Visualize predicted coefficients
###############################################################################

def coefSliderPlot(coef_data, name="Coefficient viewer", titles=None):
    """
    coef_data : ndarray (N_samples, N_coeffs, 2)
                [:, i, 0] = truth
                [:, i, 1] = prediction
    titles    : list of str (optional)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    N_samples, N_coeffs, _ = coef_data.shape
    x = np.arange(N_samples)

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.subplots_adjust(bottom=0.25)

    # --- počáteční index ---
    idx = 0

    truth = coef_data[:, idx, 0]
    pred  = coef_data[:, idx, 1]
    mid   = (truth + pred) / 2
    err   = np.abs(truth - pred) / 2

    # --- scatter ploty ---
    scat_truth = ax.scatter(x, truth, label="truth")
    scat_pred  = ax.scatter(x, pred, label="prediction")

    err_container = ax.errorbar(x, mid, yerr=err, fmt="s", alpha=0.5)

    # kompatibilní přístup
    err_line = err_container[0]        # Line2D (marker)
    err_caps = err_container[1]        # caplines (list)
    err_bars = err_container[2][0]     # LineCollection (vertical bars)

    ax.set_xlabel("element from test set")
    ax.set_ylabel("value")
    ax.legend()

    def update_title(i):
        if titles is not None:
            ax.set_title(f"{name} – {titles[i]}")
        else:
            ax.set_title(f"{name} – coef {i}")

    update_title(idx)

    # --- slider ---
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.04])
    slider = Slider(
        ax_slider,
        "Coef index",
        0,
        N_coeffs - 1,
        valinit=0,
        valstep=1
    )

    def update(val):
        idx = int(slider.val)

        truth = coef_data[:, idx, 0]
        pred  = coef_data[:, idx, 1]
        mid   = (truth + pred) / 2
        err   = np.abs(truth - pred) / 2

        # update scatter
        scat_truth.set_offsets(np.c_[x, truth])
        scat_pred.set_offsets(np.c_[x, pred])

        # update errorbar marker line
        err_line.set_ydata(mid)

        # update vertical bars (LineCollection)
        segments = [
            ((xi, m - e), (xi, m + e))
            for xi, m, e in zip(x, mid, err)
        ]
        err_bars.set_segments(segments)

        update_title(idx)

        y_all = np.concatenate([
            truth,
            pred,
            mid - err,
            mid + err
        ])

        ymin = y_all.min()
        ymax = y_all.max()

        margin = 0.05 * (ymax - ymin if ymax != ymin else 1.0)

        ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_xlim(-1, N_samples)

        #ax.relim()
        #ax.autoscale_view()
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()


coef_data = np.stack([y_test[:50], koefs_pred[:50]], axis=2)

titles = [
    "4 - Defocus",
    "5 - Primary astigmatism - oblique",
    "6 - Primary astigmatism - vert.",
    "7 - Primary coma - Y"
]

titles = titles + [str(i) for i in range(8,37)]

coefSliderPlot(coef_data, name="Zernike coefficients", titles=titles)


###############################################################################


# Simulate interferogram from predicted coefficients
from interferogram import Interferogram

resolution = X_test.shape[1]

phases_pred = Interferogram(koefs_pred_denorm[:15], show=False, size=resolution, tilt=False).phase()
phases_pred = np.reshape(phases_pred, (np.shape(phases_pred)[0], resolution, resolution, 1))

phases_test = Interferogram(y_test[:15], show=False, size=resolution, tilt=False).phase()
phases_test = np.reshape(phases_test, (np.shape(phases_test)[0], resolution, resolution, 1))



"""
n = 6  # počet trojic které chceme zobrazit

difference_phase = np.subtract(phases_pred, phases_test)
# Visual control of predicted interferograms
fig, axes = plt.subplots(3, n, figsize=(10, 8))
axes = axes.ravel()

im_last = None  # uložíme poslední imshow

for i, (pred, test, diff) in enumerate(zip(phases_pred[:n], phases_test[:n], difference_phase[:n])):
    # horní řada: predikce
    axes[i].imshow(pred[:, :, 0], cmap='hsv')
    axes[i].set_title(f'Predicted phase {i+1}')
    axes[i].axis('off')

    # prostřední řada: odpovídající test
    axes[i + n].imshow(test[:, :, 0], cmap='hsv')
    axes[i + n].set_title(f'Test phase {i+1}')
    axes[i + n].axis('off')

    # dolní řada: rozdíl
    im_last = axes[i + 2*n].imshow(diff[:, :, 0], cmap='hsv')
    axes[i + 2*n].set_title(f'Test phase {i+1}')
    axes[i + 2*n].axis('off')

# colorbar jen pro poslední řádek
cbar = fig.colorbar(im_last, ax=axes[2*n:], orientation='horizontal', fraction=0.05, pad=0.05)


#plt.tight_layout()
plt.show()
"""


###############################################################################
# Generate detailed visualizations using plots.py
###############################################################################

# Create visualizations directory
viz_dir = os.path.join(par_dir, folder, "output", "visualizations")
os.makedirs(viz_dir, exist_ok=True)

# Generate visualizations for first few test samples
num_samples_to_visualize = min(10, len(X_test))

for i in range(num_samples_to_visualize):
    interferogram = X_test[i, :, :, 0]  # Remove channel dimension
    true_coeffs = y_test[i]
    pred_coeffs = koefs_pred_denorm[i]
    true_phase = phases_test[i, :, :, 0]
    pred_phase = phases_pred[i, :, :, 0]

    # Calculate metrics
    mse = np.mean((pred_coeffs - true_coeffs) ** 2)
    rmse = np.sqrt(mse)
    max_error = np.max(np.abs(pred_coeffs - true_coeffs))

    # Per-coefficient RMSE
    per_coeff_rmse = np.sqrt(np.mean((pred_coeffs - true_coeffs) ** 2, axis=0))

    # Wavefront RMS (difference between predicted and true phase)
    phase_diff = pred_phase - true_phase
    # Only consider pixels within unit circle (where Zernike basis is valid)
    y_coords, x_coords = np.ogrid[:resolution, :resolution]
    center = resolution // 2
    radius = resolution // 2
    mask = (x_coords - center) ** 2 + (y_coords - center) ** 2 <= radius ** 2
    wavefront_rms = np.sqrt(np.mean(phase_diff[mask] ** 2))

    save_path = os.path.join(viz_dir, f"test_sample_{i:03d}.png")
    plot_training_sample(
        interferogram=interferogram,
        true_coeffs=true_coeffs,
        pred_coeffs=pred_coeffs,
        true_phase=true_phase,
        pred_phase=pred_phase,
        mse=mse,
        rmse=rmse,
        max_error=max_error,
        per_coeff_rmse=per_coeff_rmse,
        wavefront_rms=wavefront_rms,
        save_path=save_path,
    )

print(f"Generated {num_samples_to_visualize} detailed visualizations in {viz_dir}")


#################################################
# Quantitative evaluation of the whole test set
#################################################

def MAE(X, y):
    return np.mean(np.abs(y - X))

def STD(X, y):
    return np.std(y - X)

maes = np.array([])
stds = np.array([])
r2s = np.array([])

for i in range(np.shape(y_test)[1]):
    mae = MAE(koefs_pred_denorm[:, i], y_test[:, i])
    std = STD(koefs_pred_denorm[:, i], y_test[:, i])

    r2 = r2_score(y_test[:, i], koefs_pred_denorm[:, i])

    maes = np.append(maes, mae)
    stds = np.append(stds, std)
    r2s = np.append(r2s, r2)

x_ax = [i for i in range(4,37)]

fig, ax = plt.subplots()
plt.plot(x_ax,maes, label='MAE')
plt.plot(x_ax,maes+stds, color='c', label='+-STD')
plt.plot(x_ax,maes-stds, color='c')
plt.fill_between(x_ax,maes-stds,maes+stds,alpha=0.3)
#plt.plot(stds, label='STD')
ax.set_xlabel('coefficient')
ax.set_ylabel('error')
ax.legend()
ax.set_title('MAE with standard deviation for each coefficient')
plt.xticks(x_ax)
plt.savefig(os.path.join(par_dir, folder, "output", "maes_with_stds_2.pdf"))
plt.show()


fig, ax = plt.subplots(figsize=(10, 5))
plt.plot(x_ax, r2s, marker='o', label='R²')
ax.axhline(1.0, color='g', linestyle='--', alpha=0.5)
ax.axhline(0.0, color='r', linestyle='--', alpha=0.5)
ax.set_xlabel('coefficient')
ax.set_ylabel('R² score')
ax.set_title('R² score for each Zernike coefficient')
plt.xticks(x_ax)
plt.ylim(min(-0.1, np.min(r2s) - 0.05), 1.05)
ax.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(par_dir, folder, "output", "r2_scores.pdf"))
plt.show()

print("\nR² per coefficient:")
for i, r2 in enumerate(r2s, start=4):
    print(f"Z{i:02d}: R² = {r2:.4f}")

print(f"\n Mean R2 score across all coefficients: {np.mean(r2s):.4f}")

