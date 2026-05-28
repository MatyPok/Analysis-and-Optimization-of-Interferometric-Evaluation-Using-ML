
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models import ResNetRegressor
from plots import plot_training_sample
import json
import time

from sklearn.metrics import r2_score

# Path to model

par_dir = "meta_output/zernike/Zs_A_mag_40_60_40dB_var_224x224"
folder = "resnet_10k_224_zernike_pretrained_normy_weighted_combined_wd_2_lr_3_hd_0.1_BEST"

model_path = os.path.join(par_dir, folder, "output", "model.pth")

test_dir = "tests"

image = "exp_inten_1.png"
intens_path = os.path.join(test_dir, image)
zernikes_path = os.path.join(test_dir, "zernikes_1.txt")


###############################################################################
# LOAD IMAGE
###############################################################################


# Load intensity image
intensity_image = cv2.imread(intens_path, cv2.IMREAD_GRAYSCALE)
if intensity_image is None:
    raise FileNotFoundError(f"Could not load image: {intens_path}")
# Normalize to [0, 1]
intensity_image = intensity_image.astype(np.float32) / 255.0

# Target resolution of the image
target_size = (224, 224)

# Resize the image to the target size
intensity_image = cv2.resize(
    intensity_image, 
    target_size, 
    interpolation=cv2.INTER_NEAREST)

# To numpy array and add channel dimension
X_test = np.expand_dims(intensity_image, axis=-1)  # shape (224, 224, 1)
X_test = np.expand_dims(X_test, axis=0)  # shape (1, 224, 224, 1)

# Load Zernike coefficients
zernikes = np.loadtxt(
    zernikes_path,
    usecols=1,
    dtype=np.float32
)

gt_tilt = zernikes[1:3]  # Assuming the file has two columns and we want the second one (coefficients) and we want to skip the first 3 coefficients (piston, tip, tilt)
zernikes = zernikes[3:]   # skip Z0, Z1, Z2


def zemax_to_iso(zemax_coeffs):
    """
    Convert Zemax-style coefficient ordering
    to your ISO ordering.

    Parameters
    ----------
    zemax_coeffs : ndarray shape (N,)
        Input Zemax coefficients

    Returns
    -------
    iso_coeffs : ndarray shape (N,)
        Reordered coefficients in ISO convention
    """
    # This assumes input is from Z0 to ZN
    # We have input from Z3 to ZN
    iso_coeffs = np.copy(zemax_coeffs)
    iso_coeffs = np.insert(iso_coeffs, 0, 0)  # Insert Z0=0 at the beginning
    iso_coeffs = np.insert(iso_coeffs, 0, 0)  # Insert Z1=0 and Z2=0 for tip and tilt
    iso_coeffs = np.insert(iso_coeffs, 0, 0)

    # --------------------------------------------------------
    # Primary astigmatism swap
    # Zemax:
    #   Z4 = astig 45°
    #   Z5 = astig 0°
    #
    # ISO:
    #   Z4 = astig 0°
    #   Z5 = astig 45°
    # --------------------------------------------------------

    iso_coeffs[4] = zemax_coeffs[5]
    iso_coeffs[5] = zemax_coeffs[4]

    # --------------------------------------------------------
    # If coma is swapped in your data:
    #
    # uncomment this block
    # --------------------------------------------------------

    iso_coeffs[6] = zemax_coeffs[7]
    iso_coeffs[7] = zemax_coeffs[6]

    iso_coeffs[8] = zemax_coeffs[10]

    iso_coeffs[9] = zemax_coeffs[8]
    iso_coeffs[10] = zemax_coeffs[9]

    # --------------------------------------------------------
    # Example for trefoil swap
    # uncomment if needed
    # --------------------------------------------------------

    # iso_coeffs[9]  = zemax_coeffs[10]
    # iso_coeffs[10] = zemax_coeffs[9]

    return iso_coeffs


y_test = zemax_to_iso(zernikes)

y_test = np.expand_dims(y_test[3:], axis=0)

###############################################################################
# EXTRACT TILT
###############################################################################

# Extract tilt coefficients using the non-learned module
from tilt_extractor import TiltExtractor
tilt_extractor = TiltExtractor(image_size=target_size[0])
tilt_y, tilt_x = tilt_extractor.extract(intensity_image)
tilt_test = np.array([[tilt_y, tilt_x]], dtype=np.float32)

print(f"Tilt Y: {tilt_y:.6f}, GT Tilt Y: {gt_tilt[0]:.6f}")
print(f"Tilt X: {tilt_x:.6f}, GT Tilt X: {gt_tilt[1]:.6f}")


#########################################################################
normalize_y = "normy" in folder
print(f"Normy:{normalize_y}" )

if normalize_y:
    y_norm_stats_path = par_dir + "/" + folder + "/" + "output/y_norm_stats.npz"
    y_norm = np.load(y_norm_stats_path)
    y_mean = y_norm["mean"]
    y_std = y_norm["std"]

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
print(f"Prediction time per one interferogram:{(1000*(end-start)/(koefs_pred.shape[0])):.2f} ms")



# Simulate interferogram from predicted coefficients
from interferogram import Interferogram

resolution = X_test.shape[1]

phases_pred = Interferogram(koefs_pred_denorm, show=False, size=resolution, tilt=False).phase()
phases_pred = np.reshape(phases_pred, (np.shape(phases_pred)[0], resolution, resolution, 1))

phases_test = Interferogram(y_test, show=False, size=resolution, tilt=False).phase()
phases_test = np.reshape(phases_test, (np.shape(phases_test)[0], resolution, resolution, 1))

###############################################################################
# Generate detailed visualizations using plots.py
###############################################################################

# Create visualizations directory
viz_dir = os.path.join(test_dir, "visualizations")
os.makedirs(viz_dir, exist_ok=True)

# Generate visualizations for first few test samples
num_samples_to_visualize = min(1, len(X_test))


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

    save_path = os.path.join(viz_dir, f"test_sample_{image}")
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

