import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models import ResNetRegressor
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from polynomials import Zernikes

model = ResNetRegressor(num_outputs=33, pretrained=True)

# Assuming Zernikes is available or implement similar
# from polynomials import Zernikes  # For now, we'll assume it's available

# Loading configuration
def load_config(path="train_config.json"):
    with open(path, 'r') as file:
        config = json.load(file)
    return config

# Load config
wdir = os.getcwd()
config = load_config(os.path.join(wdir, "train_config.json"))

compute_with_noise = config["compute_with_noise"]
num_of_sets = config["num_of_sets"]
model_name = config["model"]
loss_function = config["loss_function"]
pretrained = config["pretrained"]
fine_tuning = config["fine_tuning"]
dropout_rate = config["dropout_rate"]
correlation_loss = config["correlation_loss"]
ncpus = config["resources"]["ncpus"]
num_workers = int(ncpus)
normalize_y = config["normalize_y"]
normalize_X = config["normalize_X"]
normalize_tilt = config["normalize_tilt"]
weight_loss = config["weight_loss"]
epochs = config["epochs"]

size = config["resolution"]

# Create unit disk mask and Zernike basis
x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
X_mesh, Y_mesh = np.meshgrid(x, y)
mask_np = (np.sqrt(X_mesh**2 + Y_mesh**2) <= 1).astype(np.float32)
mask = torch.from_numpy(mask_np).float()

R = np.sqrt(X_mesh**2 + Y_mesh**2)
theta = np.arctan2(Y_mesh, X_mesh)

# Create Zernike polynomials (Z4-Z36, skipping Z1-Z3 which are tilts)

zernike_polynomials = Zernikes(R, theta).zernike_array()
z_basis = torch.from_numpy(np.array(zernike_polynomials[3:], dtype=np.float32))  # Skip Z1-Z3

def wavefront_from_coeffs(coeffs: torch.Tensor) -> torch.Tensor:
    """Reconstruct wavefront from Zernike coefficients.
    
    Args:
        coeffs: (batch, num_coeffs) tensor of Zernike coefficients
        
    Returns:
        wavefront: (batch, H, W) tensor of reconstructed phase
    """
    coeffs_expanded = coeffs.unsqueeze(-1).unsqueeze(-1)  # (batch, num_coeffs, 1, 1)
    wavefront = torch.sum(coeffs_expanded * z_basis.unsqueeze(0), dim=1)  # (batch, H, W)
    return wavefront

# Loss scheduler for dynamic weighting
class LossScheduler(nn.Module):
    def __init__(self, total_epochs: int, y_mean: torch.Tensor, y_std: torch.Tensor, 
                 coef_start_epoch: int = None, 
                 normalize_y: bool = True, huber_delta: float = 1.0):
        super().__init__()
        self.total_epochs = total_epochs
        self.register_buffer('y_mean', y_mean)
        self.register_buffer('y_std', y_std)
        self.normalize_y = normalize_y
        self.current_epoch = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        # Kdy začít přidávat coefficient loss (defaultně od 60% tréninku)
        self.coef_start_epoch = coef_start_epoch if coef_start_epoch is not None else int(0.6 * total_epochs)
        self.huber_delta = huber_delta

    def forward(self, true_coeffs: torch.Tensor, pred_coeffs: torch.Tensor) -> torch.Tensor:
        epoch = self.current_epoch.item()
        
        # Phase loss počítáme vždy jako primární položku
        # Denormalize coefficients pro výpočet wavefront - jen pokud byly normalizované
        if self.normalize_y:
            true_denorm = true_coeffs * self.y_std + self.y_mean
            pred_denorm = pred_coeffs * self.y_std + self.y_mean
        else:
            # Pokud nebyly normalizované, použijeme přímo coefficienty
            true_denorm = true_coeffs
            pred_denorm = pred_coeffs
        
        # Phase loss z rekonstruovaných wavefrontů
        true_phase = wavefront_from_coeffs(true_denorm)
        pred_phase = wavefront_from_coeffs(pred_denorm)
        
        # Apply unit disk mask
        true_phase_masked = true_phase * mask
        pred_phase_masked = pred_phase * mask
        
        # Normalizujeme phase loss na skálu [0, 1] aby byl srovnatelný s coeff_loss
        valid_pixels = mask.sum().clamp(min=1.0)

        phase_rms = torch.sqrt(
            torch.sum(true_phase_masked ** 2) / valid_pixels
        )

        phase_diff = (true_phase_masked - pred_phase_masked) ** 2

        phase_loss = torch.sum(phase_diff) / valid_pixels
        phase_loss = phase_loss / (phase_rms ** 2 + 1e-8)
        
        # Coefficient loss přidáváme až později
        if epoch < self.coef_start_epoch:
            total_loss = phase_loss

            return {
                "total_loss": total_loss,
                "phase_loss": phase_loss.detach(),
                "coeff_loss": torch.tensor(0.0, device=phase_loss.device)
            }

        coeff_loss = nn.functional.smooth_l1_loss(
            true_coeffs,
            pred_coeffs,
            reduction='mean',
            beta=self.huber_delta
        )

        # Smooth transition
        progress = (epoch - self.coef_start_epoch) / (
            self.total_epochs - self.coef_start_epoch
        )

        progress = max(0.0, min(progress, 1.0))

        coeff_alpha = 0.5 * (1 - np.cos(np.pi * progress))
        phase_alpha = 1.0 - coeff_alpha

        total_loss = (
            phase_alpha * phase_loss +
            coeff_alpha * coeff_loss
        )

        return {
            "total_loss": total_loss,
            "phase_loss": phase_loss.detach(),
            "coeff_loss": coeff_loss.detach()
        }

    def update_epoch(self, epoch: int):
        self.current_epoch.data = torch.tensor(float(epoch))


class PerCoefficientWeightedLoss(nn.Module):
    """Loss that weights each coefficient by its variance in training data.
    
    This helps balance learning across coefficients with different scales.
    Higher variance coefficients get lower weights, lower variance get higher weights.
    """
    def __init__(self, y_train: np.ndarray, loss_type: str = "mse", huber_delta: float = 0.1):
        super().__init__()
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        
        # Calculate per-coefficient variance
        y_var = np.var(y_train, axis=0)
        # Invert and normalize: higher variance → lower weight, lower variance → higher weight
        # Avoid division by zero
        weights = 1.0 / (y_var + 1e-8)
        weights = weights / np.sum(weights) * len(weights)  # Normalize so mean weight = 1
        
        self.weights = torch.from_numpy(weights).float()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            diff = (pred - target) ** 2
        elif self.loss_type == "mae":
            diff = torch.abs(pred - target)
        else:  # huber
            diff = torch.nn.functional.smooth_l1_loss(pred, target, reduction='none', beta=self.huber_delta)
        
        # Apply per-coefficient weights
        weighted_diff = diff * self.weights.unsqueeze(0).to(pred.device)
        return torch.mean(weighted_diff)
   

X_file = 'intensity_circular.npz'
y_file = 'target_coefficients.npz'

X_dict = np.load(X_file)
y_dict = np.load(y_file)

X_data = X_dict['arr_0']
X_data = X_data[:num_of_sets]
if X_data.shape[1] != X_data.shape[2]:
    raise ValueError(f"Expected square images, but got shape {X_data.shape}")

resolution = X_data.shape[1]
X_data = np.reshape(X_data, (num_of_sets, resolution, resolution, 1))

y_data = y_dict['arr_0']
y_data = y_data[:num_of_sets]

tilt = np.load("tilt.npz")['arr_0']
tilt = tilt[:num_of_sets]
#phases = np.load("phases_circular.npz")['arr_0']
#phases = phases[:num_of_sets]

# Just for case print already all the shapes of loaded data
print(X_data.shape)
print(y_data.shape)
print(tilt.shape)
#print(phases.shape)

# Split data
X_train, X_rem, y_train, y_rem, tilt_train, tilt_rem = train_test_split(
    X_data, y_data, tilt, train_size=0.9, random_state=42
)
X_val, X_test, y_val, y_test, tilt_val, tilt_test = train_test_split(
    X_rem, y_rem, tilt_rem, test_size=0.5, random_state=42
)

# Save test data for later inference
test_data_path = os.path.join("output", "test_data.npz")
try:
    os.chmod(test_data_path, 0o664)
except Exception:
    pass
np.savez(test_data_path, X_test=X_test, y_test=y_test, tilt_test=tilt_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, 1, H, W)
X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

tilt_train = torch.tensor(tilt_train, dtype=torch.float32)
tilt_val = torch.tensor(tilt_val, dtype=torch.float32)
tilt_test = torch.tensor(tilt_test, dtype=torch.float32)

# Create DataLoaders
train_dataset = torch.utils.data.TensorDataset(X_train, tilt_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, tilt_val, y_val)
test_dataset = torch.utils.data.TensorDataset(X_test, tilt_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=num_workers)

# Model
model = ResNetRegressor(num_outputs=y_data.shape[1], pretrained=pretrained)

# Normalization (similar to TensorFlow version)
if normalize_y:
    y_mean = np.mean(y_train.numpy(), axis=0)
    y_std = np.std(y_train.numpy(), axis=0) + 1e-8
    y_train = (y_train - torch.tensor(y_mean, dtype=torch.float32)) / torch.tensor(y_std, dtype=torch.float32)
    y_val = (y_val - torch.tensor(y_mean, dtype=torch.float32)) / torch.tensor(y_std, dtype=torch.float32)
    y_test = (y_test - torch.tensor(y_mean, dtype=torch.float32)) / torch.tensor(y_std, dtype=torch.float32)

    np.savez("output/y_norm_stats.npz", mean=y_mean, std=y_std)
    os.chmod('output/y_norm_stats.npz', 0o664)

# Loss functions
def huber_loss(true, pred):
    # Simplified version - just Huber loss on normalized coefficients
    return nn.functional.smooth_l1_loss(true, pred, reduction='mean', beta=0.1)


def save_history(history, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(output_dir, "history.npy")
    np.save(history_path, history, allow_pickle=True)
    try:
        os.chmod(history_path, 0o664)
    except Exception:
        pass



def plot_loss(loss, val_loss, name, output_dir="output"):
    fig, ax = plt.subplots()
    ax.plot(loss, label="train_loss")
    ax.plot(val_loss, label="val_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.set_title("Training loss")
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"{name}.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    try:
        os.chmod(out_path, 0o664)
    except Exception:
        pass


# Setup loss function based on config
if normalize_y:
    y_mean_tensor = torch.from_numpy(y_mean).float()
    y_std_tensor = torch.from_numpy(y_std).float()
else:
    y_mean_tensor = torch.zeros(y_data.shape[1], dtype=torch.float32)
    y_std_tensor = torch.ones(y_data.shape[1], dtype=torch.float32)

if loss_function == "combined":
    phase_loss_weight = config["phase_loss_weight"]  # Váha phase loss (doporučeno 0.1-1.0)
    coef_start_epoch = config["coef_start_epoch"]  # Kdy začít coefficient loss (None = 60% tréninku)
    huber_delta = config["huber_delta"]
    loss_scheduler = LossScheduler(
        config.get("epochs", epochs), 
        y_mean_tensor, 
        y_std_tensor,
        coef_start_epoch=coef_start_epoch,
        normalize_y=normalize_y,
        huber_delta=huber_delta
    )
    criterion = loss_scheduler
elif loss_function == "mse":
    criterion = nn.MSELoss()
elif loss_function == "mae":
    criterion = nn.L1Loss()
elif loss_function == "huber":
    criterion = nn.SmoothL1Loss()
elif weight_loss and loss_function == "mse":
    y_std_weights = 1.0 / (y_std + 1e-8)
    criterion = PerCoefficientWeightedLoss(y_train.numpy(), loss_type="mse")
elif weight_loss and loss_function == "huber":
    y_std_weights = 1.0 / (y_std + 1e-8)
    criterion = PerCoefficientWeightedLoss(y_train.numpy(), loss_type="huber", huber_delta=config["huber_delta"])
elif weight_loss and loss_function == "mae":
    y_std_weights = 1.0 / (y_std + 1e-8)
    criterion = PerCoefficientWeightedLoss(y_train.numpy(), loss_type="mae")
else:
    criterion = nn.MSELoss()  # default fallback

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
y_mean_tensor = y_mean_tensor.to(device)
y_std_tensor = y_std_tensor.to(device)
mask = mask.to(device)
z_basis = z_basis.to(device)

if hasattr(criterion, 'to'):
    criterion.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr"), weight_decay=config.get("weight_decay"))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get("epochs", epochs))

train_losses = []
val_losses = []

train_phase_losses = []
train_coeff_losses = []

val_phase_losses = []
val_coeff_losses = []

for epoch in range(config.get("epochs", epochs)):
    # Update loss scheduler epoch
    if hasattr(criterion, 'update_epoch'):
        criterion.update_epoch(epoch)
    
    model.train()
    train_loss = 0
    train_phase_loss = 0
    train_coeff_loss = 0
    for images, tilts, targets in train_loader:
        images, tilts, targets = images.to(device), tilts.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(images, tilts)
        loss_dict = criterion(targets, preds)
        loss = loss_dict["total_loss"]
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_phase_loss += loss_dict["phase_loss"].item()
        train_coeff_loss += loss_dict["coeff_loss"].item()
    scheduler.step()

    # Validation
    model.eval()
    val_loss = 0
    val_phase_loss = 0
    val_coeff_loss = 0
    with torch.no_grad():
        for images, tilts, targets in val_loader:
            images, tilts, targets = images.to(device), tilts.to(device), targets.to(device)
            preds = model(images, tilts)
            loss_dict = criterion(targets, preds)
            loss = loss_dict["total_loss"]
            val_loss += loss.item()
            val_phase_loss += loss_dict["phase_loss"].item()
            val_coeff_loss += loss_dict["coeff_loss"].item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    train_phase_loss /= len(train_loader)
    train_coeff_loss /= len(train_loader)

    val_phase_loss /= len(val_loader)
    val_coeff_loss /= len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_phase_losses.append(train_phase_loss)
    train_coeff_losses.append(train_coeff_loss)

    val_phase_losses.append(val_phase_loss)
    val_coeff_losses.append(val_coeff_loss)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

print("Training complete.")

# Save history and model
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

history = {
    "loss": train_losses,
    "val_loss": val_losses,

    "phase_loss": train_phase_losses,
    "val_phase_loss": val_phase_losses,

    "coeff_loss": train_coeff_losses,
    "val_coeff_loss": val_coeff_losses,
}
save_history(history, output_dir)

model_path = os.path.join(output_dir, "model.pth")
torch.save(model.state_dict(), model_path)
try:
    os.chmod(model_path, 0o664)
except Exception:
    pass

plot_loss(train_losses, val_losses, "train_loss", output_dir)
plot_loss(train_phase_losses, val_phase_losses, "train_phase_loss", output_dir)
plot_loss(train_coeff_losses, val_coeff_losses, "train_coeff_loss", output_dir)


def plot_predictions_vs_truths(y_true, y_pred, titles, output_dir="output", num_coeffs=6):
    os.makedirs(output_dir, exist_ok=True)
    num_plots = min(num_coeffs, y_true.shape[1])
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    axes = axes.ravel()
    for i in range(num_plots):
        ax = axes[i]
        ax.scatter(np.arange(y_true.shape[0]), y_true[:, i], label="truth", s=10)
        ax.scatter(np.arange(y_pred.shape[0]), y_pred[:, i], label="prediction", s=10)
        ax.errorbar(np.arange(y_true.shape[0]), (y_true[:, i]+y_pred[:, i])/2, yerr=np.abs((y_true[:, i] - y_pred[:, i])/2), fmt='s', alpha=0.5)
        ax.set_title(titles[i])
        ax.set_ylabel("value")
        ax.legend(fontsize="small")
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    out_path = os.path.join(output_dir, "predictions_vs_truths_first_6_coeffs.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    try:
        os.chmod(out_path, 0o664)
    except Exception:
        pass


titles = [
    #"coef_1 - Tilt X",
    #"coef_2 - Tilt Y",
    "coef_3 - Defocus",
    "coef_4 - Primary astigmatism - oblique",
    "coef_5 - Primary astigmatism - vert.",
    "coef_6 - Primary coma - X",
    "coef_7 - Primary coma - Y",
    "coef_8 - Primary spherical aberration"
]

# Evaluate on test set and create visualizations
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for images, tilts, targets in test_loader:
        images, tilts = images.to(device), tilts.to(device)
        preds = model(images, tilts)
        all_preds.append(preds.cpu())
        all_targets.append(targets)

if all_preds:
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    if normalize_y:
        y_mean = y_mean.astype(np.float32)
        y_std = y_std.astype(np.float32)
        all_preds = all_preds * y_std + y_mean
        #all_targets = all_targets * y_std + y_mean

    plot_predictions_vs_truths(all_targets[:50], all_preds[:50], titles, output_dir)
    print(f"Saved training history and visualization to {output_dir}")


def MAE(X, y):
    return np.mean(np.abs(y - X))

def STD(X, y):
    return np.std(y - X)

maes = np.array([])
stds = np.array([])

for i in range(np.shape(all_preds)[1]):
        mae = MAE(all_preds[:,i],all_targets[:,i])
        std = STD(all_preds[:,i],all_targets[:,i])
        maes = np.append(maes, mae)
        stds = np.append(stds, std)

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
plt.savefig('output/maes_with_stds.pdf')
plt.clf()

os.chmod('output/maes_with_stds.pdf', 0o664)
