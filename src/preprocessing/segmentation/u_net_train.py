

""" Structure of dataset directory
dataset/
 ├── train/
 │    ├── images/
 │    │     ├── 001.png
 │    │     ├── 002.png
 │    └── masks/
 │          ├── 001.png
 │          ├── 002.png
 ├── val/
 │    ├── images/
 │    └── masks/
"""

import os

cwd = os.getcwd()
if cwd.startswith("/home/poky/"):
    print("Running in local environment")
    local = True
else:    
    print("Running in server environment")
    local = False


os.makedirs("output", exist_ok=True)

# ===========================================
# U-Net trénink pro binární masky interferogramů
# ===========================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
#from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp

import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import json

# ---- 1. Dataset ----
class InterferogramDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = np.array(image)
        image = np.repeat(image[..., None], 3, axis=2)  # L → RGB (tři kopie)
        image = Image.fromarray(image)


        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        mask = (mask > 0.5).float()  # binární maska
        return image, mask




class ResizeWithPadding:
    """Změní velikost obrázku na cílový rozměr se zachováním poměru stran (doplní okraje)."""
    def __init__(self, size, fill=0):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.fill = fill  # barva výplně (0 = černá)

    def __call__(self, img):
        # původní rozměry
        w, h = img.size
        target_w, target_h = self.size

        # poměry stran
        ratio = min(target_w / w, target_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)

        # změna velikosti se zachováním poměru
        img = img.resize((new_w, new_h), Image.NEAREST if img.mode == 'L' else Image.BILINEAR)

        # doplnění okrajů (zarovnání na střed)
        delta_w = target_w - new_w
        delta_h = target_h - new_h
        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
        img = ImageOps.expand(img, padding, fill=self.fill)

        return img


# Load config

def load_confing(path="config.json"):
    with open(path, 'r') as file:
        config = json.load(file)

        return config

if local:
    config_path = "/home/poky/Dokumenty/ML-Interferometry/src/training/segmentation/train_config.json"
else:    
    config_path = "train_config.json"

config = load_confing(config_path)

num_of_sets = config["num_of_sets"]
mode = config["mode"]
model_num_unet = config["model_num_unet"]
pretrained = config["pretrained"]
version = config["version"]

loss_function = config["loss_function"]
max_resolution = config["max_resolution"]
epochs = config["epochs"]
batch_size = config["batch_size"]
learning_rate = config["learning_rate"]

output_base_path = config["output_base_path"]
 
dataset = config["dataset"]

model_name = f"unet_interferogram_{model_num_unet}_v{version}.pth"

if local:
    datadir = f"/home/poky/Dokumenty/ML-Interferometry/data/preprocessing/mask/{dataset}"
    model_save_path = os.path.join("/home/poky/Dokumenty/ML-Interferometry/models", model_name)

else:
    datadir = os.path.join(output_base_path, mode, "data", dataset)
    model_save_path = os.path.join("output", model_name)

# ---- 2. Transformace ----
transform = transforms.Compose([
    ResizeWithPadding((256, 256)),
    transforms.ToTensor(),
])

train_dataset = InterferogramDataset(os.path.join(datadir, "train", "images"), os.path.join(datadir, "train", "masks"), transform)
val_dataset   = InterferogramDataset(os.path.join(datadir, "val", "images"), os.path.join(datadir, "val", "masks"), transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

models = {
    1: "resnet18",
    2: "resnet34",
    3: "resnet101"
    }

# ---- 3. Model ----
model = smp.Unet(
    encoder_name=models[model_num_unet],        # lehká backbone
    encoder_weights="imagenet" if pretrained else None,     # transfer learning
    in_channels=3,                  # RGB vstup
    classes=1,                      # binární výstup
)

# ---- 4. Loss, optimizer ----
loss_fn = smp.losses.DiceLoss(mode='binary')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Inicializace polí pro logování loss
train_losses = []
val_losses = []
    

best_val_loss = float('inf')
# ---- 5. Trénink ----
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = loss_fn(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Výpočet val_loss pro uložení nejlepšího modelu
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for val_imgs, val_masks in val_loader:
            val_imgs, val_masks = val_imgs.to(device), val_masks.to(device)
            val_preds = model(val_imgs)
            val_loss += loss_fn(val_preds, val_masks).item()
    model.train()

    # Uložit nejlepší model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_save_path)
        os.chmod(model_save_path, 0o664)
        print(f"✓ Model uložen (val_loss: {val_loss/len(val_loader):.4f})")
    
            # Logování loss hodnot
    train_losses.append(total_loss)
    val_losses.append(val_loss)
        

    print(f"Epoch {epoch+1}, loss: {total_loss/len(train_loader):.4f}, val_loss: {val_loss/len(val_loader):.4f}")

print(f"Trénink dokončen. Nejlepší model uložen jako {model_save_path} s val_loss: {best_val_loss/len(val_loader):.4f}")

# Uložení loss historii jako numpy pole
train_losses_np = np.array(train_losses)
val_losses_np = np.array(val_losses)
    

loss_history_path = os.path.join("output", f"loss_history_unet_{model_num_unet}.npy")
np.save(loss_history_path, {"train_losses": train_losses_np, "val_losses": val_losses_np})
os.chmod(loss_history_path, 0o664)

# ---- 6. Uložení modelu ----
#torch.save(model.state_dict(), "unet_interferogram_2_0.pth")
#print("Model uložen.")


# Model info: U-Net modely mají jména unet_interferogram_A_B.pth
# A je verze architektury (1 = resnet18, 2 = resnet34, 3 = resnet50)
# B je verze tréninku, která prakticky znamená změnu datové sady nebo tréninkových parametrů
