"""
Inference skript - pipeline U-Net + Refiner
=============================================
Demonstrace kompletního pipeline s oběma modely.
"""

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import numpy as np
import os
from refiner_train import MaskRefiner1, MaskRefiner2, ResidualRefiner
from matplotlib import pyplot as plt


class ResizeWithPadding:
    """Změní velikost obrázku na 256x256 se zachováním poměru stran."""
    def __init__(self, size=(256, 256), fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        target_w, target_h = self.size

        ratio = min(target_w / w, target_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)

        img = img.resize((new_w, new_h), Image.NEAREST if img.mode == 'L' else Image.BILINEAR)

        delta_w = target_w - new_w
        delta_h = target_h - new_h
        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
        img = ImageOps.expand(img, padding, fill=self.fill)

        return img


def load_models(device, unet_path, model_num):
    """Načte U-Net a Refiner modely."""
    # U-Net

    models = {
    1: "resnet18",
    2: "resnet34",
    3: "resnet101"
    }


    unet = smp.Unet(
        encoder_name=models[model_num],
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    
    if os.path.exists(unet_path):
        unet.load_state_dict(torch.load(unet_path, map_location=device))
        print("✓ U-Net model načten")
    else:
        print("U-Net model nenalezen")
    
   
    unet.to(device)
    
    return unet


@torch.no_grad()
def predict(image_path, gt_mask_path, unet, device):
    """
    Predikuje masku se úplným pipeline.
    
    Kroky:
    1. Načti obrázek (originální rozměr)
    2. U-Net: 256x256 → maska 256x256
    3. Interpolace: masku zpět na orig. velikost
    
    Args:
        image_path: cesta k obrázku
        unet: U-Net model
        device: torch device
    
    Returns:
        tuple: (orig_image, unet_interp_mask, mask_quality)
    """
    
    # Načti obrázek
    image = Image.open(image_path).convert("L")
    orig_size = image.size  # (W, H)
    
    print(f"\nObrázek: {os.path.basename(image_path)}")
    print(f"Původní rozměry: {orig_size[0]} x {orig_size[1]} px")

    # Načti GT masku (pro vizualizaci)
    gt_mask = Image.open(gt_mask_path).convert("L")

    # Konverze na RGB
    image_np = np.array(image)
    image_np = np.repeat(image_np[..., None], 3, axis=2)
    image_rgb = Image.fromarray(image_np)
    
    # Resize na 256x256
    resize_transform = ResizeWithPadding((256, 256))
    image_256 = resize_transform(image_rgb)
    image_256_tensor = transforms.ToTensor()(image_256).unsqueeze(0).to(device)  # (1, 3, 256, 256)
    
    # ---- KROK 1: U-Net predikce ----
    print("1. U-Net predikce (256x256)...", end=" ")
    pred_unet = unet(image_256_tensor)  # (1, 1, 256, 256)
    print("✓")

    # ---- KROK 1.1: Sigmoid a binarizace ----
    pred_unet = torch.sigmoid(pred_unet)
    pred_unet = (pred_unet > 0.5).float()  # Binarizace

    # ---- KROK 2: Interpolace na orig. velikost ----
    print(f"2. Interpolace na orig. velikost ({orig_size[0]} x {orig_size[1]})...", end=" ")
    target_size = (orig_size[1], orig_size[0])  # PyTorch: (H, W)

    """
    pred_interp = F.interpolate(pred_unet, size=target_size, mode='bilinear', align_corners=False) # (1, 1, H, W)
    print("✓")
    
    # Konverze na PIL
    pred_interp_np = pred_interp.squeeze().cpu().numpy()
    unet_mask_pil = Image.fromarray((pred_interp_np * 255).astype(np.uint8))
    """

    # Konverze na PIL a interpolace v PIL
    pred_unet_np = pred_unet.squeeze().cpu().numpy()
    unet_mask_pil = Image.fromarray((pred_unet_np * 255).astype(np.uint8))
    unet_interp_pil = unet_mask_pil.resize(orig_size, resample=Image.NEAREST)
    pred_interp_np = np.array(unet_interp_pil) / 255.0
    print("✓")
    
    # ---- Qualita masky jako Intersection over Union (IoU) ----
    # Převod GT masky na numpy
    gt_mask_np = np.array(gt_mask.convert("L")) > 128  # Binarizace
    pred_interp_np = pred_interp_np > 0.5  # Binarizace

    intersection = np.sum(gt_mask_np & pred_interp_np)
    union = np.sum(gt_mask_np | pred_interp_np)

    mask_quality = intersection / (union + 1e-8)  # Přidej malé číslo kvůli dělení nulou

    # Hodnocení podle absolutní hodnoty špatných pixelů
    diff = np.abs(gt_mask_np.astype(int) - pred_interp_np.astype(int))
    num_of_all_pixels = diff.size
    diff = np.sum(diff, axis=0)  # Součet přes kanály
    diff = np.sum(diff)
    wrong_pixels = diff / num_of_all_pixels  # Počet špatných pixelů v poměru k celkovému počtu pixelů


    # Vytvoření vizualizace - maska na původním obrázku
    # Create red overlay
    overlay = Image.new("RGB", image.size, (255, 0, 0))  # červená

    # binární maska (0 nebo 255)
    #mask_np = (pred_refiner_np > 0.5).astype(np.uint8) * 255
    #mask_pil = Image.fromarray(mask_np)

    # vytvoř alpha masku (zeslab ji → průhlednost)
    alpha = unet_interp_pil.point(lambda p: int(p * 0.4))  # 0.4 = průhlednost

    # převedení originálu na RGB
    image_rgb = image.convert("RGB")

    # složení overlaye pouze tam, kde je maska
    mask_on_image = image_rgb.copy()
    mask_on_image.paste(overlay, (0, 0), alpha)
    
    return image, unet_interp_pil, gt_mask, mask_quality, diff, wrong_pixels, num_of_all_pixels, mask_on_image


def visualize_results(orig_image, unet_mask, gt_mask, mask_quality, diff, wrong_pixels, num_of_all_pixels, mask_on_image, output_path=None):
    """Vytvoří vizualizaci výsledků."""
    from PIL import Image as PILImage, ImageDraw, ImageFont
    
    # Uprav rozměry
    width, height = orig_image.size

    # Připrav font
    font_size = height // 20
    fnt = ImageFont.load_default(size=font_size)

    # Vytvoř koláž s obrázky vedle sebe
    total_width = width * 2  # 4 obrázky + mezery
    max_height = height * 2 + font_size*2  # Místo pro popis
    
    canvas = PILImage.new('RGB', (total_width, max_height), color=(0, 0, 0))
    
    # Vlož obrázky
    canvas.paste(orig_image, (0, 0))
    canvas.paste(gt_mask, (width, 0))
    canvas.paste(unet_mask, (0, height))
    canvas.paste(mask_on_image, (width, height))

    # Přidej popis - Zvětšený font aby byl čitelný
    draw = ImageDraw.Draw(canvas)
    draw.text((font_size/4, font_size/4), f"a", fill=(255, 255, 255), font=fnt)
    draw.text((width + font_size/2, font_size/4), f"b", fill=(255, 255, 255), font=fnt)
    draw.text((font_size/4, height + font_size/4), f"c", fill=(255, 255, 255), font=fnt)
    draw.text((width + font_size/2, height + font_size/4), f"d", fill=(255, 255, 255), font=fnt)

    draw.text((font_size/4, 2*height + font_size/2), f"IoU: {mask_quality:.2f}, Wrong Pixels: {diff}, All Pixels: {num_of_all_pixels}, Wrong to All Pixels Ratio: {wrong_pixels*100:.1f}%", fill=(255, 255, 255), font=fnt)

    if output_path:
        canvas.save(output_path)
        print(f"✓ Vizualizace uložena: {output_path}")
    
    return canvas


def load_show_history(history_path, output_path):
    history = np.load(history_path, allow_pickle=True)[()]

    loss = np.array(history["train_losses"])
    #loss = loss[loss < 1.5]

    val_loss = np.array(history["val_losses"])
    #val_loss = val_loss[val_loss < 1.5]


    fig, ax = plt.subplots()
    plt.plot(loss[5:], label='loss')
    plt.plot(val_loss[5:], label='val_loss')
    ax.set_xlabel('iterations')
    ax.set_ylabel('loss')
    ax.legend()
    plt.savefig(output_path + "/training_history.png")
    plt.show()


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    import glob

    # The dataset the model was trained on - only for naming the results directory
    dataset = "unet_dataset_6"

    # As the dataset_5 is a combination of dataset_4 and dataset_6, we can always use thhe dataset_5
    test_dir = f"data/preprocessing/mask/unet_dataset_5/test/images"
    model_num = 3

    unet_path = f"meta_output/segmentation/dataset_5_6/unet_m{model_num}_pretrained_{dataset}_v1/output/unet_interferogram_{model_num}_v1.pth"
    
    history_path = f"meta_output/segmentation/dataset_5_6/unet_m{model_num}_pretrained_{dataset}_v1/output/loss_history_unet_{model_num}.npy"

    # Inference results directory
    INFERENCE_DIR = f"results/{dataset}/inference_results_unet_m{model_num}_{dataset}"
    os.makedirs(INFERENCE_DIR, exist_ok=True)

    load_show_history(history_path, INFERENCE_DIR)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}\n")
    
    # Načti modely
    print("Načítám modely...")
    unet = load_models(DEVICE, unet_path, model_num)
    
    # Testovací obrázky
    
    test_images = sorted(glob.glob(os.path.join(test_dir, "*")))
    
    if not test_images:
        print(f"❌ Žádné testovací obrázky v {test_dir}")
        exit(1)
    
    ious = []
    wrong_pixel_ratios = []
    
    for img_path in test_images:
        img_name = os.path.basename(img_path).replace(".", "_")
        
        # Predikce
        gt_mask_path = img_path.replace("/images/", "/masks/")
        orig_img, unet_mask, gt_mask, quality, diff, wrong_pixels, num_of_all_pixels, mask_on_image = predict(img_path, gt_mask_path, unet, DEVICE)
        
        # Vizualizace
        output_path = f"{INFERENCE_DIR}/{img_name}_result.png"
        visualize_results(orig_img, unet_mask, gt_mask, quality, diff, wrong_pixels, num_of_all_pixels, mask_on_image, output_path)
        print(f"   Kvalita masky: {quality:.4f}")
        print(f"Počet špatných pixelů: {diff}")
        print(f"Poměr špatných pixelů: {wrong_pixels*100:.2f}%")

        ious.append(quality)
        wrong_pixel_ratios.append(wrong_pixels)
        



    # Print quality metrics in text file
    with open(f"{INFERENCE_DIR}/quality_metrics.txt", "w") as f:
        f.write("Quality Metrics\n")
        f.write("=" * 30 + "\n")
        f.write(f"Average IoU: {np.mean(ious):.4f}\n")
        f.write(f"Average Wrong Pixels Ratio: {np.mean(wrong_pixel_ratios)*100:.2f}%\n")
        f.write("=" * 30 + "\n")
        f.write("\nDetailed per-image metrics:\n")
        for img_path, iou, wrong_ratio in zip(test_images, ious, wrong_pixel_ratios):
            img_name = os.path.basename(img_path)
            f.write(f"{img_name}: IoU={iou:.4f}, Wrong Pixels Ratio={wrong_ratio*100:.2f}%\n")


    print("\n✓ Inference hotov!")
    print("Výsledky v: " + INFERENCE_DIR + "/")
