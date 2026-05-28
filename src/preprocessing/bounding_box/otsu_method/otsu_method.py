import cv2
import numpy as np
import matplotlib.pyplot as plt

# ======================
# === Parametry ===
# ======================
image_path = "src/preprocessing/ImageSource_00_00.bmp"

# Velikost jádra pro morfologické operace
morph_kernel_size = 9

# Jestli ignorovat pixely s hodnotou 0 (černé okraje)
ignore_zeros = True

blur_size = 5

def Otsu(image, morph_kernel_size:int=3, blur_size:int=5, ignore_zeros:bool=True) -> np.ndarray:
    """
    Funkce pro detekci a oříznutí interferogramu z obrázku pomocí Otsuova prahování
    a morfologických operací.

    Parametry:
    - image: cesta k obrázku nebo již načtený obrázek v odstínech šedi jako numpy array
    - morph_kernel_size: velikost jádra pro morfologické operace (musí být liché číslo)
    - ignore_zeros: zda ignorovat pixely s hodnotou 0 při výpočtu Otsuova prahu

    Vrací:
    - slovník s klíči:
        - "img": původní obrázek
        - "mask": binární maska po Otsu
        - "mask_clean": vyčištěná maska po morfologických operacích
        - "bbox": souřadnice ohraničujícího boxu (x, y, x+w, y+h)
        - "roi": vyříznutý interferogram
        - "vis": vizualizace s ohraničujícím boxem
    """
    # ======================
    # === 1. Načtení obrázku ===

    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Obrázek se nepodařilo načíst. Zkontrolujte cestu k souboru.")
    else:
        img = image.copy()
        if len(img.shape) != 2:
            raise ValueError("Poskytnutý obrázek musí být v odstínech šedi (2D numpy array).") 
    # ======================

    # 1.5. Vyhlazení obrazu Gaussovským filtrem
    img = cv2.GaussianBlur(img, (blur_size, blur_size), 0)

    # ======================
    # === 2. Otsuovo prahování ===
    # ======================
    # Pokud chceme ignorovat černé pixely (0), vytvoříme masku jen pro nenulové hodnoty
    if ignore_zeros:
        # extrakce jen nenulových pixelů
        nonzero_pixels = img[img > 0]
        # Otsu na těchto hodnotách
        thresh_val, _ = cv2.threshold(nonzero_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # pak použijeme tento práh na celý obraz
        _, mask = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)
    else:
        # klasické Otsu na celém obraze
        thresh_val, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print(f"Otsuův práh: {thresh_val}")

    # ======================
    # === 3. Morfologické čištění ===
    # ======================
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ======================
    # === 4. Analýza komponent ===
    # ======================
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean)

    # vybrání největší komponenty (ignorujeme label 0 = pozadí)
    largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, w, h, area = stats[largest_component]

    # bounding box do kopie původního obrázku
    img_bbox = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_bbox, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # ======================
    # === 5. Vyříznutí ROI ===
    # ======================
    roi = img[y:y+h, x:x+w]

    # ======================
    # === 6. Vizualizace výsledků ===
    # ======================
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Původní obraz")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Maska po Otsu")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(mask_clean, cmap="gray")
    plt.title("Vyčištěná maska")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(img_bbox[..., ::-1])
    plt.title("Bounding box")
    plt.axis("off")

    # Histogram s černými okraji
    plt.subplot(2, 3, 5)
    plt.hist(img.ravel(), bins=256, range=(0, 256), alpha=0.6, label="Všechny pixely")
    if ignore_zeros:
        plt.hist(nonzero_pixels.ravel(), bins=256, range=(0, 256), alpha=0.6, label="Bez nul")
    plt.axvline(thresh_val, color="r", linestyle="--", label=f"Prahová hodnota = {thresh_val:.0f}")
    plt.title("Histogram intenzit")
    plt.legend()

    plt.subplot(2, 3, 6)
    plt.imshow(roi, cmap="gray")
    plt.title("Oříznutý interferogram")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    #plt.savefig("src/preprocessing/out_otsu.png")

    return roi

roi = Otsu(image_path, morph_kernel_size, blur_size, ignore_zeros)
#roi = Otsu(roi, morph_kernel_size=3, ignore_zeros=True)