
#import os
#import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_and_crop_via_edges(path: str,
                              blur_ksize: int=5,
                              canny_t1: int=50, canny_t2: int=150,
                              dilate_kernel: int=7,
                              closing_kernel: int=15,
                              min_area_ratio: float=0.01,
                              use_convex_hull: bool=True) -> dict:
    """
    Detekce interferogramu z hran: Canny -> dilatace -> closing -> fill -> bounding box.
    - dilate_kernel, closing_kernel: should be odd integers; increase if fringes are thin and sparse.
    - min_area_ratio: minimal area (fraction of image) to consider (filter small noise).
    - use_convex_hull: if True, compute convex hull of union of contours before bbox (good for square shapes).
    Returns: dict with images and bbox/roi.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or cannot be read.")
    h, w = img.shape

    # 1) Blur -> reduce noise
    img_blur = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    # 2) Canny edges
    edges = cv2.Canny(img_blur, canny_t1, canny_t2)

    # 3) Dilate edges to thicken them (helps join nearby fringes)
    dk = max(1, int(dilate_kernel))
    kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
    edges_dilated = cv2.dilate(edges, kernel_d)

    # 4) Closing to fill small gaps between dilated edges
    ck = max(1, int(closing_kernel))
    if ck % 2 == 0:
        ck += 1
    kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
    mask_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel_c)

    # 5) Fill holes: find contours on mask_closed and draw filled
    contours, _ = cv2.findContours(mask_closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask_closed)
    cv2.drawContours(mask_filled, contours, -1, 255, thickness=cv2.FILLED)

    # 6) Filter small components by area
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_filled, connectivity=8)
    # Compute minimal allowed area
    min_area = min_area_ratio * h * w
    large_mask = np.zeros_like(mask_filled)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            large_mask[labels == i] = 255

    # If nothing remains, fallback to mask_filled
    if large_mask.sum() == 0:
        large_mask = mask_filled.copy()

    # 7) Option: compute convex hull of union of contours to get tight, convex shape
    if use_convex_hull:
        cnts, _ = cv2.findContours(large_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_pts = np.vstack(cnts) if len(cnts) > 0 else None
        if all_pts is not None and all_pts.size > 0:
            hull = cv2.convexHull(all_pts)
            hull_mask = np.zeros_like(large_mask)
            cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)
            final_mask = hull_mask
        else:
            final_mask = large_mask
    else:
        final_mask = large_mask

    # 8) Bounding box of final_mask
    contours_final, _ = cv2.findContours(final_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_final) == 0:
        raise RuntimeError("No contours found after processing.")
    cnt = max(contours_final, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(cnt)

    # ensure bbox in image bounds
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + bw), min(h, y + bh)

    roi = img[y1:y2, x1:x2]

    # prepare visualization images
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return {
        "img": img,
        "img_blur": img_blur,
        "edges": edges,
        "edges_dilated": edges_dilated,
        "mask_closed": mask_closed,
        "mask_filled": mask_filled,
        "final_mask": final_mask,
        "bbox": (x1, y1, x2, y2),
        "roi": roi,
        "vis": vis
    }


# === Example usage ===
if __name__ == "__main__":
    p = "src/preprocessing/ImageSource_00_00.bmp"
    out = detect_and_crop_via_edges(p,
                                   blur_ksize=9,  # velikost jádra pro Gaussovo rozmazání (liché). Větší hodnoty potlačí šum, ale rozmažou jemné interferenční proužky.
                                   canny_t1=100,  # Canny - nižší práh: citlivost na slabší hrany (nižší -> detekuje více hran, více šumu)
                                   canny_t2=250,  # Canny - vyšší práh: prah pro pevné hrany (vyšší -> detekuje jen silné hrany)
                                   dilate_kernel=9,  # velikost jádra pro dilataci (v pixelech). Zvýší tloušťku hran a spojuje blízké úseky.
                                   closing_kernel=51,  # velikost jádra pro morfologické closing (liché). Vyplní mezery mezi dilatovanými hranami; větší -> vyplní větší díry.
                                   min_area_ratio=0.04,  # minimální plocha komponenty jako podíl z celkové plochy obrázku; menší hodnoty ponechají drobnější objekty
                                   use_convex_hull=True)  # pokud True, spojí nalezené kontury do konvexního obalu (hladší, kompaktnější maska pro bbox)
    
    print(np.shape(out["img"]))
    print(np.shape(out["roi"]))

    """plt.figure(figsize=(12,8))
    plt.subplot(2,3,1); plt.imshow(out["img"], cmap="gray"); plt.title("Původní snímek"); plt.axis("off")
    plt.subplot(2,3,2); plt.imshow(out["img_blur"], cmap="gray"); plt.title("Vyhlazený snímek"); plt.axis("off")
    plt.subplot(2,3,3); plt.imshow(out["edges"], cmap="gray"); plt.title("Hrany"); plt.axis("off")
    plt.subplot(2,3,4); plt.imshow(out["final_mask"], cmap="gray"); plt.title("Finální maska"); plt.axis("off")
    plt.subplot(2,3,5); plt.imshow(out["vis"][..., ::-1]); plt.title("Snímek s ohraničujicím obdélníkem"); plt.axis("off")
    plt.subplot(2,3,6); plt.imshow(out["roi"], cmap="gray"); plt.title("Ořízlý interferogram"); plt.axis("off")
    plt.tight_layout()
    plt.show()"""


    plt.figure(figsize=(8,8))
    plt.subplot(2,2,1); plt.imshow(out["img"], cmap="gray"); plt.title("Původní snímek"); plt.axis("off")
    plt.subplot(2,2,2); plt.imshow(out["img_blur"], cmap="gray"); plt.title("Vyhlazený snímek"); plt.axis("off")
    plt.subplot(2,2,3); plt.imshow(out["edges"], cmap="gray"); plt.title("Hrany"); plt.axis("off")
    plt.subplot(2,2,4); plt.imshow(out["final_mask"], cmap="gray"); plt.title("Finální maska"); plt.axis("off")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,8))
    plt.subplot(2,2,1); plt.imshow(out["img"], cmap="gray"); plt.title("Původní snímek"); plt.axis("off")
    plt.subplot(2,2,2); plt.imshow(out["final_mask"], cmap="gray"); plt.title("Finální maska"); plt.axis("off")
    plt.subplot(2,2,3); plt.imshow(out["vis"][..., ::-1]); plt.title("Snímek s ohraničujicím obdélníkem"); plt.axis("off")
    plt.subplot(2,2,4); plt.imshow(out["roi"], cmap="gray"); plt.title("Ořízlý interferogram"); plt.axis("off")
    plt.tight_layout()
    plt.show()

    """   # Uložení rovnou do souboru
    output_path = os.path.join(os.path.dirname(__file__), "edge_detection_output.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Výsledek uložen do: {output_path}")
    plt.close('all')
    """