import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

model_output_dir = "src/preprocessing/bounding_box/results/output_2"

# === 1️⃣ Registrace datasetu (aby vizualizace znala metadata) ===

register_coco_instances(
    "interferogram_val",
    {},
    "data/preprocessing/bounding_box/unet_dataset_4/annotations/instances_val.json", 
    "data/preprocessing/bounding_box/unet_dataset_4/val")

metadata = MetadataCatalog.get("interferogram_val")


# === 2️⃣ Načtení konfigurace ===
cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"   # pokud máš GPU, dej "cuda"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))

# musíš nastavit stejné parametry, jaké jsi měl při tréninku:
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

# cesta k uloženému modelu
cfg.MODEL.WEIGHTS = os.path.join(model_output_dir, "model_best.pth")

# === 3️⃣ Načtení modelu ===
predictor = DefaultPredictor(cfg)

# === 4️⃣ Načtení testovacích obrázků ===
test_image_path = "data/preprocessing/bounding_box/unet_dataset_4/test"
test_annotations_path = "data/preprocessing/bounding_box/unet_dataset_4/annotations/instances_test.json"

image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG']
test_file_list = [path for ext in image_extensions for path in Path(test_image_path).rglob(ext)]
print(f"Total test images: {len(test_file_list)}")

# Pokud nejsou testovací obrázky
if len(test_file_list) == 0:
    print(f"\n⚠️  Žádné testovací obrázky nenalezeny!")
    print(f"Očekávaná cesta: {test_image_path}")
    print(f"Prosím vlož testovací obrázky do tohoto adresáře.")
    print(f"Podporované formáty: {', '.join(image_extensions)}")
    import sys
    sys.exit(1)

# === 5️⃣ Test na všech obrázcích ===
output_dir = os.path.join(model_output_dir, "test_results")
os.makedirs(output_dir, exist_ok=True)

# Připraví se seznam vizualizací pro zobrazení
visualizations = []
num_to_display = min(4, len(test_file_list))  # Zobraz max 4 obrázky

def IoU(box_truth, box_pred):
    xA = max(box_truth[0], box_pred[0])
    yA = max(box_truth[1], box_pred[1])
    xB = min(box_truth[2], box_pred[2])
    yB = min(box_truth[3], box_pred[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (box_truth[2] - box_truth[0] + 1) * (box_truth[3] - box_truth[1] + 1)
    boxBArea = (box_pred[2] - box_pred[0] + 1) * (box_pred[3] - box_pred[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

for idx, img_path in enumerate(test_file_list):
    img = cv2.imread(str(img_path))
    outputs = predictor(img)

    # Načti ground truth boxy z COCO anotací
    from pycoco_utils import COCO
    coco = COCO(test_annotations_path)
    img_id = coco.getImgIds(imgIds=[int(img_path.stem)])[0]  # předpokládáme, že název souboru je ID obrázku
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)
    gt_boxes = [ann['bbox'] for ann in anns]  # COCO formát: [x, y, width, height]
    gt_boxes = [[x, y, x + w, y + h] for (x, y, w, h) in gt_boxes]  # převod na [x1, y1, x2, y2]

    # Vypsat výsledky
    print(f"\n📷 {img_path.name}")
    print("Bounding boxes:", outputs["instances"].pred_boxes)
    print("Scores:", outputs["instances"].scores)

    # Vizualizace
    #v = Visualizer(img[:, :, ::-1], metadata=metadata)
    #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #out_img = out.get_image()[:, :, ::-1]

    from detectron2.utils.visualizer import _create_text_labels, ColorMode

    instances = outputs["instances"].to("cpu")
    colors = [(1.0, 0.0, 0.0)] * len(instances)  # červené boxy

    v = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    out = v.overlay_instances(
        boxes=instances.pred_boxes,
        labels=_create_text_labels(
            instances.pred_classes,
            instances.scores,
            metadata.get("thing_classes", None)
        ),
        assigned_colors=colors
    )
    out_img = out.get_image()[:, :, ::-1]

    # Uložení vizualizace
    out_name = img_path.name.split('.')[0] + ".png" # ulož jako PNG
    cv2.imwrite(os.path.join(output_dir, f"pred_{out_name}"), out_img)

    # Uložení ROI
    boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        roi = img[int(y1):int(y2), int(x1):int(x2)]
        name = img_path.name.split('.')[0] # bez přípony
        if i > 0:
            cv2.imwrite(os.path.join(output_dir, f"pred_{name}_roi_{i}.png"), roi)
        else:
            cv2.imwrite(os.path.join(output_dir, f"pred_{name}_roi.png"), roi)

    # Přidej do seznamu pro vizualizaci (pouze prvních N obrázků)
    if idx < num_to_display:
        visualizations.append({
            'image': cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            'boxes': boxes,
            'scores': instances.scores.numpy(),
            'name': img_path.name
        })

    # Volitelné zobrazení
    #cv2.imshow("result", out_img)
    #cv2.waitKey(0)

try:
    cv2.destroyAllWindows()
except cv2.error:
    pass  # Bez GUI podpory

if visualizations:
    num_results = len(visualizations)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for plot_idx, vis_data in enumerate(visualizations):
        ax = axes[plot_idx]
        ax.imshow(vis_data['image'])
        ax.set_title(f"{vis_data['name']}", fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Vykreslení boxů
        boxes = vis_data['boxes']
        scores = vis_data['scores']
        
        for box_idx, (x1, y1, x2, y2) in enumerate(boxes):
            width = x2 - x1
            height = y2 - y1
            
            # Červený obdélník
            rect = patches.Rectangle(
                (x1, y1), width, height, 
                linewidth=2, 
                edgecolor='red', 
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Skóre
            score = scores[box_idx]
            ax.text(
                x1, y1 - 5, 
                f'{score:.2f}', 
                color='red', 
                fontsize=10, 
                fontweight='bold',
                bbox=dict(facecolor='yellow', alpha=0.7, pad=1)
            )
    
    # Skryj prázdné subploty
    for plot_idx in range(num_results, 4):
        axes[plot_idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "results_preview.png"), dpi=150, bbox_inches='tight')
    print(f"\n📊 Vizualizace uložena: {os.path.join(output_dir, 'results_preview.png')}")
    plt.show()

print(f"\n✅ Hotovo. Výsledky jsou uložené v: {output_dir}")
