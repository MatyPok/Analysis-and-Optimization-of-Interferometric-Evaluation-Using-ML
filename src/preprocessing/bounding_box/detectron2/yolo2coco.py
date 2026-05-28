import os
import json
from PIL import Image

# Cesty
#images_dir = "bounding_box/dataset/images"   # složka s obrázky
#labels_dir = "bounding_box/dataset/labels"   # složka s YOLO .txt
#output_json = "bounding_box/dataset/annotations/instances_train.json"


# Images directory is current directory
#images_dir = os.getcwd()
#labels_dir = images_dir  # Assuming labels are in the same directory

images_dir = "data/preprocessing/bounding_box/unet_dataset_4/test/"
labels_dir = "data/preprocessing/bounding_box/unet_dataset_4/yolo_labels/test/"

category_name = os.path.basename(images_dir)  # Use the name of the images directory as category name
output_json = os.path.join(images_dir, f"instances_{category_name}.json") # Output JSON file, named after the images directory

# Definice tříd (podle pořadí class_id v YOLO)
classes = ["interferogram"]  # můžeš přidat víc

coco_output = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Přidáme kategorie
for idx, name in enumerate(classes):
    coco_output["categories"].append({
        "id": idx,
        "name": name,
        "supercategory": "none"
    })

annotation_id = 1
image_id = 1

for filename in os.listdir(images_dir):
    if not (filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".bmp")):
        continue
    
    img_path = os.path.join(images_dir, filename)
    label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")

    # Načti rozměry obrázku
    with Image.open(img_path) as img:
        width, height = img.size

    coco_output["images"].append({
        "id": image_id,
        "file_name": filename,
        "width": width,
        "height": height
    })

    # Zpracuj labely (YOLO formát)
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, w, h = map(float, parts)
                class_id = int(class_id)

                # Přepočet z normalizovaných souřadnic do COCO bbox formátu [xmin, ymin, width, height]
                xmin = (x_center - w/2) * width
                ymin = (y_center - h/2) * height
                bw = w * width
                bh = h * height

                coco_output["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [xmin, ymin, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0
                })
                annotation_id += 1

    image_id += 1

# Ulož do JSONu
os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, "w") as f:
    json.dump(coco_output, f, indent=2)

print(f"COCO anotace uložena do {output_json}")
