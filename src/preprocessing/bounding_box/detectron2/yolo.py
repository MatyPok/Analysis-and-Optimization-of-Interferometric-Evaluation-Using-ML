def xyxy_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    # vstup: pixelové souřadnice (xmin,ymin,xmax,ymax)
    # výstup: normalized (x_center, y_center, width, height)
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height

# Příklad: zapsání do souboru
def write_yolo_label(filename_txt, class_id, xmin, ymin, xmax, ymax, img_w, img_h):
    x_c, y_c, w, h = xyxy_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
    with open(filename_txt, "w") as f:
        f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")


"""

dataset/
 ├── images/
 │    ├── train/
 │    └── val/
 └── labels/
      ├── train/
      └── val/

dataset.yaml
train: /abs/path/to/dataset/images/train
val: /abs/path/to/dataset/images/val

nc: 1
names: ['interferogram']


"""

import subprocess

subprocess.run(["git", "clone", "https://github.com"])
subprocess.run(["/ultralytics/yolov5.git"])
subprocess.run(["cd", "yolov5"])
subprocess.run(["pip", "install", "-r", "requirements.txt"])

"""
git clone https://github.com
/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
"""

# Trénink (příklad příkazu, upravte podle potřeby)
#python train.py --img 640 --batch 16 --epochs 50 --data /path/to/dataset.yaml --weights yolov5s.pt
subprocess.run(["python", "train.py", "--img", "640", "--batch", "16", "--epochs", "50", "--data", "/path/to/dataset.yaml", "--weights", "yolov5s.pt"], cwd="yolov5")


import torch
import cv2
import os

# Načti model (lokální weights nebo z runs/train/exp/weights/best.pt)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt', force_reload=False)

img_path = "dataset/images/val/img101.jpg"
img = cv2.imread(img_path)
results = model(img)             # inference
results.print()                  # stdout: detected classes, conf

# DataFrame s boxy (xmin,ymin,xmax,ymax,conf,class)
df = results.pandas().xyxy[0]
print(df)

# Uložení ROI
out_dir = "detected_rois"
os.makedirs(out_dir, exist_ok=True)
for i, row in df.iterrows():
    x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        continue
    cv2.imwrite(os.path.join(out_dir, f"{os.path.basename(img_path)}_roi_{i}.png"), roi)
