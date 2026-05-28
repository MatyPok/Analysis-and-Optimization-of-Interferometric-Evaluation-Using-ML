
#import subprocess
#import sys

#import sys, os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


# instalace PyTorch (příklad pro CUDA 11.8)
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

#subprocess.run(["pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])

# instalace Detectron2 (přímý wheel od autora)
#pip install 'git+https://github.com/facebookresearch/detectron2.git'

#subprocess.run(["pip", "install", "git+https://github.com/facebookresearch/detectron2.git"])
#subprocess.run([sys.executable, "-m", "pip", "install", "git+https://github.com/facebookresearch/detectron2.git"])

"""
dataset/
 ├── train/
 │    ├── img001.jpg
 │    ├── img002.jpg
 │    └── ...
 ├── val/
 │    ├── img101.jpg
 │    └── ...
 ├── annotations/
 │    ├── instances_train.json
 │    └── instances_val.json

"""

import os

# Training
from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.engine import DefaultTrainer
from detectron2.engine import hooks
from detectron2.evaluation import COCOEvaluator


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def build_hooks(self):
        hooks_list = super().build_hooks()

        # Přidání BestCheckpointeru
        hooks_list.insert(-1, hooks.BestCheckpointer(
            self.cfg.TEST.EVAL_PERIOD,
            self.checkpointer,
            "bbox/AP",   # sledovaná metrika
            mode="max"
        ))

        return hooks_list

register_coco_instances("interferogram_train", {}, "data/preprocessing/bounding_box/unet_dataset_4/annotations/instances_train.json", "data/preprocessing/bounding_box/unet_dataset_4/train")
register_coco_instances("interferogram_val", {}, "data/preprocessing/bounding_box/unet_dataset_4/annotations/instances_val.json", "data/preprocessing/bounding_box/unet_dataset_4/val")

metadata = MetadataCatalog.get("interferogram_train")
dataset_dicts = DatasetCatalog.get("interferogram_train")

#################################


cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu" # nebo "cuda" pokud máš GPU
cfg.TEST.EVAL_PERIOD = 100  # každých x iterací
cfg.OUTPUT_DIR = "src/preprocessing/bounding_box/results/output_2"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")) # nebo COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml
cfg.DATASETS.TRAIN = ("interferogram_train",)
cfg.DATASETS.TEST = ("interferogram_val",)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")  # transfer learning
cfg.SOLVER.IMS_PER_BATCH = 1 # taky na 1 pokud to znovu padne
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000   # zvětši dle datasetu
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # jen "interferogram"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

################################

# Inference 
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

# Get list of all paths to test images
from pathlib import Path
import sys
test_image_path = 'data/preprocessing/bounding_box/dataset/test'
jpeg_file_list = [path for path in Path(test_image_path).rglob('*.jpeg')]
jpg_file_list = [path for path in Path(test_image_path).rglob('*.jpg    ')]
png_file_list = [path for path in Path(test_image_path).rglob('*.png')]
bmp_file_list = [path for path in Path(test_image_path).rglob('*.bmp')]

if sys.platform == 'linux':
    JPEG_file_list = [path for path in Path(test_image_path).rglob('*.JPEG')]
    JPG_file_list = [path for path in Path(test_image_path).rglob('*.JPG')]
    test_file_list = jpg_file_list + JPG_file_list + png_file_list + bmp_file_list + JPEG_file_list + jpeg_file_list
else:
    test_file_list = jpg_file_list + png_file_list + bmp_file_list + jpeg_file_list
print('Total test images: %d' % len(test_file_list))

# Příklad predikce na jednom obrázku
test_image = test_file_list[0]
img = cv2.imread(str(test_image))
outputs = predictor(img)

# outputs obsahuje predikované boxy a skóre
print(outputs["instances"].pred_boxes)
print(outputs["instances"].scores)

# vizualizace
v = Visualizer(img[:, :, ::-1], metadata=metadata)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("result", out.get_image()[:, :, ::-1])
cv2.waitKey(0)


##################

# Uložení ROI
boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
for i, (x1, y1, x2, y2) in enumerate(boxes):
    roi = img[int(y1):int(y2), int(x1):int(x2)]
    cv2.imwrite(f"roi_{i}.png", roi)


