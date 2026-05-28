
import numpy as np
import os
import cv2


"""
Create ROI of images and their corresponding masks from labels in yolo format (txt files with bounding box coordinates).
The names of images, masks and labels must be the same, except for the extension and folder

Parameters:
   labels_dir: directory with txt files with bounding box coordinates in yolo format
   images_dir: directory with images corresponding to the labels
   masks_dir: directory with masks corresponding to the labels
   output_dir: directory where the ROI images and masks will be saved

"""


def create_roi_from_labels(labels_dir, images_dir, masks_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for label_file in os.listdir(labels_dir):
        if label_file.endswith(".txt"):
            base_name = os.path.splitext(label_file)[0]
            image_path = os.path.join(images_dir, base_name + ".bmp")  # Assuming images are in BMP format
            mask_path = os.path.join(masks_dir, base_name + ".bmp")  # Assuming masks have the same name and are in BMP format

            if not os.path.exists(image_path) or not os.path.exists(mask_path):
                print(f"Warning: Image or mask for {base_name} not found. Skipping.")
                continue

            # Load image and mask
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Read bounding box coordinates from label file
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"Warning: Invalid label format in {label_file}. Skipping line.")
                        continue
                    
                    class_id, x_center, y_center, width, height = map(float, parts)
                    
                    # Convert from YOLO format to pixel coordinates
                    img_height, img_width = image.shape[:2]
                    x_center *= img_width
                    y_center *= img_height
                    width *= img_width
                    height *= img_height
                    
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

                    # Ensure coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_width - 1, x2)
                    y2 = min(img_height - 1, y2)

                    # Crop the ROI from the image and mask
                    roi_image = image[y1:y2, x1:x2]
                    roi_mask = mask[y1:y2, x1:x2]

                    # Save the ROI images and masks
                    roi_image_path = os.path.join(output_dir, "images", f"{base_name}_roi.bmp")
                    roi_mask_path = os.path.join(output_dir, "masks", f"{base_name}_roi.bmp")

                    os.makedirs(os.path.dirname(roi_image_path), exist_ok=True)
                    os.makedirs(os.path.dirname(roi_mask_path), exist_ok=True)

                    cv2.imwrite(roi_image_path, roi_image)
                    cv2.imwrite(roi_mask_path, roi_mask)

if __name__ == "__main__":
    labels_dir = "data/preprocessing/bounding_box/unet_dataset_4/yolo_labels/test/"
    images_dir = "data/preprocessing/mask/unet_dataset_4/test/images/"
    masks_dir = "data/preprocessing/mask/unet_dataset_4/test/masks/"
    output_dir = "data/preprocessing/mask/roi_dataset_4/test/"

    create_roi_from_labels(labels_dir, images_dir, masks_dir, output_dir)