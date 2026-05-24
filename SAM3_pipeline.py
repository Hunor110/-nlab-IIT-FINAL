import os
import cv2 as cv
from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics import YOLO
import numpy as np


# ----------------------
# Paths
# ----------------------
IMAGE_DIR = r"C:\Users\Redux\Desktop\code\rgb02"
MASK_DIR = r"C:\Users\Redux\Desktop\code\masks"

os.makedirs(MASK_DIR, exist_ok=True)

# ----------------------
# SAM3 setup
# ----------------------
sam_model_path = r"C:\Users\Redux\Desktop\code\sam3.pt"
sam_overrides = dict(
    conf=0.5,
    task="segment",
    mode="predict",
    model=sam_model_path,
    half=True,  
    save=False, 
    device=0     # use CUDA
)
sam_predictor = SAM3SemanticPredictor(overrides=sam_overrides)

CLASSES = ["stone pavement","grass", "gravel", "dirt"]

# ----------------------
# Processing loop
# ----------------------
for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    print(f"Processing {img_name} ...")

    # --- SAM3 segmentation ---
    sam_predictor.set_image(img_path)
    seg_results = sam_predictor(text=CLASSES)

    #cv.plot(seg_results)

    for r in seg_results:

        masks = r.masks.data.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for i, mask in enumerate(masks):

            class_id = classes[i]
            class_name = CLASSES[class_id]

            mask_img = (mask * 255).astype(np.uint8)

            mask_path = os.path.join(
                MASK_DIR,
                f"{os.path.splitext(img_name)[0]}_{class_name}_{i}.png"
            )

            cv.imwrite(mask_path, mask_img)

print("Finito")