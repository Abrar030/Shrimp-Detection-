import json
import os
from pathlib import Path


COCO_JSON_PATH = "data/coco/annotations.json"  
OUTPUT_LABELS_DIR = "data/labels"             


Path(OUTPUT_LABELS_DIR).mkdir(parents=True, exist_ok=True)


with open(COCO_JSON_PATH) as f:
    coco_data = json.load(f)

for image_info in coco_data["images"]:
    image_id = image_info["id"]
    image_name = image_info["file_name"].split(".")[0]  
    image_width = image_info["width"]
    image_height = image_info["height"]

   
    annotations = [a for a in coco_data["annotations"] if a["image_id"] == image_id]

    yolo_lines = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        x_center = (x + w/2) / image_width
        y_center = (y + h/2) / image_height
        w_norm = w / image_width
        h_norm = h / image_height
        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

   
    txt_path = os.path.join(OUTPUT_LABELS_DIR, f"{image_name}.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(yolo_lines))

print(f"Generated {len(coco_data['images'])} YOLO labels in {OUTPUT_LABELS_DIR}")