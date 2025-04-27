import json
import os
import numpy as np
import cv2
from pycocotools.coco import COCO

def merge_polygons(labelme_dir, output_coco_path):
    
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "shrimp"}]
    }

    annotation_id = 1


    for json_file in os.listdir(labelme_dir):
        if not json_file.endswith(".json"):
            continue

        with open(os.path.join(labelme_dir, json_file)) as f:
            data = json.load(f)


        shrimp_instances = {}
        for shape in data["shapes"]:
            shrimp_id = shape["group_id"]  
            if shrimp_id not in shrimp_instances:
                shrimp_instances[shrimp_id] = []
            shrimp_instances[shrimp_id].append(shape)


        for shrimp_id, parts in shrimp_instances.items():
            merged_mask = np.zeros((data["imageHeight"],     data["imageWidth"]), dtype=np.uint8)
            for part in parts:
                polygon = np.array(part["points"], dtype=np.int32)
                cv2.fillPoly(merged_mask, [polygon], 1)


            contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if contour.size < 6:  
                    continue
                segmentation = contour.flatten().tolist()


                coco_dataset["annotations"].append({
                    "id": annotation_id,
                    "image_id": len(coco_dataset["images"]) + 1,
                    "category_id": 1,
                    "segmentation": [segmentation],
                    "area": cv2.contourArea(contour),
                    "bbox": list(cv2.boundingRect(contour)),
                    "iscrowd": 0
                })
                annotation_id += 1


        coco_dataset["images"].append({
            "id": len(coco_dataset["images"]) + 1,
            "file_name": data["imagePath"],
            "width": data["imageWidth"],
            "height": data["imageHeight"]
        })


    with open(output_coco_path, "w") as f:
        json.dump(coco_dataset, f)

if __name__ == "__main__":
    merge_polygons(
        labelme_dir="data/annotations",
        output_coco_path="data/coco/annotations.json"
    )

