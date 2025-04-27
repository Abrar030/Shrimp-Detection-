import json


with open("data/coco/annotations.json") as f:
    data = json.load(f)


empty_images = []
for img in data["images"]:
    img_id = img["id"]
    anns = [a for a in data["annotations"] if a["image_id"] == img_id]
    if len(anns) == 0:
        empty_images.append(img["file_name"])

if empty_images:
    print(f"Found {len(empty_images)} images without annotations:")
    print("\n".join(empty_images))
else:
    print("All images have annotations!")