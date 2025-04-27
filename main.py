import json

file_path = 'data/coco/annotations.json'

with open(file_path,'r') as f:
    coco_data = json.load(f)

#print(coco_data.keys())
 
print(coco_data['image'])