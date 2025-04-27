import csv
import os
from ultralytics import YOLO

def run_inference():
   
    os.makedirs("outputs/visuals", exist_ok=True)
  
    model = YOLO("outputs/trained_model/shrimp_segmentation_lowmem3/weights/best.pt")

    results = []
    test_images = os.listdir("data/raw_images") 

    
    for img_name in test_images:
        img_path = os.path.join("data/raw_images", img_name)
        
     
        prediction = model.predict(img_path, save=True, conf=0.5, save_dir="outputs/visuals")
        
    
        shrimp_count = len(prediction[0].boxes)
        results.append({"image_name": img_name, "shrimp_count": shrimp_count})
        
        print(f"Processed {img_name}: {shrimp_count} shrimp")

w
    with open("outputs/results.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "shrimp_count"])
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    run_inference()