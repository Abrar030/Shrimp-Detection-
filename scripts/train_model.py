from ultralytics import YOLO

def main():
    
    model = YOLO("yolov8n.pt")  

   
    model.train(
        data="data/shrimp.yaml",
        epochs=50,
        imgsz=416,         
        batch=4,            
        workers=2,          
        device=0,           
        name="shrimp_segmentation_lowmem",
        project="outputs/trained_model"
)


if __name__ == "__main__":
    main()
