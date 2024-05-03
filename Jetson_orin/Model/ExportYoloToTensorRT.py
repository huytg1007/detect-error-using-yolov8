from ultralytics import YOLO
import locale

locale.getpreferredencoding = lambda: "UTF-8"

# Load the YOLOv8 model
model = YOLO("/home/orin/Documents/Ultralytic/best_bottleneck_1.pt")

# Export the model to TensorRT format
model.export(format="engine", half=True, imgsz=(640,640))  # creates 'yolov8n.engine'
