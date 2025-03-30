import os
from ultralytics import YOLO
import torch



num_gpus = torch.cuda.device_count()

devices = [i for i in range(num_gpus)] 

script_dir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(script_dir, 'data', 'data.yaml') 

model = YOLO("yolov8n.pt")

model.train(data=data_path, epochs=50, imgsz=640, batch=8, name="yolov8n_ship", device=devices)

model.export(format="onnx", path=script_dir)


