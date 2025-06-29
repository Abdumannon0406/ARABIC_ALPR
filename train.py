from ultralytics import YOLO
import torch
print(torch.cuda.is_available())
model=YOLO("yolo11n.pt")

results=model.train(data="dataset/data.yaml")

