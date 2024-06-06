from ultralytics import YOLO

model = YOLO('YOLOv8n.pt')

results = model(source="Recordings\CAPTURE1108.MP4", show=True, conf=0.3, save=True, project='./Recordings', classes=0)