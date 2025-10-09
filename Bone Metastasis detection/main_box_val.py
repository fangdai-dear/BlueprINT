from ultralytics import YOLO

#val
model = YOLO("path/to/best.pt")  # load a custom model
# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered

