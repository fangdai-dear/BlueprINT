<div align="center">

# BlueprINT Detection Module

**A YOLO11-based Object Detection Framework for Medical Imaging**

</div>



## Overview

This repository provides the **detection module** of the BlueprINT framework — a YOLO11-based deep learning system optimized for **medical lesion localization and classification**.  
It supports **3D lesion detection**, **ROI extraction**, and **radiologic subtype classification** from CT or PET/CT images, and can be easily adapted to other medical imaging tasks.

Compared with the standard Ultralytics YOLO implementation, this version retains only the **object detection task**, simplifying the model for research and clinical use.



## Model Architecture

- **Backbone:** CSP-based encoder with FeatUp feature enhancement  
- **Neck:** FPN + PAN structure for multi-scale fusion  
- **Head:** YOLO detection head with adaptive loss weighting  
- **Loss:** IoU + objectness + class-balanced BCE  
- **Supported formats:** PyTorch, ONNX, TensorRT

```
ultralytics/
└── models/
    └── yolo/
        ├── detect/
        │   ├── train.py
        │   ├── val.py
        │   ├── predict.py
        │   └── loss.py
        ├── modules/
        │   ├── backbone.py
        │   ├── head.py
        │   ├── conv.py
        │   └── block.py
        └── cfg/
            ├── yolo11s.yaml
            ├── yolo11m.yaml
            └── yolo11l.yaml
```


## Installation

Python ≥3.8 and PyTorch ≥1.8 are required.

```bash
# Clone repository
git clone https://github.com/yourname/BlueprINT-Detection.git
cd BlueprINT-Detection

# Install dependencies
pip install -r requirements.txt
```



## Training

You can train the model using the standard YOLO CLI or Python API.

### **CLI mode**
```bash
yolo detect train data=data/bone.yaml model=models/yolo11m.yaml epochs=100 imgsz=640 batch=16 device=0
```

### **Python API**
```python
from ultralytics import YOLO

model = YOLO("models/yolo11m.yaml")
model.train(data="data/bone.yaml", epochs=100, imgsz=640, device=0)
```


## Inference

```bash
yolo predict model=weights/best.pt source=images/test/ imgsz=640 conf=0.25 save=True
```

or directly in Python:
```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")
results = model("images/test/sample.png")
results[0].show()
```



## Dataset Structure

```
data/
├── bone.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each label file follows YOLO format:  
```
class x_center y_center width height
```


## Evaluation

Evaluate trained models:
```bash
yolo val model=weights/best.pt data=data/bone.yaml imgsz=640
```
Metrics include:
- **mAP@0.5–0.95**
- **Precision / Recall**
- **Lesion-level F1**
- **Inference Speed (ms/img)**


## Export

Export the trained model for deployment:
```bash
yolo export model=weights/best.pt format=onnx dynamic=True
```


## Acknowledgements

This repository is built upon:
- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [BlueprINT: AI Framework for Bone Metastasis Evaluation (2025)](https://your-publication-link.com)


<div align="center">

**Maintainer:** Fang Dai, Xiaomeng Li, Siqiong Yao  
**Institution:** Shanghai Jiao Tong University – National Center for Translational Medicine  

</div>

