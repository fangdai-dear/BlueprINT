# 🦴 BlueprINT: Bone Lesion Unified Evaluation Procedure Intelligence

**BlueprINT** is a comprehensive CT-based AI platform for multidimensional whole-body analysis of **bone metastases (BMs)**.  
Trained on **7,231 PET/CT cases** from four multicenter cohorts (~2.06M CT slices), BlueprINT performs 3D lesion detection, segmentation, classification (osteoblastic, osteolytic, mixed, occult), metabolic estimation, and report generation directly from routine CT scans.

---

## 🚀 Overview

Bone metastases (BMs) are a highly prevalent form of distant spread, affecting nearly 80% of patients with advanced solid tumors.  
While CT offers superior anatomical detail, it lacks metabolic information and requires manual interpretation.  
**BlueprINT** bridges this gap by integrating multitask deep learning and metabolic inference models for automated, explainable analysis.

---

## 🧠 Model Framework

BlueprINT integrates **six coordinated modules**:

| Module | Description |
|---------|--------------|
| 🩻 **Detection** | YOLOv11 + FeatUp + RFM-SR for multi-scale lesion detection |
| 🧬 **Segmentation** | nnUNet + FPN + PPM for accurate lesion boundary delineation |
| 🧩 **Classification** | MobileNetV2-based 4-class subtype classifier |
| 🔥 **Metabolic Estimation** | BMPET-GAN predicts SUVmean/SUVmax maps from CT |
| 🦴 **3D Reconstruction** | MedVoxel3D Builder reconstructs full skeletal structure |
| 🩺 **LLM Reporting** | GPT-powered medical report generation (HTML / JSON output) |

---

## 📂 Dataset Structure

Organize your dataset as follows before training:

```
BlueprINT/
│
├── data/
│   ├── train/
│   │   ├── images/              # CT images (.nii.gz / .png)
│   │   ├── masks/               # Segmentation masks
│   │   ├── labels.csv           # Bounding boxes, HU metrics, lesion types
│   │   └── pet/                 # (Optional) paired PET data for GAN training
│   ├── val/
│   └── test/
│
├── configs/
│   ├── detection.yaml
│   ├── segmentation.yaml
│   ├── classifier.yaml
│   └── gan.yaml
│
├── models/                      # Pretrained checkpoints
├── scripts/                     # Training & inference scripts
└── reports/                     # Generated reports (HTML / JSON)
```

---

## ⚙️ Environment Setup

**Tested Environment**

- Ubuntu 22.04 LTS  
- Python 3.10  
- CUDA 12.2  
- PyTorch 2.5.1  
- 8× NVIDIA A100 GPUs (80 GB)

**Installation**

```bash
git clone https://github.com/fangdai-dear/BlueprINT.git
cd BlueprINT
conda create -n blueprint python=3.10
conda activate blueprint
pip install -r requirements.txt
```

**Key Dependencies**
```
torch>=2.5.0
torchvision
monai
opencv-python
simpleitk
pydicom
matplotlib
scikit-learn
pyradiomics
shap
```

---

## 🧩 Training Commands

Train each module separately or sequentially:

```bash
# Detection (YOLOv11 + RFM-SR)
python train_detection.py --cfg configs/detection.yaml

# Segmentation (nnUNet + FPN)
python train_segmentation.py --cfg configs/segmentation.yaml

# Classification
python train_classifier.py --cfg configs/classifier.yaml

# PET synthesis (BMPET-GAN)
python train_gan.py --cfg configs/gan.yaml
```

For distributed (multi-GPU) training:
```bash
torchrun --nproc_per_node=8 train_detection.py --cfg configs/detection.yaml
```

---

## 💡 Inference & Application

Run the full BlueprINT pipeline on a new CT scan:

```bash
python run_blueprint.py   --input /path/to/ct_series   --output ./reports   --generate_report   --use_llm
```

**Outputs**
- `lesion_masks.nii.gz` — segmentation results  
- `3D_reconstruction.obj` — skeletal reconstruction  
- `synthetic_pet.nii.gz` — predicted metabolic map  
- `report.html` — structured radiology-style report  

---

## 📊 Evaluation Metrics

| Task | Metrics |
|------|----------|
| Detection | IoU, mAP, F1, Recall |
| Segmentation | Dice, IoU, Precision |
| Classification | Accuracy, AUC, F1 |
| PET Synthesis | PSNR, SSIM, Pearson’s r |

---

## 🙏 Acknowledgments

This repository builds upon the following open-source and foundational works:

- **YOLOv11** (Ultralytics, 2025)  
- **nnU-Net** (Isensee et al., *Nat Methods*, 2021)  
- **CycleGAN** (Zhu et al., *ICCV*, 2017)  
- **FeatUp Super-Resolution** (Li et al., 2024)  
- **MedVoxel3D Builder** (Zhang et al., 2023)  
- **SHAP** & **Grad-CAM** for model interpretability  

We thank the contributors from **Shanghai Jiao Tong University**, **Peking Union Medical College**, **East China Normal University**, and **Renji/Ruijin Hospitals** for data and clinical expertise.

---

## 📜 Citation

If you find this repository useful, please cite:

```
Li X., Dai F., Niu C., Wang X., Sun P., et al. 
BlueprINT: a PET/CT-based platform for multidimensional whole-body analysis of bone metastases. (2025)
GitHub: https://github.com/fangdai-dear/BlueprINT
```
 
