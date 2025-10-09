from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld

if __name__ == "__main__":
    # 使用自己的YOLOv11.yamy文件搭建模型并加载预训练权重训练模型
    model = YOLO(r"F:\NCR\bone_metastasis\Yolov11\ultralytics-main\ultralytics-main\ultralytics\cfg\models\11\yolo11_PPA.yaml").load(r'F:\NCR\bone_metastasis\Yolov11\ultralytics-main\ultralytics-main\yolo11n.pt')  # build from YAML and transfer weights
    results = model.train(
        data=r'F:\NCR\bone_metastasis\Yolov11\ultralytics-main\ultralytics-main\datasets\bone_cancer\bone_cancer.yaml',
        cache=True,
        imgsz=640,
        epochs=200,
        single_cls=True,
        batch=4,
        close_mosaic=50,
        workers=0,
        device='0',
        amp=True,
        project='runs/train',
        name='exp_bone_nPPA',
    )

