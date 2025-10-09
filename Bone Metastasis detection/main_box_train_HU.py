from ultralytics import YOLO
#总的灰度均值: 0.5115629255547741, 总的均方差: 0.16052455512425842


if __name__ == "__main__":
    # 使用自己的 YOLOv11 配置文件并加载预训练权重进行训练
    model = YOLO("/export/home/daifang/ncr/Yolo11/ultralytics-main/ultralytics/cfg/models/11/yolo11_CAFM_PPA_P2.yaml").load('/export/home/daifang/ncr/Yolo11/ultralytics-main/runs/train/BONE_BOX_250117BESTTT/weights/best.pt')

    results = model.train(
        data='/export/home/daifang/ncr/Yolo11/datasets_0422/bone_cancer.yaml',  # 数据配置文件
        resume=False,
        cache = True,
        imgsz=(512, 640, 768),  # 输入图像大小
        epochs=250,
        single_cls=True,  # 如果是单类检测
        batch=150,
        workers=8, 
        device='cuda:3',  # 使用 GPU 0，确保 CUDA 配置正确
        optimizer='SGD',  
        amp=True,  # 自动混合精度训练，确保 PyTorch 支持
        project='runs/train',  # 输出结果目录
        name='0512_PPA_CAFM',  # 训练任务名称
    )
