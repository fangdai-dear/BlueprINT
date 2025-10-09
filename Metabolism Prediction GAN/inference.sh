# !/bin/bash

# ================================
# Pix2Pix Model Inference Script
# Author: [Your Name]
# Date: $(date +%Y-%m-%d)
# Description: This script executes a Pix2Pix model on the specified dataset.
# ================================

# 拼接两个图片
# echo "Inference completed. Results are saved in $RESULTS_DIR"

# echo "Inference completed. Results are saved in $RESULTS_DIR"
# python datasets/make_dataset_aligned.py --dataset-path  /export/home/daifang/ncr/pix2pix/1000001854_1_png

# GAN生成图片
# DATAROOT="/export/home/daifang/ncr/pix2pix/1000001854_1_png"
# MODEL="pix2pix"
# NUM_TEST=536
# RESULTS_DIR="/export/home/daifang/ncr/pix2pix/1000001854_1_png/result"
# PHASE="test"
# CROP_SIZE=512
# EPOCH="latest"
# NAME="骨转移_pix2pix_512"
# DISPLAY_WINSIZE=512
# LOAD_SIZE=512
# DIRECTION="AtoB"
# DATASET_MODE="aligned"
# NETG="resnet_9blocks"

# # Execute the model
# echo "Starting Pix2Pix Inference..."
# echo "Model: $MODEL | Epoch: $EPOCH | Dataset: $DATAROOT"

# python test.py \
#     --dataroot "$DATAROOT" \
#     --model "$MODEL" \
#     --num_test "$NUM_TEST" \
#     --results_dir "$RESULTS_DIR" \
#     --phase "$PHASE" \
#     --crop_size "$CROP_SIZE" \
#     --epoch "$EPOCH" \
#     --name "$NAME" \
#     --display_winsize "$DISPLAY_WINSIZE" \
#     --load_size "$LOAD_SIZE" \
#     --direction "$DIRECTION" \
#     --dataset_mode "$DATASET_MODE" \
#     --netG "$NETG"

# ## 结果建新文件夹
# ## 设置路径参数
# SOURCE_FOLDER="/export/home/daifang/ncr/pix2pix/1000001854_1_png/result"
# FAKE_B_FOLDER="/export/home/daifang/ncr/pix2pix/1000001854_1_png/result"
# REAL_A_FOLDER="/export/home/daifang/ncr/pix2pix/医生挑选数据/4class/Occult_deal/real_A"
# REAL_B_FOLDER="/export/home/daifang/ncr/pix2pix/医生挑选数据/4class/Occult_deal/real_B"

# # 执行 Python 脚本
# python3 2.py \
#     --source-folder "$SOURCE_FOLDER" \
#     --fake-B-folder "$FAKE_B_FOLDER" \
#     --real-A-folder "$REAL_A_FOLDER" \
#     --real-B-folder "$REAL_B_FOLDER"

# echo "文件整理完成，输出路径如下："
# echo "fake_B: $FAKE_B_FOLDER"
# echo "real_A: $REAL_A_FOLDER"
# echo "real_B: $REAL_B_FOLDER"



#!/bin/bash
# PET叠加在CT上面
# ================================
# Image Processing Script
# Author: [Your Name]
# Date: $(date +%Y-%m-%d)
# Description: This script processes images using the Python script for heatmap generation and overlay.
# ================================

# 设置路径参数
CT_DIR="/export/home/daifang/ncr/pix2pix/1000001854_1_png/result/骨转移_pix2pix_512/test_latest/images/real_A"
REAL_PET_DIR="/export/home/daifang/ncr/pix2pix/1000001854_1_png/PET"
FAKE_PET_DIR="/export/home/daifang/ncr/pix2pix/1000001854_1_png/result/骨转移_pix2pix_512/test_latest/images/fake_B"
SAVE_REAL_DIR="/export/home/daifang/ncr/pix2pix/1000001854_1_png/result/output_real"
SAVE_FAKE_DIR="/export/home/daifang/ncr/pix2pix/1000001854_1_png/result/output_fake"

# 设置其他参数
OPACITY_REAL=0.5
OPACITY_FAKE=0.5
CONTRAST=1.0
BRIGHTNESS=1.0

# 执行 Python 脚本
python3 alignment_gen_data_bone_win.py \
    --ct-dir "$CT_DIR" \
    --real-pet-dir "$REAL_PET_DIR" \
    --fake-pet-dir "$FAKE_PET_DIR" \
    --save-real-dir "$SAVE_REAL_DIR" \
    --save-fake-dir "$SAVE_FAKE_DIR" \
    --opacity-real "$OPACITY_REAL" \
    --opacity-fake "$OPACITY_FAKE" \
    --compare "$CONTRAST" \
    --lighting "$BRIGHTNESS"

echo "图像处理完成，结果已保存至："
echo "Real PET + CT: $SAVE_REAL_DIR"
echo "Fake PET + CT: $SAVE_FAKE_DIR"
