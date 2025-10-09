set -ex
python train.py --dataroot ./datasets/训练 --name 骨转移_pix2pix --model pix2pix --netG unet_512 --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0
