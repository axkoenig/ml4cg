#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate team6

max_epochs=40
gpus=2
batch_size=32

for gamma in 0.4
do
    ( echo "Training G2G with --batch_size=$batch_size --gamma=$gamma" && \
    CUDA_VISIBLE_DEVICES=5,6 python train_v2.py --gpus=$gpus --batch_size=$batch_size --gamma=$gamma --max_epochs=$max_epochs 2>&1 ) | tee new_grid_logs/05_09_v2_gamma${gamma}.log
done