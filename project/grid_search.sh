#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate team6

max_epochs=20
gpus=2

for batch_size in 16 32 64
do
    for gamma in 0.01 0.4 1 1.5 2
    do
        ( echo "Training G2G with --batch_size=$batch_size --gamma=$gamma" && \
        CUDA_VISIBLE_DEVICES=0,1 python train.py --gpus=$gpus --batch_size=$batch_size --gamma=$gamma --max_epochs=$max_epochs 2>&1 ) | tee grid_logs/bs${batch_size}_gamma${gamma}.log
    done
done