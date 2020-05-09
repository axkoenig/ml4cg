#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate team6

max_epochs=40
gpus=2
batch_size=32

# for batch_size in 32 64
# do
    for gamma in 0.3 0.4 0.5
    do
        ( echo "Training G2G with --batch_size=$batch_size --gamma=$gamma" && \
        CUDA_VISIBLE_DEVICES=1,2 python train.py --gpus=$gpus --batch_size=$batch_size --gamma=$gamma --max_epochs=$max_epochs 2>&1 ) | tee new_grid_logs/05_07_gamma${gamma}.log
    done
# done
