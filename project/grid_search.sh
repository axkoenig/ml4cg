#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate team6

max_epochs=20
gpus=4
batch_size=32
lr=0.0002
alpha=1.0 

for gamma in 1.3 2 10 100
do
    ( echo "Training G2G with --gamma=$gamma" && \
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --gamma=$gamma --gpus=$gpus --max_epochs=$max_epochs 2>&1 ) | tee funit_grid_logs/gamma${gamma}.log
done