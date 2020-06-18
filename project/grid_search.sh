#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate team6

max_epochs=10
gpus=4
batch_size=16

for gamma in 20.0 50.0 200.0
do
    ( echo "Training G2G with --gamma=$gamma" && \
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --gamma=$gamma --gpus=$gpus --max_epochs=$max_epochs --batch_size=$batch_size 2>&1 ) | tee logs_gam50/gamma${gamma}.log
done