#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate team6

max_epochs=20
gpus=2
batch_size=16

for lambda_g in 0.01 1 10
do
    for lambda_d in 0.01 1 10
    do
        ( echo "Training G2G with --batch_size=$batch_size --lambda_g=$lambda_g --lambda_d=$lambda_d" && \
        CUDA_VISIBLE_DEVICES=5,6 python train.py --gpus=$gpus --batch_size=$batch_size --lambda_g=$lambda_g --lambda_d=$lambda_d --max_epochs=$max_epochs 2>&1 ) | tee grid_logs/bs${batch_size}_lambda_g${lambda_g}_lambda_d${lambda_d}.log
    done
done
