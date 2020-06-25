#!/bin/bash

eval "$(conda shell.bash hook)"

max_epochs=10
gpus=5
batch_size=2
lr_gen=0.0001
lr_dis=0.0001
num_workers=1

for delta in 1.0
do
    ( echo "Training G2G with --delta=$delta" && \
    CUDA_VISIBLE_DEVICES=0,2,4,5,7 python train.py --num_workers=$num_workers --lr_gen=$lr_gen --lr_dis=$lr_dis --log_name=mser_resnorm_delta_$delta --delta=$delta --gpus=$gpus --max_epochs=$max_epochs --batch_size=$batch_size 2>&1 ) | tee logs/grid_logs/delta_${delta}_resnorm.log 
done