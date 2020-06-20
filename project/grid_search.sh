#!/bin/bash

eval "$(conda shell.bash hook)"

max_epochs=10
gpus=4
batch_size=8
lr_gen=0.0001
lr_dis=0.0001

for delta in 5.0 20.0
do
    ( echo "Training G2G with --delta=$delta" && \
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --lr_gen=$lr_gen --lr_dis=$lr_dis --log_name=grid_delta_$delta --delta=$delta --gpus=$gpus --max_epochs=$max_epochs --batch_size=$batch_size 2>&1 ) | tee logs/grid_logs/delta_${delta}.log 
done