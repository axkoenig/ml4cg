#!/bin/bash

eval "$(conda shell.bash hook)"

max_epochs=6
gpus=4
batch_size=4
lr_gen=0.0001
lr_dis=0.0001
num_plot_triplets=2

for gamma in 1000 100
do
    for delta_max in 1.0 5.0
    do
        ( echo "Training G2G with --delta_max=$delta_max --gamma=$gamma" && \
        CUDA_VISIBLE_DEVICES=0,1,3,4 python train.py --num_plot_triplets=$num_plot_triplets --lr_gen=$lr_gen --lr_dis=$lr_dis --log_name=final_maxdelta${delta_max}_gamma${gamma} --gamma=$gamma --delta_max=$delta_max --gpus=$gpus --max_epochs=$max_epochs --batch_size=$batch_size 2>&1 ) | tee logs/grid_logs/final_maxdelta${delta_max}_gamma${gamma}.log 
    done
done