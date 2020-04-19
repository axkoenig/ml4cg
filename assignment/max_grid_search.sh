#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate team6

# grid search over 
# --batch_size: Batch size during training
# --lr: Learning rate for optimizer

for lr in 0.0001 0.0002
do  
    for batch_size in 16 32 64
    do  
        echo "Training Autoencoder with --batch_size=$batch_size --lr=$lr"
        python max_lit_autoencoder.py --batch_size=$batch_size --lr=$lr
    done
done