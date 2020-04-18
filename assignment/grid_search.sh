#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate team6

# grid search over 
# --lr: Learning rate for optimizer
# --batch_size: Batch size during training
# --nf: Size of feature maps in encoder & decoder

for lr in 0.0001 0.0002
do
    for batch_size in 4 8 16 32 64
    do  
        # number of feature maps in encoder & decoder
        for nf in 4 8 16 32 64
        do  
            echo "Training Autoencoder with --lr=$lr --batch_size=$batch_size --nfe=$nf --nfd=$nf"
            python lit_autoencoder.py --lr=$lr --batch_size=$batch_size --nfe=$nf --nfd=$nf
        done
    done
done