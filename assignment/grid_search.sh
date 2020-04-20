#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate team6

# grid search over 
# --batch_size: Batch size during training
# --nf: Size of feature maps in encoder & decoder

for batch_size in 64 128 256
do  
    for nf in 64 128 256
    do  
        echo "Training Autoencoder with --batch_size=$batch_size --nfe=$nf --nfd=$nf"
        python autoencoder.py --batch_size=$batch_size --nfe=$nf --nfd=$nf
    done
done