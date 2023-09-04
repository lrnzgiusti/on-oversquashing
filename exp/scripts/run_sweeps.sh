#!/bin/bash

#friendly usage:
# $1  is the entity
# $2  is the name of the project
# $3  is the sweep id

for i in {0..3}  # GPU in use
do
    for _ in {0..1}  # agent (per GPU)
    do
        CUDA_VISIBLE_DEVICES=$((i % 4)) wandb agent "$1"/"$2"/"$3" & 
    done
done
