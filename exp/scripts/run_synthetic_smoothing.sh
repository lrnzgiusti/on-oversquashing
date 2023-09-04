#!/bin/bash

mkdir -p synthetic_smoothing/lollipop-gcn
mkdir -p synthetic_smoothing/ring-gcn
mkdir -p synthetic_smoothing/crossed-ring-gcn

for i in {1..10} 
do
    for j in {1..3}
    do
        # SMOOTHING RING
        L=$((i/2))
        if [ ! -f synthetic_smoothing/ring-gcn/extra-layers-$i-seed-$j ]
        then 
            python exp/run.py --dataset RING --bs 128 --epochs 100 --hidden_dim 64 --model gcn --mpnn_layers $(( 30 + i )) --synthetic_size 30 --add_crosses 0 --seed $j > synthetic_smoothing/ring-gcn/extra-layers-$i-seed-$j
        fi

        # SMOOTHING CROSSED RING
        if [ ! -f synthetic_smoothing/crossed-ring-gcn/extra-layers-$i-seed-$j ]
        then 
            python exp/run.py --dataset RING --bs 128 --epochs 100 --hidden_dim 64 --model gcn --mpnn_layers $(( 30 + i )) --synthetic_size 30 --add_crosses 1 --seed $j > synthetic_smoothing/crossed-ring-gcn/extra-layers-$i-seed-$j
        fi

        # SMOOTHING LOLLIPOP
        L=$((i/2 + 1))

        # synthetic size is 28 because that gives a distance of 15 (14 for path + 1 for clique) between source and target
        if [ ! -f synthetic_smoothing/lollipop-gcn/extra-layers-$i-seed-$j ]
        then 
            python exp/run.py --dataset LOLLIPOP --bs 128 --epochs 100 --hidden_dim 64 --model gcn --mpnn_layers $(( 28 + i )) --synthetic_size 28 --seed $j > synthetic_smoothing/lollipop-gcn/extra-layers-$i-seed-$j
        fi
    done
done