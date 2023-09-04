#!/bin/bash

mkdir -p synthetic_raw_results/crossed-ring-gin
mkdir -p synthetic_raw_results/crossed-ring-sage
mkdir -p synthetic_raw_results/crossed-ring-gcn
mkdir -p synthetic_raw_results/crossed-ring-gat

for i in {2..30..2} 
    do
    for j in {1..3}
    do
        L=$((i/2))
        # GIN CROSSED RING
        if [ ! -f synthetic_raw_results/crossed-ring-gin/size-$i-seed-$j ]
        then 
            python exp/run.py --dataset RING --bs 128 --epochs 100 --hidden_dim 64 --model gin --mpnn_layers $L --synthetic_size $i --add_crosses 1 --seed $j > synthetic_raw_results/crossed-ring-gin/size-$i-seed-$j
        fi

        # SAGE CROSSED RING
        if [ ! -f synthetic_raw_results/crossed-ring-sage/size-$i-seed-$j ]
        then 
            python exp/run.py --dataset RING --bs 128 --epochs 100 --hidden_dim 64 --model sage --mpnn_layers $L --synthetic_size $i --add_crosses 1 --seed $j > synthetic_raw_results/crossed-ring-sage/size-$i-seed-$j
        fi

        # GCN CROSSED RING
        if [ ! -f synthetic_raw_results/crossed-ring-gcn/size-$i-seed-$j ]
        then 
            python exp/run.py --dataset RING --bs 128 --epochs 100 --hidden_dim 64 --model gcn --mpnn_layers $L --synthetic_size $i --add_crosses 1 --seed $j > synthetic_raw_results/crossed-ring-gcn/size-$i-seed-$j
        fi

        # GAT CROSSED RING
        if [ ! -f synthetic_raw_results/crossed-ring-gat/size-$i-seed-$j ]
        then 
            python exp/run.py --dataset RING --bs 128 --epochs 100 --hidden_dim 64 --model gat --mpnn_layers $L --synthetic_size $i --add_crosses 1 --seed $j > synthetic_raw_results/crossed-ring-gat/size-$i-seed-$j
        fi
    done
done