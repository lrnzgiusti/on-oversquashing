#!/bin/bash

mkdir -p synthetic_raw_results/lollipop-gin
mkdir -p synthetic_raw_results/lollipop-sage
mkdir -p synthetic_raw_results/lollipop-gcn
mkdir -p synthetic_raw_results/lollipop-gat

for i in {2..30..2} 
    do
    for j in {1..3}
    do
        L=$((i/2 + 1))
        # GIN lollipop
        if [ ! -f synthetic_raw_results/lollipop-gin/size-$i-seed-$j ]
        then 
            python exp/run.py --dataset LOLLIPOP --bs 128 --epochs 100 --hidden_dim 64 --model gin --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/lollipop-gin/size-$i-seed-$j
        fi

        # SAGE lollipop
        if [ ! -f synthetic_raw_results/lollipop-sage/size-$i-seed-$j ]
        then 
            python exp/run.py --dataset LOLLIPOP --bs 128 --epochs 100 --hidden_dim 64 --model sage --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/lollipop-sage/size-$i-seed-$j
        fi

        # GCN lollipop
        if [ ! -f synthetic_raw_results/lollipop-gcn/size-$i-seed-$j ]
        then 
            python exp/run.py --dataset LOLLIPOP --bs 128 --epochs 100 --hidden_dim 64 --model gcn --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/lollipop-gcn/size-$i-seed-$j
        fi

        # GAT lollipop
        if [ ! -f synthetic_raw_results/lollipop-gat/size-$i-seed-$j ]
        then 
            python exp/run.py --dataset LOLLIPOP --bs 128 --epochs 100 --hidden_dim 64 --model gat --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/lollipop-gat/size-$i-seed-$j
        fi
    done
done