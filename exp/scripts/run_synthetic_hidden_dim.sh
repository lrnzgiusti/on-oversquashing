#!/bin/bash


for d in 1 2 4 8 16 32
    do
    mkdir -p synthetic_raw_results/ring-gcn-hidden-dim-$d
    mkdir -p synthetic_raw_results/crossed-ring-gcn-hidden-dim-$d
    mkdir -p synthetic_raw_results/lollipop-gcn-hidden-dim-$d
    for i in {2..30..2} 
    do
        for j in {1..3}
        do
            # GCN RING
            L=$((i/2))
            if [ ! -f synthetic_raw_results/ring-gcn-hidden-dim-$d/size-$i-seed-$j ]
            then 
                python exp/run.py --dataset RING --bs 128 --epochs 100 --hidden_dim $d --model gcn --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/ring-gcn-hidden-dim-$d/size-$i-seed-$j
            fi

            # GCN CROSSED RING
            if [ ! -f synthetic_raw_results/crossed-ring-gcn-hidden-dim-$d/size-$i-seed-$j ]
            then 
                python exp/run.py --dataset RING --bs 128 --epochs 100 --hidden_dim $d --model gcn --mpnn_layers $L --synthetic_size $i --add_crosses 1 --seed $j > synthetic_raw_results/crossed-ring-gcn-hidden-dim-$d/size-$i-seed-$j
            fi

            # GCN LOLLIPOP
            L=$((i/2 + 1))
            if [ ! -f synthetic_raw_results/lollipop-gcn-hidden-dim-$d/size-$i-seed-$j ]
            then 
                python exp/run.py --dataset LOLLIPOP --bs 128 --epochs 100 --hidden_dim $d --model gcn --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/lollipop-gcn-hidden-dim-$d/size-$i-seed-$j
            fi
        done
    done
done