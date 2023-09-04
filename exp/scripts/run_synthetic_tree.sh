#!/bin/bash

mkdir -p synthetic_raw_results/tree-arity-2-gin
mkdir -p synthetic_raw_results/tree-arity-2-sage
mkdir -p synthetic_raw_results/tree-arity-2-gcn
mkdir -p synthetic_raw_results/tree-arity-2-gat

for i in {2..30..2} 
    do
    for j in {1..3}
    do
        L=$((i/2))
        # GIN TREE
        if [ ! -f synthetic_raw_results/tree-arity-2-gin/size-$i-seed-$j ]
        then 
            python exp/run.py --dataset TREE --bs 128 --epochs 100 --hidden_dim 64 --model gin --arity 2 --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/tree-arity-2-gin/size-$i-seed-$j
        fi

        # SAGE TREE
        if [ ! -f synthetic_raw_results/tree-arity-2-sage/size-$i-seed-$j ]
        then 
            python exp/run.py --dataset TREE --bs 128 --epochs 100 --hidden_dim 64 --model sage --arity 2 --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/tree-arity-2-sage/size-$i-seed-$j
        fi

        # GCN TREE
        if [ ! -f synthetic_raw_results/tree-arity-2-gcn/size-$i-seed-$j ]
        then 
            python exp/run.py --dataset TREE --bs 128 --epochs 100 --hidden_dim 64 --model gcn --arity 2 --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/tree-arity-2-gcn/size-$i-seed-$j
        fi

        # GAT TREE
        if [ ! -f synthetic_raw_results/tree-arity-2-gat/size-$i-seed-$j ]
        then 
            python exp/run.py --dataset TREE --bs 128 --epochs 100 --hidden_dim 64 --model gat --arity 2 --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/tree-arity-2-gat/size-$i-seed-$j
        fi
    done
done