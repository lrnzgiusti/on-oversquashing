import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import smooth_plot, get_resistances
from utils.factory import NetFactory

ROOT_DIR = '.'
DATASET_NAME = 'NCI1'
NUM_VERTICES_SAMPLED = 10
ARCHS = ['gin', 'sage', 'gcn', 'gat']

def initialize_architecture(arch, layers=10, dim_h=5):
    """
    Initialize an MPNN architecture.
    
    Parameters:
        arch (str): Type of architecture to initialize ('gin', 'sage', 'gcn', 'gat')
        layers (int, optional): Number of layers. Defaults to 10.
        dim_h (int, optional): Dimension. Defaults to 5.
    
    Returns:
        torch.nn.Module: Initialized model
    """
    return NetFactory(arch=arch, num_layers=layers, dim_h=dim_h)

def process_graph_data(dataset_item, arch):
    """
    Process data for a specific graph architecture.
    
    Parameters:
        dataset_item (Data): Torch geometric dataset item.
        arch (str): Type of architecture to use.
        
    Returns:
        list: List of tuples with processed data.
    """
    model = initialize_architecture(arch)
    G = to_networkx(dataset_item, to_undirected=True)
    distances = nx.floyd_warshall_numpy(G)

    pairs_runaway = []

    for _ in range(NUM_VERTICES_SAMPLED):
        source = np.random.randint(0, len(G))
        max_t_A_st = distances[source].max()

        x = torch.zeros_like(dataset_item.x)
        x[source] = torch.randn_like(dataset_item.x[source])
        x[source] = x[source].softmax(dim=-1)
        dataset_item.x.data = x

        out = model(dataset_item)
        out = (out / out.sum()).cpu().detach().numpy()
        out = np.nan_to_num(out, copy=True, nan=0.0, posinf=None, neginf=None)

        acc = 0.0
        for j in range(len(out)):
            acc += out[j] * distances[j, source]

        propagation = ((1/max_t_A_st) * acc).mean()
        total_effective_resistance = sum(get_resistances(G, source).values())

        pairs_runaway.append((total_effective_resistance, propagation))

    return pairs_runaway

def plot_data(ax, data, title):
    x = MinMaxScaler().fit_transform(data[:, 0].reshape(-1, 1)).flatten()
    y = MinMaxScaler().fit_transform(data[:, 1].reshape(-1, 1)).flatten()
    smooth_plot(x=x, y=y, ax=ax, halflife=2)
    ax.set_title(title, fontsize=20)

def main():
    dataset = TUDataset(root=ROOT_DIR, name=DATASET_NAME)

    pairs = {arch: [] for arch in ARCHS}

    for i, data in enumerate(dataset):
        try:
            for arch in ARCHS:
                arch_runaway = process_graph_data(data, arch)
                pairs[arch].append(tuple(np.array(arch_runaway).mean(axis=0).tolist()))
        except Exception as e:
            print(f"Error processing graph {i}: {str(e)}")

    fig, axs = plt.subplots(2, 2, figsize=(18, 15))
    titles = ["GIN", "Sage", "GCN", "GAT"]

    for ax, arch, title in zip(axs.flatten(), ARCHS, titles):
        data_array = np.array(pairs[arch])
        sorted_data = data_array[data_array[:, 0].argsort()]
        plot_data(ax, sorted_data, title)

    fig.text(0.5, 0.04, 'Normalized Total Effective Resistance', size=20, ha='center', va='center')
    fig.text(0.06, 0.5, 'Signal Propagation',  size=20, ha='center', va='center', rotation='vertical')
    plt.show()

if __name__ == "__main__":
    main()
