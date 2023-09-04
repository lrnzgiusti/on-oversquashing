import pytest
import numpy as np

import torch

from torch_geometric.data import Data
from ring_transfer import generate_ring_lookup_graph, generate_ringlookup_graph_dataset

# Test the return type of the function
def test_return_type():
    result = generate_ring_lookup_graph(5)
    assert isinstance(result, Data), "Return type should be of type Data"

# Test if the generated number of nodes is correct
def test_number_of_nodes():
    nodes = 7
    result = generate_ring_lookup_graph(nodes)
    assert result.x.shape[0] == nodes, "Generated graph should have the specified number of nodes"

# Test if the generated number of edges is correct for a ring topology
def test_number_of_edges():
    nodes = 6
    result = generate_ring_lookup_graph(nodes)
    assert result.edge_index.shape[1] == 2 * nodes, "Generated graph should have twice the number of edges as nodes"

# Test if the generated graph contains self-loops
def test_self_loops():
    nodes = 5
    result = generate_ring_lookup_graph(nodes)
    assert not any(result.edge_index[0] == result.edge_index[1]), "Generated graph should not have self-loops"

# Test if the source node has the correct features
def test_source_node_features():
    nodes = 6
    result = generate_ring_lookup_graph(nodes)
    assert result.x[0].sum() == 1, "Source node should have a single one-hot encoded key"

# Test if the graph completes the ring by connecting last and first node
def test_ring_closure():
    nodes = 8
    result = generate_ring_lookup_graph(nodes)
    assert [0, nodes - 1] in result.edge_index.T.tolist() and [nodes - 1, 0] in result.edge_index.T.tolist(), "Graph should close the ring"

# Test for various numbers of nodes using parametrize
@pytest.mark.parametrize("nodes", [3, 4, 5, 10, 15, 20])
def test_various_nodes(nodes):
    result = generate_ring_lookup_graph(nodes)
    assert result.x.shape[0] == nodes, f"For {nodes} nodes, generated graph should have {nodes} nodes"
    assert result.edge_index.shape[1] == 2 * nodes, f"For {nodes} nodes, generated graph should have {2*nodes} edges"

# Test that all non-source nodes have unique features
def test_unique_features():
    nodes = 7
    result = generate_ring_lookup_graph(nodes)
    non_source_features = result.x[1:].numpy()
    unique_features = np.unique(non_source_features, axis=0)
    assert len(unique_features) == nodes - 1, "All non-source nodes should have unique features"

# Test if mask is correctly set for the source node
def test_mask_for_source_node():
    nodes = 8
    result = generate_ring_lookup_graph(nodes)
    assert result.mask[0].item() == True and result.mask[1:].sum() == 0, "Only the source node should have mask set to True"

# Test that y tensor contains the correct value
def test_y_tensor_value():
    nodes = 6
    result = generate_ring_lookup_graph(nodes)
    assert 0 <= result.y.item() < nodes - 1, "y tensor should contain a value from the vals array"

# Test the upper limit of nodes (Optional, you can adjust this based on your use case)
@pytest.mark.parametrize("nodes", [100, 200, 500])
def test_upper_limit_nodes(nodes):
    result = generate_ring_lookup_graph(nodes)
    assert result.x.shape[0] == nodes, f"For {nodes} nodes, generated graph should have {nodes} nodes"



# Test if the dataset generated has the expected length.
@pytest.mark.parametrize(
    "nodes, samples, expected_length",
    [
        (5, 10, 10),               # Basic test with small nodes and samples
        (10, 1, 1),               # Only one sample
        (50, 100, 100),           # Larger graph with moderate samples
        (2, 100, 100),            # Minimal nodes
        (100, 0, 0),              # Zero samples
    ]
)
def test_generate_ringlookup_graph_dataset_length(nodes, samples, expected_length):
    dataset = generate_ringlookup_graph_dataset(nodes, samples)
    assert len(dataset) == expected_length

# Test if each generated graph in the dataset has the correct number of nodes.
def test_generate_ringlookup_graph_dataset_nodes():
    nodes = 10
    samples = 5
    dataset = generate_ringlookup_graph_dataset(nodes, samples)
    for data in dataset:
        assert data.x.size(0) == nodes

# Test that the generated dataset has non-empty values for each graph's features.
def test_generate_ringlookup_graph_dataset_non_empty():
    dataset = generate_ringlookup_graph_dataset(5, 5)
    for data in dataset:
        assert data.x is not None
        assert data.edge_index is not None

# Test the mask generated for each graph. 
# It should have the correct size and a True value for the target node.  
def test_generate_ringlookup_graph_dataset_mask():
    dataset = generate_ringlookup_graph_dataset(5, 5)
    for data in dataset:
        # Check the mask has the correct size and has a True value for the target node
        assert data.mask.size(0) == 5
        assert data.mask[0]

# Test that each graph in the dataset has an associated label.
def test_generate_ringlookup_graph_dataset_label():
    dataset = generate_ringlookup_graph_dataset(5, 5)
    for data in dataset:
        assert data.y is not None
