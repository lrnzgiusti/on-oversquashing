import pytest
from typing import List
import numpy as np
import torch
from torch_geometric.data import Data
from ring_transfer import generate_lollipop_transfer_graph, generate_lollipop_transfer_graph_dataset

# Utility function to check if two nodes are connected
def are_connected(edge_index, node1, node2):
    return [node1, node2] in edge_index.T.tolist() or [node2, node1] in edge_index.T.tolist()

# Tests for generate_lollipop_transfer_graph
@pytest.mark.parametrize("nodes, target_label", [
    (4, [1, 0]),
    (5, [0, 1]),
    (6, [1, 0]),
    (7, [0, 1, 0]),
    (8, [0, 0, 1]),
    (9, [1, 0, 0]),
    (10, [0, 1, 0])
])
def test_generate_lollipop_transfer_graph(nodes, target_label):
    graph = generate_lollipop_transfer_graph(nodes, target_label)
    
    # Check node attributes
    assert torch.equal(graph.x[0], torch.tensor([0.0] * len(target_label), dtype=torch.float32))
    assert torch.equal(graph.x[-1], torch.tensor(target_label, dtype=torch.float32))

    # Check edges for clique
    for i in range(nodes // 2):
        for j in range(nodes // 2):
            if i != j:
                assert are_connected(graph.edge_index, i, j)

    # Check edges for path
    for i in range(nodes // 2, nodes - 1):
        assert are_connected(graph.edge_index, i, i + 1)
    
    # Check connection between last node of the clique and the first node of the path
    assert are_connected(graph.edge_index, nodes // 2 - 1, nodes // 2)
    
    # Check mask
    assert graph.mask[0]
    assert not graph.mask[1:].any()
    
    # Check y (target label)
    assert graph.y.item() == np.argmax(target_label)

# Tests for generate_lollipop_transfer_graph_dataset
@pytest.mark.parametrize("nodes, classes, samples", [
    (4, 3, 9),
    (5, 4, 8),
    (6, 5, 10),
    (7, 3, 9),
    (8, 2, 4),
    (9, 5, 15),
    (10, 4, 12),
    (11, 3, 9),
    (12, 4, 8),
    (13, 5, 10)
])
def test_generate_lollipop_transfer_graph_dataset(nodes, classes, samples):
    dataset = generate_lollipop_transfer_graph_dataset(nodes, classes, samples)
    
    # Check dataset length
    assert len(dataset) == samples
    
    # Validate graphs in dataset
    for i, graph in enumerate(dataset):
        expected_label = np.zeros(classes)
        expected_label[i // (samples // classes)] = 1.0
        assert torch.equal(graph.x[-1], torch.tensor(expected_label, dtype=torch.float32))
        assert graph.y.item() == np.argmax(expected_label)


# Test for minimum number of nodes (should be at least 2)
def test_minimum_nodes():
    with pytest.raises(ValueError):
        generate_lollipop_transfer_graph(1, [1, 0])

# Test if function raises an exception when provided with inconsistent label length
def test_inconsistent_target_label():
    with pytest.raises(ValueError):
        generate_lollipop_transfer_graph(5, [1])

# Test if the graph has the expected number of edges
@pytest.mark.parametrize("nodes", [4, 5, 6, 7, 8])
def test_number_of_edges(nodes):
    graph = generate_lollipop_transfer_graph(nodes, [1, 0])
    # For a lollipop graph: edges = (nodes // 2) * (nodes // 2 - 1) + nodes - 1
    expected_edges = (nodes // 2 * (nodes // 2 - 1)) + 2 * (nodes - nodes // 2) + 2
    assert graph.edge_index.shape[1] == expected_edges, f"Expected {expected_edges}, but got {graph.edge_index.shape[1]}"

# Test if all nodes have degree at least one (i.e., all nodes are connected)
@pytest.mark.parametrize("nodes", [4, 5, 6, 7, 8])
def test_all_nodes_connected(nodes):
    graph = generate_lollipop_transfer_graph(nodes, [1, 0])
    for i in range(nodes):
        assert (graph.edge_index == i).any()

# Test if node features are of type float32
def test_node_features_dtype():
    graph = generate_lollipop_transfer_graph(5, [1, 0])
    assert graph.x.dtype == torch.float32

# Test if edge index is of type long
def test_edge_index_dtype():
    graph = generate_lollipop_transfer_graph(5, [1, 0])
    assert graph.edge_index.dtype == torch.long

# Test if mask is of type bool
def test_mask_dtype():
    graph = generate_lollipop_transfer_graph(5, [1, 0])
    assert graph.mask.dtype == torch.bool

# Test if only the first node is masked
def test_mask_first_node_only():
    graph = generate_lollipop_transfer_graph(5, [1, 0])
    assert graph.mask[0]
    assert not graph.mask[1:].any()

# Test if target label is of type long
def test_target_label_dtype():
    graph = generate_lollipop_transfer_graph(5, [1, 0])
    assert graph.y.dtype == torch.long

# Test if only the last node contains the target label
def test_last_node_target_label():
    target_label = [0, 1, 0]
    graph = generate_lollipop_transfer_graph(5, target_label)
    assert torch.equal(graph.x[-1], torch.tensor(target_label, dtype=torch.float32))
    for i in range(len(graph.x) - 1):
        assert not torch.equal(graph.x[i], torch.tensor(target_label, dtype=torch.float32))

# Test for minimum number of samples (should be at least 1)
def test_minimum_samples():
    with pytest.raises(ValueError):
        generate_lollipop_transfer_graph_dataset(5, 5, 0)

# Test if function raises an exception when provided with fewer classes than samples
def test_more_samples_than_classes():
    dataset = generate_lollipop_transfer_graph_dataset(5, 3, 5)
    assert len(dataset) == 5

# Test if the generated dataset has the expected length
@pytest.mark.parametrize("samples", [100, 500, 1000])
def test_dataset_length(samples):
    dataset = generate_lollipop_transfer_graph_dataset(5, 5, samples)
    assert len(dataset) == samples

# Test if every sample in dataset has the expected number of nodes
@pytest.mark.parametrize("nodes", [5, 6, 7, 8])
def test_samples_nodes_length(nodes):
    dataset = generate_lollipop_transfer_graph_dataset(nodes, 5, 100)
    for graph in dataset:
        assert graph.x.shape[0] == nodes

# Test if target labels are distributed uniformly across the dataset
def test_uniform_label_distribution():
    classes = 5
    samples = 100
    dataset = generate_lollipop_transfer_graph_dataset(5, classes, samples)
    class_counts = [0] * classes
    for graph in dataset:
        label = graph.y.item()
        class_counts[label] += 1
    for count in class_counts:
        assert count == samples // classes

# Test if dataset generation supports additional kwargs
def test_generate_with_additional_kwargs():
    def mock_lollipop_graph(nodes, target_label):
        return "mock_graph"
    dataset = generate_lollipop_transfer_graph_dataset(5, 5, 100, custom_function=mock_lollipop_graph)
    assert all(data == "mock_graph" for data in dataset)

# Test if function raises exception for negative nodes
def test_negative_nodes():
    with pytest.raises(ValueError):
        generate_lollipop_transfer_graph_dataset(-5, 5, 100)

# Test if function raises exception for negative classes
def test_negative_classes():
    with pytest.raises(ValueError):
        generate_lollipop_transfer_graph_dataset(5, -5, 100)

# Test if function raises exception for negative samples
def test_negative_samples():
    with pytest.raises(ValueError):
        generate_lollipop_transfer_graph_dataset(5, 5, -100)

# Test if node features are of type float32 for all samples in the dataset
def test_node_features_dtype_in_dataset():
    dataset = generate_lollipop_transfer_graph_dataset(5, 5, 100)
    for graph in dataset:
        assert graph.x.dtype == torch.float32

