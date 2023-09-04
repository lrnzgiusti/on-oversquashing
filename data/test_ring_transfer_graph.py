import numpy as np
import torch
import pytest
from ring_transfer import generate_ring_transfer_graph, generate_ring_transfer_graph_dataset
from torch_geometric.data import Data

# Test the return type of the function
def test_return_type_transfer():
    result = generate_ring_transfer_graph(5, [1,0], False)
    assert isinstance(result, Data), "Return type should be of type Data"

# Test if the generated number of nodes is correct
def test_number_of_nodes_transfer():
    nodes = 7
    result = generate_ring_transfer_graph(nodes, [0,1], True)
    assert result.x.shape[0] == nodes, "Generated graph should have the specified number of nodes"

# Test if the generated number of edges is correct for a ring topology without crosses
def test_number_of_edges_no_crosses():
    nodes = 6
    result = generate_ring_transfer_graph(nodes, [1,0], False)
    assert result.edge_index.shape[1] == 2 * nodes, "Generated graph should have twice the number of edges as nodes for ring without crosses"

# Test if the generated number of edges is correct for a ring topology with crosses
def test_number_of_edges_with_crosses():
    nodes = 6
    result = generate_ring_transfer_graph(nodes, [0,1], True)
    # For even nodes, each node has 3 connections except for first and last node
    expected_edges = 4*nodes - 4
    assert result.edge_index.shape[1] == expected_edges, "Generated graph should have correct number of edges for ring with crosses"

# Test if the graph completes the ring by connecting last and first node
def test_ring_closure_transfer():
    nodes = 8
    result = generate_ring_transfer_graph(nodes, [1,0], True)
    assert [0, nodes - 1] in result.edge_index.T.tolist() and [nodes - 1, 0] in result.edge_index.T.tolist(), "Graph should close the ring"

# Test for various numbers of nodes using parametrize
@pytest.mark.parametrize("nodes", [3, 4, 5, 10, 15, 20])
def test_various_nodes_transfer(nodes):
    result = generate_ring_transfer_graph(nodes, [1,0], False)
    assert result.x.shape[0] == nodes, f"For {nodes} nodes, generated graph should have {nodes} nodes"
    # Ensure the edges match the expected number for a ring topology
    assert result.edge_index.shape[1] == 2 * nodes, f"For {nodes} nodes, generated graph should have {2*nodes} edges"

# Test if mask is correctly set for the source node
def test_mask_for_source_node_transfer():
    nodes = 8
    result = generate_ring_transfer_graph(nodes, [1,0], True)
    assert result.mask[0].item() == True and result.mask[1:].sum() == 0, "Only the source node should have mask set to True"

# Test that y tensor contains the correct value
def test_y_tensor_value_transfer():
    nodes = 6
    target_label = [0,1,0]
    result = generate_ring_transfer_graph(nodes, target_label, False)
    assert result.y.item() == torch.tensor(target_label).argmax().item(), "y tensor should contain the correct index of the target label"

# Test the upper limit of nodes (Optional, you can adjust this based on your use case)
@pytest.mark.parametrize("nodes", [100, 200, 500])
def test_upper_limit_nodes_transfer(nodes):
    result = generate_ring_transfer_graph(nodes, [1,0], True)
    assert result.x.shape[0] == nodes, f"For {nodes} nodes, generated graph should have {nodes} nodes"

# Test the return type of the function
def test_return_type_transfer_dataset():
    result = generate_ring_transfer_graph_dataset(5)
    assert isinstance(result, list) and isinstance(result[0], Data), "Return should be a list of Data type"

# Test the number of graphs in the dataset
def test_number_of_graphs():
    samples = 100
    result = generate_ring_transfer_graph_dataset(7, samples=samples)
    assert len(result) == samples, "Dataset should contain specified number of graphs"

# Test that dataset contains correct number of classes
def test_number_of_classes():
    classes = 4
    samples = 400
    result = generate_ring_transfer_graph_dataset(6, classes=classes, samples=samples)
    unique_classes = len(set(graph.y.item() for graph in result))
    assert unique_classes == classes, "Dataset should contain specified number of unique classes"

# Test that graphs have correct number of nodes
def test_number_of_nodes_dataset():
    nodes = 8
    result = generate_ring_transfer_graph_dataset(nodes)
    for graph in result:
        assert graph.x.shape[0] == nodes, "Each graph should have the specified number of nodes"

# Test distribution of samples among classes
def test_samples_distribution():
    classes = 5
    samples = 500
    result = generate_ring_transfer_graph_dataset(7, classes=classes, samples=samples)
    class_counts = [0] * classes
    for graph in result:
        class_counts[graph.y.item()] += 1
    for count in class_counts:
        assert count == samples // classes, "Each class should have an equal number of samples"

# Test if classes are distributed sequentially
def test_class_sequential_distribution():
    classes = 3
    samples = 300
    result = generate_ring_transfer_graph_dataset(5, classes=classes, samples=samples)
    previous_label = -1
    for graph in result:
        current_label = graph.y.item()
        assert current_label >= previous_label, "Classes in dataset should be distributed sequentially"
        previous_label = current_label

# Test for various numbers of nodes using parametrize
@pytest.mark.parametrize("nodes", [3, 4, 5, 10, 15, 20])
def test_various_nodes_dataset(nodes):
    result = generate_ring_transfer_graph_dataset(nodes)
    for graph in result:
        assert graph.x.shape[0] == nodes, f"For {nodes} nodes, each graph in dataset should have {nodes} nodes"

# Test with and without cross edges
@pytest.mark.parametrize("add_crosses", [True, False])
def test_cross_edges_dataset(add_crosses):
    nodes = 5
    result = generate_ring_transfer_graph_dataset(nodes, add_crosses=add_crosses)
    # Just an assertion to ensure function runs correctly with both flags, more detailed edge testing can be done in generate_ring_transfer_graph function tests
    assert len(result) > 0, "Dataset should be generated with and without cross edges"

# Test for various number of samples using parametrize
@pytest.mark.parametrize("samples", [50, 100, 500, 1000])
def test_various_samples_dataset(samples):
    result = generate_ring_transfer_graph_dataset(6, samples=samples)
    assert len(result) == samples, f"For {samples} specified samples, dataset should contain {samples} graphs"

# Test for various number of classes using parametrize
@pytest.mark.parametrize("classes", [2, 4, 8, 10])
def test_various_classes_dataset(classes):
    result = generate_ring_transfer_graph_dataset(6, classes=classes)
    unique_classes = len(set(graph.y.item() for graph in result))
    assert unique_classes == classes, f"Dataset should contain {classes} unique classes"

# Test upper limit of nodes (Optional, you can adjust based on use case)
@pytest.mark.parametrize("nodes", [100, 200, 500])
def test_upper_limit_nodes_dataset(nodes):
    result = generate_ring_transfer_graph_dataset(nodes)
    for graph in result:
        assert graph.x.shape[0] == nodes, f"For {nodes} nodes, each graph in dataset should have {nodes} nodes"