import torch
import pytest
from ring_transfer import generate_tree_transfer_graph, generate_tree_transfer_graph_dataset
from torch_geometric.data import Data



# Test the return type of the function
def test_return_type():
    result = generate_tree_transfer_graph(2, [0, 1, 0], 2)
    assert isinstance(result, Data), "Return type should be of type Data"

# Test if the generated number of nodes is correct based on tree depth and arity
def test_number_of_nodes():
    depth, arity = 3, 2
    expected_nodes = (arity ** (depth + 1) - 1) // (arity - 1)
    result = generate_tree_transfer_graph(depth, [0, 1, 0], arity)
    assert result.x.shape[0] == expected_nodes, f"Expected {expected_nodes} nodes for depth {depth} and arity {arity}"

# Test if target node has the correct features
def test_target_node_features():
    label = [0, 1, 0]
    result = generate_tree_transfer_graph(3, label, 2)
    assert (result.x[-1] == torch.tensor(label)).all(), "Target node should have the specified target label"

# Test if root node has correct features
def test_root_node_features():
    result = generate_tree_transfer_graph(3, [0, 1, 0], 2)
    assert result.x[0].sum() == 0.0, "Root node should have feature values of 0.0"

# Test if the number of edges is consistent with a tree structure
def test_number_of_edges():
    depth, arity = 2, 3
    result = generate_tree_transfer_graph(depth, [0, 1, 0], arity)
    expected_edges = (arity ** (depth + 1) - arity) * 2  # Each edge is bidirectional
    assert result.edge_index.shape[1] == expected_edges//2, f"Expected {expected_edges} edges for depth {depth} and arity {arity}"

# Test that all edges are valid (no out-of-bounds indices)
def test_valid_edges():
    result = generate_tree_transfer_graph(4, [0, 1, 0], 2)
    max_node_index = result.x.shape[0] - 1
    assert all((0 <= idx <= max_node_index) for idx in result.edge_index.reshape(-1)), "All edge indices should be valid node indices"

# Test if mask is correctly set for the root node
def test_mask_for_root_node():
    result = generate_tree_transfer_graph(4, [0, 1, 0], 2)
    assert result.mask[0].item() == True and result.mask[1:].sum() == 0, "Only the root node should have mask set to True"

# Test the y tensor value (target label)
def test_y_tensor_value():
    label = [0, 1, 0]
    result = generate_tree_transfer_graph(4, label, 2)
    assert result.y.item() == label.index(1), "y tensor should contain the index of the 1 in the target label"

# Test for various tree depths and arities using parametrize
@pytest.mark.parametrize("depth, arity", [(2, 2), (3, 2), (2, 3), (3, 3)])
def test_various_depths_and_arities(depth, arity):
    result = generate_tree_transfer_graph(depth, [0, 1, 0], arity)
    expected_nodes = (arity ** (depth + 1) - 1) // (arity - 1)
    assert result.x.shape[0] == expected_nodes, f"Expected {expected_nodes} nodes for depth {depth} and arity {arity}"

# Test the upper limit of tree depth and arity (Optional, you can adjust this based on your use case)
@pytest.mark.parametrize("depth, arity", [(5, 2), (6, 2)])
def test_upper_limit_depth_and_arity(depth, arity):
    result = generate_tree_transfer_graph(depth, [0, 1, 0], arity)
    expected_nodes = (arity ** (depth + 1) - 1) // (arity - 1)
    assert result.x.shape[0] == expected_nodes, f"Expected {expected_nodes} nodes for depth {depth} and arity {arity}"



# Test if the returned dataset is a list
def test_return_type():
    dataset = generate_tree_transfer_graph_dataset(2, 2)
    assert isinstance(dataset, list), "Return type should be a list"

# Test if the returned dataset has the correct number of graphs
def test_dataset_size():
    samples = 100
    dataset = generate_tree_transfer_graph_dataset(2, 2, samples=samples)
    assert len(dataset) == samples, f"Dataset should contain {samples} graphs"

# Test if each graph in the dataset is of type Data
def test_graph_type():
    dataset = generate_tree_transfer_graph_dataset(2, 2)
    assert all(isinstance(graph, Data) for graph in dataset), "Each graph in the dataset should be of type Data"

# Test if the graphs in the dataset have the correct depth and arity
def test_depth_and_arity():
    depth, arity = 3, 2
    dataset = generate_tree_transfer_graph_dataset(depth, arity)
    expected_nodes = (arity ** (depth + 1) - 1) // (arity - 1)
    for graph in dataset:
        assert graph.x.shape[0] == expected_nodes, f"Each graph should have {expected_nodes} nodes"

# Test if the graphs have the correct labels based on classes
def test_graph_labels():
    classes = 4
    dataset = generate_tree_transfer_graph_dataset(2, 2, classes=classes)
    labels = [graph.y.item() for graph in dataset]
    unique_labels = set(labels)
    assert len(unique_labels) == classes, f"Dataset should contain {classes} unique labels"

# Test if each class has an approximately equal number of samples
def test_balanced_classes():
    samples, classes = 1000, 5
    dataset = generate_tree_transfer_graph_dataset(2, 2, classes=classes, samples=samples)
    label_counts = [0] * classes
    for graph in dataset:
        label_counts[graph.y.item()] += 1
    for count in label_counts:
        assert samples // classes == count, "Each class should have an equal number of samples"

# Test for various depths and arities using parametrize
@pytest.mark.parametrize("depth, arity", [(2, 2), (3, 2), (2, 3), (3, 3)])
def test_various_depths_and_arities(depth, arity):
    dataset = generate_tree_transfer_graph_dataset(depth, arity)
    expected_nodes = (arity ** (depth + 1) - 1) // (arity - 1)
    for graph in dataset:
        assert graph.x.shape[0] == expected_nodes, f"Each graph should have {expected_nodes} nodes"

# Test that each label is one-hot encoded
def test_one_hot_encoded_labels():
    classes = 4
    dataset = generate_tree_transfer_graph_dataset(2, 2, classes=classes)
    for graph in dataset:
        label_index = graph.y.item()
        one_hot_label = graph.x[-1].numpy()
        assert one_hot_label[label_index] == 1 and sum(one_hot_label) == 1, "Labels should be one-hot encoded"

# Test the upper limit of samples (Optional, you can adjust this based on your use case)
@pytest.mark.parametrize("samples", [100, 200, 500])
def test_upper_limit_samples(samples):
    dataset = generate_tree_transfer_graph_dataset(2, 2, samples=samples)
    assert len(dataset) == samples, f"Dataset should contain {samples} graphs"