#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module to generate various types of graphs: Ring, Tree, and Lollipop.


Authors:
    - CWN project authors
    - On Oversquashing project authors
"""


import numpy as np
import torch
import random
from torch_geometric.data import Data
from sklearn.preprocessing import LabelBinarizer
from typing import List



def generate_ring_lookup_graph(nodes:int):
    """
    Generate a dictionary lookup ring graph.

    Args:
    - nodes (int): Number of nodes in the ring.

    Returns:
    - Data: Torch geometric data structure containing graph details.
    
    Note: This function is currently deprecated.
    """
    
    if nodes <= 1: raise ValueError("Minimum of two nodes required")
    # Generate unique keys and random values for all the nodes except the source node
    keys = np.arange(1, nodes)  # Create an array of unique keys from 1 to nodes-1
    vals = np.random.permutation(nodes - 1)  # Shuffle these keys to serve as values

    # One-hot encode keys and values
    oh_keys = np.array(LabelBinarizer().fit_transform(keys))
    oh_vals = np.array(LabelBinarizer().fit_transform(vals))

    # Concatenate one-hot encoded keys and values to create node features
    oh_all = np.concatenate((oh_keys, oh_vals), axis=-1)
    x = np.empty((nodes, oh_all.shape[1]))
    x[1:, :] = oh_all  # Add these as features for all nodes except the source node

    # Randomly select one key for the source node and associate a random value to it
    key_idx = random.randint(0, nodes - 2)  # Random index for choosing a key
    val = vals[key_idx]  # Corresponding value from the randomized list

    # Set the source node's features: all zeros except the chosen key which is set to one-hot encoded value
    x[0, :] = 0  
    x[0, :oh_keys.shape[1]] = oh_keys[key_idx]  # Assigning one-hot encoded key to source node

    # Convert to tensor for PyTorch compatibility
    x = torch.tensor(x, dtype=torch.float32)

    # Generate edges for the ring topology
    edge_index = []
    for i in range(nodes - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    
    # Add the edges to complete the ring
    edge_index.append([0, nodes - 1])
    edge_index.append([nodes - 1, 0])

    # Convert to tensor and transpose for Torch Geometric compatibility
    edge_index = np.array(edge_index, dtype=np.compat.long).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Create a mask to highlight the source node (used later for graph-level predictions)
    mask = torch.zeros(nodes, dtype=torch.bool)
    mask[0] = 1  # Source node is the node of interest

    # Add a label based on the random value associated with the source node's key
    y = torch.tensor([val], dtype=torch.long)

    # Return a Torch Geometric Data object containing all graph information
    return Data(x=x, edge_index=edge_index, mask=mask, y=y)


def generate_ringlookup_graph_dataset(nodes:int, samples:int=10000):
    """
    Generate a dataset of ring lookup graphs.

    Args:
    - nodes (int): Number of nodes in each graph.
    - samples (int): Number of graphs in the dataset.

    Returns:
    - list[Data]: List of Torch geometric data structures.
    """
    if nodes <= 1: raise ValueError("Minimum of two nodes required")
    dataset = []
    for i in range(samples):
        graph = generate_ring_lookup_graph(nodes)
        dataset.append(graph)
    return dataset

def generate_ring_transfer_graph(nodes, target_label, add_crosses: bool):
    """
    Generate a ring transfer graph with an option to add crosses.

    Args:
    - nodes (int): Number of nodes in the graph.
    - target_label (list): Label of the target node.
    - add_crosses (bool): Whether to add cross edges in the ring.

    Returns:
    - Data: Torch geometric data structure containing graph details.
    """
    assert nodes > 1, ValueError("Minimum of two nodes required")
    # Determine the node directly opposite to the source (node 0) in the ring
    opposite_node = nodes // 2

    # Initialise feature matrix with a uniform feature. 
    # This serves as a placeholder for features of all nodes.
    x = np.ones((nodes, len(target_label)))

    # Set feature of the source node to 0 and the opposite node to the target label
    x[0, :] = 0.0
    x[opposite_node, :] = target_label

    # Convert the feature matrix to a torch tensor for compatibility with Torch geometric
    x = torch.tensor(x, dtype=torch.float32)

    # List to store edge connections in the graph
    edge_index = []
    for i in range(nodes-1):
        # Regular connections that make the ring
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
        
        # Conditionally add cross edges, if desired
        if add_crosses and i < opposite_node:
            # Add edges from a node to its direct opposite
            edge_index.append([i, nodes - 1 - i])
            edge_index.append([nodes - 1 - i, i])

            # Extra logic for ensuring additional "cross" edges in some conditions
            if nodes + 1 - i < nodes:
                edge_index.append([i, nodes + 1 - i])
                edge_index.append([nodes + 1 - i, i])

    # Close the ring by connecting the last and the first nodes
    edge_index.append([0, nodes - 1])
    edge_index.append([nodes - 1, 0])

    # Convert edge list to a torch tensor
    edge_index = np.array(edge_index, dtype=np.compat.long).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Create a mask to identify the target node in the graph. Only the source node (index 0) is marked true.
    mask = torch.zeros(nodes, dtype=torch.bool)
    mask[0] = 1

    # Determine the graph's label based on the target label. This is a singular value indicating the index of the target label.
    y = torch.tensor([np.argmax(target_label)], dtype=torch.long)

    # Return the graph with nodes, edges, mask and the label
    return Data(x=x, edge_index=edge_index, mask=mask, y=y)


def generate_ring_transfer_graph_dataset(nodes:int, add_crosses:bool=False, classes:int=5, samples:int=10000, **kwargs):
    """
    Generate a dataset of ring transfer graphs.

    Args:
    - nodes (int): Number of nodes in each graph.
    - add_crosses (bool): Whether to add cross edges in the ring.
    - classes (int): Number of different classes or labels.
    - samples (int): Number of graphs in the dataset.

    Returns:
    - list[Data]: List of Torch geometric data structures.
    """
    if nodes <= 1: raise ValueError("Minimum of two nodes required")
    dataset = []
    samples_per_class = samples // classes
    for i in range(samples):
        label = i // samples_per_class
        target_class = np.zeros(classes)
        target_class[label] = 1.0
        graph = generate_ring_transfer_graph(nodes, target_class, add_crosses)
        dataset.append(graph)
    return dataset

def generate_tree_transfer_graph(depth:int, target_label:List[int], arity:int):
    """
    Generate a tree transfer graph.

    Args:
    - depth (int): Depth of the tree.
    - target_label (list): Label of the target node.
    - arity (int): Number of children each node can have.

    Returns:
    - Data: Torch geometric data structure containing graph details.
    """
    if depth <= 0: raise ValueError("Minimum of depth one")
    # Calculate the total number of nodes based on the tree depth and arity
    num_nodes = int((arity ** (depth + 1) - 1) / (arity - 1))
    
    # Target node is the last node in the tree
    target_node = num_nodes - 1

    # Initialize the feature matrix with a constant feature vector
    x = np.ones((num_nodes, len(target_label)))

    # Set root node and target node feature values
    x[0, :] = 0.0
    x[target_node, :] = target_label
    x = torch.tensor(x, dtype=torch.float32)

    edge_index = []

    # To keep track of the current child node while iterating
    last_child_counter = 0
    
    # Loop to generate the edges of the tree
    for i in range(num_nodes - arity ** depth + 1):
        for child in range(1, arity + 1):
            # Ensure we don't exceed the total number of nodes
            if last_child_counter + child > num_nodes - 1: 
                break

            # Add edges for the current node and its children
            edge_index.append([i, last_child_counter + child])
            edge_index.append([last_child_counter + child, i])

        # Update the counter to point to the last child of the current node
        last_child_counter += arity

    # Convert edge index to torch tensor format
    edge_index = np.array(edge_index, dtype=np.compat.long).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Create a mask for the root node of the graph
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[0] = 1

    # Convert the target label into a single value format
    y = torch.tensor([np.argmax(target_label)], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, mask=mask, y=y)

def generate_tree_transfer_graph_dataset(depth:int, arity:int, classes:int=5, samples:int=10000, **kwargs):
    """
    Generate a dataset of tree transfer graphs.

    Args:
    - depth (int): Depth of the tree in each graph.
    - arity (int): Number of children each node can have.
    - classes (int): Number of different classes or labels.
    - samples (int): Number of graphs in the dataset.

    Returns:
    - list[Data]: List of Torch geometric data structures.
    """
    dataset = []
    samples_per_class = samples // classes
    for i in range(samples):
        label = i // samples_per_class
        target_class = np.zeros(classes)
        target_class[label] = 1.0
        graph = generate_tree_transfer_graph(depth, target_class, arity)
        dataset.append(graph)
    return dataset

def generate_lollipop_transfer_graph(nodes:int, target_label:List[int]):
    """
    Generate a lollipop transfer graph.

    Args:
    - nodes (int): Total number of nodes in the graph.
    - target_label (list): Label of the target node.

    Returns:
    - Data: Torch geometric data structure containing graph details.
    """
    if nodes <= 1: raise ValueError("Minimum of two nodes required")    
    # Initialize node features. The first node gets 0s, while the last gets the target label
    x = np.ones((nodes, len(target_label)))
    x[0, :] = 0.0
    x[nodes - 1, :] = target_label
    x = torch.tensor(x, dtype=torch.float32)

    edge_index = []

    # Construct a clique for the first half of the nodes, 
    # where each node is connected to every other node except itself
    for i in range(nodes // 2):
        for j in range(nodes // 2):
            if i == j:  # Skip self-loops
                continue
            edge_index.append([i, j])
            edge_index.append([j, i])

    # Construct a path (a sequence of connected nodes) for the second half of the nodes
    for i in range(nodes // 2, nodes - 1):
        edge_index.append([i, i+1])
        edge_index.append([i+1, i])

    # Connect the last node of the clique to the first node of the path
    edge_index.append([nodes // 2 - 1, nodes // 2])
    edge_index.append([nodes // 2, nodes // 2 - 1])

    # Convert the edge index list to a torch tensor
    edge_index = np.array(edge_index, dtype=np.compat.long).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Create a mask to indicate the target node (in this case, the first node)
    mask = torch.zeros(nodes, dtype=torch.bool)
    mask[0] = 1

    # Convert the one-hot encoded target label to its corresponding class index
    y = torch.tensor([np.argmax(target_label)], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, mask=mask, y=y)


def generate_lollipop_transfer_graph_dataset(nodes:int, classes:int=5, samples:int=10000, **kwargs):
    """
    Generate a dataset of lollipop transfer graphs.

    Args:
    - nodes (int): Total number of nodes in each graph.
    - classes (int): Number of different classes or labels.
    - samples (int): Number of graphs in the dataset.

    Returns:
    - list[Data]: List of Torch geometric data structures.
    """
    if nodes <= 1: raise ValueError("Minimum of two nodes required")
    dataset = []
    samples_per_class = samples // classes
    for i in range(samples):
        label = i // samples_per_class
        target_class = np.zeros(classes)
        target_class[label] = 1.0
        graph = generate_lollipop_transfer_graph(nodes, target_class)
        dataset.append(graph)
    return dataset