"""
RandomAttack Module
--------------------
This module implements the RandomAttack class for adversarial attacks on graphs by randomly removing edges 
connected to selected nodes.

Classes:
    - RandomAttack: A class to perform random edge removal attacks on a graph dataset.
"""

from .base_attack import BaseAttack
import torch
from torch_geometric.utils import remove_self_loops, to_undirected
import random
from typing import Any, Tuple, List


class RandomAttack(BaseAttack):
    """
    RandomAttack Class
    -------------------
    Implements an adversarial attack that randomly removes edges connected to specified nodes in a graph.

    Attributes:
        data (Any): The graph dataset.
        params (dict): Additional parameters for the attack.
    """

    def __init__(self, data: Any, **kwargs):
        """
        Initialize the RandomAttack method.

        Args:
            data (Any): The graph dataset.
            **kwargs: Additional parameters for the attack.
        """
        self.data = data
        self.params = kwargs
        self.device = kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


    def attack(self, data: Any, selected_nodes: torch.Tensor) -> Tuple[Any, List[Tuple[int, int]]]:
        """
        Perform the random edge removal attack.

        Args:
            data (Any): The graph dataset.
            selected_nodes (torch.Tensor): Nodes for which edges are to be removed. 
                                           Can be a single node or a tensor of nodes.

        Returns:
            Tuple[Any, List[Tuple[int, int]]]: 
                - The modified graph dataset with updated edges.
                - A list of removed edges.
        """
        # Normalize selected_nodes to be a 1D tensor, even if a single node is passed
        if isinstance(selected_nodes, torch.Tensor) and selected_nodes.ndimension() == 0:
            selected_nodes = selected_nodes.unsqueeze(0)

        selected_nodes = selected_nodes

        # Prepare the original dataset and edge list
        original_random_dataset = data
        edge_index = original_random_dataset.edge_index.clone()
        edges = edge_index.t().tolist()

        # Track the number of edges removed to ensure only len(selected_nodes) edges are removed
        removed_edges_count = 0
        removed_edges = []  # List to store removed edges

        # Loop over each node in selected_nodes to remove one random edge connected to it
        for node in selected_nodes.tolist():
            if removed_edges_count >= len(selected_nodes):
                break  # Stop once len(selected_nodes) edges are removed

            # Find edges associated with the current node
            node_edges = [(i, (src, dst)) for i, (src, dst) in enumerate(edges) if src == node or dst == node]

            if node_edges:
                # Select a random edge to remove
                edge_to_remove = random.choice(node_edges)
                edges.pop(edge_to_remove[0])
                removed_edges.append(edge_to_remove[1])  # Track the removed edge
                removed_edges_count += 1

        # Create a new edge_index tensor from the modified edges
        new_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Remove any self-loops, just in case
        new_edge_index, _ = remove_self_loops(new_edge_index)

        # Assign the modified edge_index to create the random_dataset
        degree_dataset = data.clone()
        degree_dataset.edge_index = new_edge_index
        updated_data = degree_dataset

        return updated_data, removed_edges
