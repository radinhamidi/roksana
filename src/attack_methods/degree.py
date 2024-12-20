"""
DegreeAttack Module
--------------------
This module implements the DegreeAttack class for adversarial attacks on graphs by selectively removing edges
connected to nodes with the highest degree.

Classes:
    - DegreeAttack: A class to perform degree-based edge removal attacks on a graph dataset.
"""

from .base_attack import BaseAttack
import torch
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import degree as calculate_degree
from typing import Any, List, Tuple


class DegreeAttack(BaseAttack):
    """
    DegreeAttack Class
    -------------------
    Implements an adversarial attack that removes edges based on the degree of connected nodes.

    Attributes:
        data (Any): The graph dataset.
        params (dict): Additional parameters for the attack.
    """

    def __init__(self, data: Any, **kwargs):
        """
        Initialize the DegreeAttack method.

        Args:
            data (Any): The graph dataset.
            **kwargs: Additional parameters for the attack.
        """
        self.data = data
        self.params = kwargs
        self.device = kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


    def attack(self, data: Any, selected_nodes: torch.Tensor) -> Tuple[Any, List[Tuple[int, int]]]:
        """
        Perform the degree-based attack on the graph dataset.

        Args:
            data (Any): The graph dataset.
            selected_nodes (torch.Tensor): Nodes to target for edge removal. Must be a 1D tensor.

        Returns:
            Tuple[Any, List[Tuple[int, int]]]:
                - updated_data (Any): The modified graph dataset with updated edges.
                - removed_edges (List[Tuple[int, int]]): A list of removed edges.
        """

        # Move data to the appropriate device
        original_data = data.to(self.device)
        edge_index = original_data.edge_index.clone().to(self.device)
        edges = edge_index.t().tolist()

        # Ensure selected_nodes is a 1D tensor
        if isinstance(selected_nodes, torch.Tensor) and selected_nodes.ndimension() == 0:
            selected_nodes = selected_nodes.unsqueeze(0)

        selected_nodes = selected_nodes.to(self.device)

        # Calculate the degree for each node
        degrees = calculate_degree(edge_index[0], num_nodes=original_data.num_nodes).to(self.device)

        removed_edges = []  # List to store removed edges

        # Track the number of edges removed to ensure only len(selected_nodes) edges are removed
        removed_edges_count = 0

        # Loop over each node in selected_nodes to remove one edge connected to the highest-degree node
        for node in selected_nodes.tolist():
            if removed_edges_count >= len(selected_nodes):
                break  # Stop once len(selected_nodes) edges are removed

            # Find edges associated with the current node
            node_edges = [(i, (src, dst)) for i, (src, dst) in enumerate(edges) if src == node or dst == node]

            if node_edges:
                # Find the edge connected to the highest-degree node
                edge_to_remove = max(
                    node_edges,
                    key=lambda x: degrees[x[1][1]] if x[1][0] == node else degrees[x[1][0]]
                )
                edges.pop(edge_to_remove[0])  # Remove the edge from the edge list
                removed_edges.append(edge_to_remove[1])  # Track the removed edge
                removed_edges_count += 1

        # Create a new edge_index tensor from the modified edges
        new_edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t().contiguous()

        # Remove any self-loops
        new_edge_index, _ = remove_self_loops(new_edge_index)

        # Assign the modified edge_index to the dataset
        updated_data = data.clone().to(self.device)
        updated_data.edge_index = new_edge_index

        return updated_data, removed_edges
