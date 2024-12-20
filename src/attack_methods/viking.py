"""
VikingAttack Module
--------------------
This module implements the VikingAttack class for adversarial attacks on graphs by perturbing edges involving selected nodes.

Classes:
    - VikingAttack: A class to perform perturbation-based edge removal attacks on a graph dataset.
"""

from .base_attack import BaseAttack
import torch
from torch_geometric.utils import remove_self_loops, to_undirected
from typing import Any, List, Tuple


class VikingAttack(BaseAttack):
    """
    VikingAttack Class
    -------------------
    Implements an adversarial attack by perturbing edges involving specified nodes in a graph.

    Attributes:
        data (Any): The graph dataset.
        params (dict): Additional parameters for the attack.
    """

    def __init__(self, data: Any, **kwargs):
        """
        Initialize the VikingAttack method.

        Args:
            data (Any): The graph dataset.
            **kwargs: Additional parameters for the attack.
        """
        self.data = data
        self.params = kwargs
        self.device = kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


    def perturbation_attack(
        self, data: Any, selected_nodes: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Perform the Viking perturbation attack by removing edges involving selected nodes.

        Args:
            data (Any): The graph dataset.
            selected_nodes (torch.Tensor): Nodes to target for edge removal.

        Returns:
            Tuple[torch.Tensor, List[Tuple[int, int]]]:
                - retained_edges (torch.Tensor): The edge index after removal of edges.
                - edges_to_remove (List[Tuple[int, int]]): A list of removed edges.
        """

        n_removals = len(selected_nodes)

        # Move edge_index to GPU
        edge_index = data.edge_index.to(self.device)

        # Mask edges involving selected nodes
        mask = torch.isin(edge_index[0], selected_nodes) | torch.isin(edge_index[1], selected_nodes)
        selected_edges = edge_index[:, mask]

        # Ignore nodes with no edges
        if selected_edges.size(1) == 0:
            return edge_index, []  # No edges to remove

        # Ensure enough edges are available for removal
        if selected_edges.size(1) < n_removals:
            raise ValueError("Not enough edges to remove the specified number.")

        # Shuffle selected edges and select exactly n_removals edges
        selected_edges = selected_edges.T[torch.randperm(selected_edges.size(1))].T
        edges_to_remove = selected_edges[:, :n_removals]

        # Remove selected edges from the edge list
        retained_edges = edge_index.T[~torch.isin(edge_index.T, edges_to_remove.T).all(dim=1)].T

        # Convert edges_to_remove to a list of tuples for compatibility
        edges_to_remove_list = edges_to_remove.t().tolist()

        return retained_edges, edges_to_remove_list

    def attack(
        self, data: Any, selected_nodes: torch.Tensor
    ) -> Tuple[Any, List[Tuple[int, int]]]:
        """
        Execute the Viking perturbation attack.

        Args:
            data (Any): The graph dataset.
            selected_nodes (torch.Tensor): Nodes to target for edge removal.

        Returns:
            Tuple[Any, List[Tuple[int, int]]]:
                - updated_data (Any): The modified graph dataset with updated edges.
                - removed_edges (List[Tuple[int, int]]): A list of removed edges.
        """

        # Ensure selected_nodes is a 1D tensor
        if isinstance(selected_nodes, torch.Tensor) and selected_nodes.ndimension() == 0:
            selected_nodes = selected_nodes.unsqueeze(0)

        selected_nodes = selected_nodes.to(self.device)

        # Clone the original data
        data_copy = data.clone()

        # Perform the perturbation attack
        updated_edge_index, removed_edges = self.perturbation_attack(data_copy, selected_nodes)

        # Create a new dataset with the updated edge index
        updated_data = data_copy.clone()
        updated_data.edge_index = updated_edge_index

        return updated_data, removed_edges