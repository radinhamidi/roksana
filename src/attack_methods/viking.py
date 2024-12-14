from .base_attack import BaseAttack
import torch
from torch_geometric.utils import remove_self_loops, to_undirected
from roksana.src.attack_methods.base_attack import *


class VikingAttack(BaseAttack):
    def __init__(self, data: Any, **kwargs):
        """
        Initialize the VikingAttack method.

        Args:
            data (Any): The graph dataset.
            **kwargs: Additional parameters for the attack.
        """
        self.data = data
        self.params = kwargs

    def perturbation_attack(self, data, selected_nodes):
        """
        Perform the Viking perturbation attack by removing edges involving selected nodes.

        Args:
            data (Any): The graph dataset.
            selected_nodes (torch.Tensor): Nodes to target for edge removal.

        Returns:
            retained_edges (torch.Tensor): Edge index after the removal.
            edges_to_remove (List[Tuple[int, int]]): List of removed edges.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_removals = len(selected_nodes)

        # Move edge_index to GPU
        edge_index = data.edge_index.to(device)

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

    def attack(self, data, selected_nodes):
        """
        Execute the Viking perturbation attack.

        Args:
            data (Any): The graph dataset.
            selected_nodes (torch.Tensor): Nodes to target for edge removal.

        Returns:
            updated_data (Any): The modified graph dataset.
            removed_edges (List[Tuple[int, int]]): List of removed edges.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Ensure selected_nodes is a 1D tensor
        if isinstance(selected_nodes, torch.Tensor) and selected_nodes.ndimension() == 0:
            selected_nodes = selected_nodes.unsqueeze(0)

        selected_nodes = selected_nodes.to(device)

        # Clone the original data
        data_copy = data.clone()

        # Perform the perturbation attack
        updated_edge_index, removed_edges = self.perturbation_attack(data_copy, selected_nodes)

        # Create a new dataset with the updated edge index
        updated_data = data_copy.clone()
        updated_data.edge_index = updated_edge_index

        return updated_data, removed_edges
