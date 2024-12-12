from roksana.src.attack_methods.base_attack import *
import torch
from torch_geometric.utils import remove_self_loops, to_undirected
import random


class RandomAttack(BaseAttack):
    def __init__(self, data: Any, **kwargs):
        """
        Initialize the RandomAttack method.

        Args:
            data (Any): The graph dataset.
            **kwargs: Additional parameters for the attack.
        """
        self.data = data
        self.params = kwargs

    def attack(self, data, selected_nodes):
        # Ensure device compatibility

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
