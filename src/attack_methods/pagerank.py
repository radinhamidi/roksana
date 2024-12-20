"""
PageRankAttack Module
----------------------
This module implements the PageRankAttack class for adversarial attacks on graphs by selectively removing edges
connected to nodes based on their PageRank scores.

Classes:
    - PageRankAttack: A class to perform PageRank-based edge removal attacks on a graph dataset.
"""

from .base_attack import BaseAttack
import torch
from torch_geometric.utils import remove_self_loops, to_undirected, to_networkx
from networkx import pagerank
from typing import Any, List, Tuple


class PageRankAttack(BaseAttack):
    """
    PageRankAttack Class
    ---------------------
    Implements an adversarial attack that removes edges connected to nodes based on their PageRank scores.

    Attributes:
        data (Any): The graph dataset.
        params (dict): Additional parameters for the attack.
    """

    def __init__(self, data: Any, **kwargs):
        """
        Initialize the PageRankAttack method.

        Args:
            data (Any): The graph dataset.
            **kwargs: Additional parameters for the attack.
        """
        self.data = data
        self.params = kwargs
        self.device = kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


    def attack(self, data: Any, selected_nodes: torch.Tensor) -> Tuple[Any, List[Tuple[int, int]]]:
        """
        Perform the PageRank-based attack on the graph dataset.

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

        # Convert to a NetworkX graph for PageRank computation
        G = to_networkx(original_data.to('cpu'), to_undirected=True)  # NetworkX operates on CPU
        pagerank_scores = pagerank(G)

        # Ensure selected_nodes is a 1D tensor
        if isinstance(selected_nodes, torch.Tensor) and selected_nodes.ndimension() == 0:
            selected_nodes = selected_nodes.unsqueeze(0)

        selected_nodes = selected_nodes.to(self.device)

        removed_edges = []  # List to store removed edges

        # Track the number of edges removed to ensure only len(selected_nodes) edges are removed
        removed_edges_count = 0

        # Loop over each node in selected_nodes to remove one edge connected to the highest PageRank node
        for node in selected_nodes.tolist():
            if removed_edges_count >= len(selected_nodes):
                break  # Stop once len(selected_nodes) edges are removed

            # Find edges associated with the current node
            node_edges = [(i, (src, dst)) for i, (src, dst) in enumerate(edges) if src == node or dst == node]

            if node_edges:
                # Find the edge connected to the highest PageRank node
                edge_to_remove = max(
                    node_edges,
                    key=lambda x: pagerank_scores[x[1][1]] if x[1][0] == node else pagerank_scores[x[1][0]]
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
