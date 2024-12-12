from .base_attack import BaseAttack
import torch
from torch_geometric.utils import remove_self_loops, to_undirected


class viking(BaseAttack):
    def __init__(self):
        super().__init__()

    def perturbation_attack(self, data, selected_nodes):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_removals = len(selected_nodes)

        # Move edge_index to GPU and filter edges
        edge_index = data.edge_index.to(device)

        # Mask edges involving selected nodes
        mask = torch.isin(edge_index[0], selected_nodes) | torch.isin(edge_index[1], selected_nodes)
        selected_edges = edge_index[:, mask]

        # Ignore nodes with no edges
        if selected_edges.size(1) == 0:
            return edge_index, torch.empty((2, 0), dtype=torch.long, device=device)  # No changes

        # Ensure enough edges are available for removal
        if selected_edges.size(1) < n_removals:
            raise ValueError("Not enough edges to remove the specified number.")

        # Shuffle selected edges and remove exactly n_removals edges
        selected_edges = selected_edges.T[torch.randperm(selected_edges.size(1))].T
        edges_to_remove = selected_edges[:, :n_removals]

        # Filter out edges to be removed
        edges_to_remove_set = set(map(tuple, edges_to_remove.cpu().numpy().T))
        retained_edges = torch.tensor(
            [tuple(edge) for edge in edge_index.T.cpu().numpy() if tuple(edge) not in edges_to_remove_set],
            dtype=torch.long,
            device=device,
        ).T

        return retained_edges, edges_to_remove

    def execute(self, data, selected_nodes, params):
        # Extract parameters
        selected_nodes = params.get('selected_nodes', None)
        if selected_nodes is None:
            raise ValueError("Parameter 'selected_nodes' is required.")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Normalize selected_nodes to be a 1D tensor
        if isinstance(selected_nodes, torch.Tensor) and selected_nodes.ndimension() == 0:
            selected_nodes = selected_nodes.unsqueeze(0)

        selected_nodes = selected_nodes.to(device)

        # Clone the original data safely
        data_copy = data.clone()
        data_instance = data.clone()

        # Run the perturbation attack
        viking_edge_index, removed_edges = self.perturbation_attack(data_instance, selected_nodes)

        # Create a modified dataset with the new edge index
        viking_dataset = data_instance.clone()
        viking_dataset.edge_index = viking_edge_index

        # Restore the original data to its initial state
        data = data_copy.clone()
        updated_data = viking_dataset

        return updated_data, removed_edges
