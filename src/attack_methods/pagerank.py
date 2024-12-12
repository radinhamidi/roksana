from .base_attack import BaseAttack
import torch
from torch_geometric.utils import remove_self_loops, to_undirected
from networkx import pagerank
from torch_geometric.utils import to_networkx

class pagerank(BaseAttack):
    def execute(self, data, selected_node, params):
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        original_random_dataset = data.to(device)
        edge_index = original_random_dataset.edge_index.clone().to(device)
        edges = edge_index.t().tolist()

        G = to_networkx(original_random_dataset.to('cpu'), to_undirected=True)  # NetworkX doesn't support GPU
        pagerank_scores = pagerank(G)

        # Normalize selected_nodes to be a 1D tensor, even if a single node is passed
        if isinstance(selected_nodes, torch.Tensor) and selected_nodes.ndimension() == 0:
            selected_nodes = selected_nodes.unsqueeze(0)

        selected_nodes = selected_nodes.to(device)

        removed_edges = []  # To store the removed edges

        # Loop over each node in selected_nodes to remove one edge connected to the highest PageRank node
        for node in selected_nodes.tolist():
            # Find edges associated with the current node
            node_edges = [(i, (src, dst)) for i, (src, dst) in enumerate(edges) if src == node or dst == node]

            if node_edges:
                # Find the edge connected to the highest PageRank node
                edge_to_remove = max(node_edges, key=lambda x: pagerank_scores[x[1][1]] if x[1][0] == node else pagerank_scores[x[1][0]])
                edges.pop(edge_to_remove[0])  # Remove the edge from the edge list
                removed_edges.append(edge_to_remove[1])  # Store the removed edge

        # Create a new edge_index tensor from the modified edges
        new_edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()

        # Remove any self-loops, just in case
        new_edge_index, _ = remove_self_loops(new_edge_index)

        # Assign the modified edge_index to create the random_dataset
        page_dataset = data.clone().to(device)
        page_dataset.edge_index = new_edge_index
        updated_data = page_dataset

        # Return both the updated data and the list of removed edges
        return updated_data, removed_edges
    
        
