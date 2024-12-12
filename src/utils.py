import torch

def compare_original_vs_updated(original_data, updated_data):
    original_num_nodes = original_data.num_nodes
    updated_num_nodes = updated_data.num_nodes

    original_num_edges = original_data.edge_index.size(1)
    updated_num_edges = updated_data.edge_index.size(1)
    print("____________________________________________")
    print("- Compare")
    print("____________________________________________")
    print("Number of nodes (Original):", original_num_nodes)
    print("Number of nodes (Updated):", updated_num_nodes)
    print("Number of edges (Original):", original_num_edges)
    print("Number of edges (Updated):", updated_num_edges)
    print("\nDifference in the number of edges:", original_num_edges - updated_num_edges)
    print("____________________________________________")

    # Find removed edges
    original_edges = original_data.edge_index.t().tolist()
    updated_edges = updated_data.edge_index.t().tolist()

    # Convert to sets for comparison
    original_edges_set = set(map(tuple, original_edges))
    updated_edges_set = set(map(tuple, updated_edges))

    removed_edges = original_edges_set - updated_edges_set

    #if removed_edges:
    #    print("\nRemoved edges:")
    #    for edge in removed_edges:
    #        print(edge)
    #else:
    #    print("\nNo edges have been removed.")

    print("____________________________________________")


def remove_edges(data, edges_to_remove):
    mask = torch.ones(data.edge_index.size(1), dtype=torch.bool, device=data.edge_index.device)
    for edge in edges_to_remove:
        u, v = edge
        
        # Check if the edge exists in either direction
        edge_exists = (
            ((data.edge_index[0] == u) & (data.edge_index[1] == v)).any() or
            ((data.edge_index[0] == v) & (data.edge_index[1] == u)).any()
        )
        
        if edge_exists:
            # Remove both directions of the edge
            mask &= ~(
                ((data.edge_index[0] == u) & (data.edge_index[1] == v)) |
                ((data.edge_index[0] == v) & (data.edge_index[1] == u))
            )
        else:
            print(f"Edge ({u}, {v}) does not exist, skipping.")
    
    # Update edge_index with the remaining edges
    data.edge_index = data.edge_index[:, mask]
    return data
