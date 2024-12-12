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