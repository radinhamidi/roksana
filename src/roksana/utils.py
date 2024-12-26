
import torch
from collections import Counter
from collections.abc import Sequence


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


def remove_edges(data, edges_to_remove, inplace=False):
    """
    Remove specified edges from the given undirected graph data object.

    This function modifies the `data.edge_index` attribute by removing the 
    specified edges. It can handle either a single list of edges or a list 
    of lists of edges. Since the graph is undirected, (u, v) and (v, u) 
    are considered the same edge.

    Args:
        data: A PyG Data object with an `edge_index` attribute.
        edges_to_remove (List[Tuple[int, int]] or List[List[Tuple[int, int]]]):
            A collection of edges to remove. For example:
            - [(u1, v1), (u2, v2), ...]
            - [[(u1, v1), (u2, v2)], [(u3, v3), ...]]
        inplace (bool, optional): If True, modifies the input data object in-place. Default is False.

    Returns:
        The modified data object with the specified edges removed.
    """

    if not inplace:
        data = data.clone()
    
    # If edges_to_remove is a list of lists, flatten it
    if any(isinstance(item, list) for item in edges_to_remove):
        edges_to_remove = [edge for sublist in edges_to_remove for edge in sublist]

    # Normalize edges so that (u, v) and (v, u) represent the same edge
    edges_to_remove = [tuple(sorted(edge)) for edge in edges_to_remove]

    # Remove duplicates by converting to a set
    edges_to_remove = set(edges_to_remove)

    # Convert data.edge_index into a list of edges
    edge_list = [(int(u.item()), int(v.item())) for u, v in zip(data.edge_index[0], data.edge_index[1])]

    # Filter the edges: keep only those not in edges_to_remove
    # Since graph is undirected, we sort each edge before checking
    filtered_edges = [e for e in edge_list if tuple(sorted(e)) not in edges_to_remove]

    # Convert filtered edges back to a tensor
    if len(filtered_edges) > 0:
        filtered_edges_tensor = torch.tensor(filtered_edges, dtype=torch.long, device=data.edge_index.device).t()
    else:
        # If no edges remain, create an empty edge_index
        filtered_edges_tensor = torch.empty((2, 0), dtype=torch.long, device=data.edge_index.device)

    data.edge_index = filtered_edges_tensor
    
    return data



def removed_edges_list_stat(data, removed_edges_list, verbose=True):
    """
    Calculate and report statistics about a list of removed edges, including 
    checking if the reverse direction of these edges exists in the main graph.

    This function takes a list of edge lists that represent edges removed during 
    multiple perturbation operations and aggregates them to determine:

    - The total number of removed edges across all operations.
    - The number of duplicate edges (edges that have appeared more than once across all operations).
    - The number of unique edges that have been removed overall.
    - The number of removed edges including their reversed counterpart present 
      in the main graph (`data.edge_index`).

    If `verbose` is True, it prints these statistics and does not return anything.
    If `verbose` is Flase, it returns the statistics as a tuple.

    Args:
        removed_edges_list (List[List[Tuple[int, int]]]): A list of lists, 
            where each inner list contains tuples representing edges that 
            were removed in a particular operation.
        data: A PyG Data object that contains the main graph edges in `data.edge_index`.
        verbose (bool, optional): If True, prints out the statistics. Defaults to True.

    Returns:
        Tuple[int, int, int, int]:
            A tuple containing:
                - int: The total number of removed edges.
                - int: The number of duplicate edges across all operations.
                - int: The number of unique edges removed overall.
                - int: The number of removed edges including their reversed counterpart 
                       in the main graph.
    """


    # Flatten the list if it is a list of lists
    if any(isinstance(item, list) for item in removed_edges_list):
        removed_edges_list = [edge for sublist in removed_edges_list for edge in sublist]

    # Normalize edges so that (u, v) and (v, u) are treated the same
    # for counting unique edges and duplicates.
    # This ensures (u, v) and (v, u) become the same tuple (min(u,v), max(u,v)).
    removed_edges_list = [tuple(sorted(edge)) for edge in removed_edges_list]

    removed_edges_list_num_edges = len(removed_edges_list)
    removed_edged_unique = set(removed_edges_list)
    removed_edges_unique_num_edges = len(removed_edged_unique)
    duplicates = removed_edges_list_num_edges - removed_edges_unique_num_edges

    # Create a set of edges from the main graph
    # For checking reverse counterparts, we do NOT normalize here, because 
    # we want to distinguish (u, v) from (v, u).
    graph_edges = {(int(u.item()), int(v.item())) for u, v in zip(data.edge_index[0], data.edge_index[1])}

    # Count how many removed edges have a reversed counterpart in the main graph.
    # We'll use the removed_edged_unique here because we want to check 
    # the actual directionality of how they were removed.
    reversed_counterparts_count = 0
    for (u, v) in removed_edged_unique:
        # The reversed edge is (v, u)
        if (v, u) in graph_edges:
            reversed_counterparts_count += 1

    if verbose:
        print(f'Number of Removed Edges in the list: {removed_edges_list_num_edges}')
        print(f'Number of Duplicate Edges: {duplicates}')
        print(f'Number of Unique Edges to Remove: {removed_edges_unique_num_edges}')
        print(f'Number of Removed Edges with Reversed Counterparts in Graph: {reversed_counterparts_count}')
        print(f'Number of Total Removed Edges: {removed_edges_unique_num_edges + reversed_counterparts_count}')

    else:
        return removed_edges_list_num_edges, duplicates, removed_edges_unique_num_edges, reversed_counterparts_count, removed_edges_unique_num_edges + reversed_counterparts_count

    