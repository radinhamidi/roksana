"""
Graph Adversarial Attack Evaluation
-----------------------------------
This script evaluates the performance of a Graph search before and after applying adversarial attacks using the Attack method on a dataset.

Steps:
1. Load the Cora dataset.
2. Prepare a search set with queries.
3. Initialize the GCN search method and attack method.
4. Perform adversarial attacks on selected nodes.
5. Evaluate the performance of GCN before and after the attack.
6. Save evaluation results to CSV, JSON, and Pickle formats.
"""
 
import sys
import os
import torch
from roksana.src.search_methods.search_methods import get_search_method
from roksana.src.datasets import load_dataset, prepare_search_set
from roksana.src.evaluation import Evaluator, save_results_to_pickle, save_results_to_json
from roksana.src.attack_methods import get_attack_method
from roksana.src.utils import *


def main():
    """
    Main function to execute the adversarial attack evaluation workflow.

    Steps:
    - Load the dataset.
    - Prepare a search set for evaluation.
    - Initialize the GCN search method and VikingAttack.
    - Perform attacks on query nodes and apply the changes to the dataset.
    - Evaluate GCN performance before and after the attack.
    - Save evaluation results in multiple formats.
    """
    # Load the Cora dataset
    dataset = load_dataset(dataset_name='cora', root='data/')
    data = dataset[0]

    # Prepare the test set with 10% of nodes as queries
    queries, query_features, gold_sets = prepare_search_set(data, percentage=0.1, seed=123)
    
    # Initialize the GCN search method before attack
    gcn_before = get_search_method('gcn', data=data, hidden_channels=64, epochs=200, lr=0.01)

    # Initialize the attack method
    attack_method = get_attack_method('random', data=data, perturbations=2)
    attacked_data = data.clone()

    # Perform attacks on all query nodes
    removed_edges_list = []

    for query in queries:
        tensor_query = torch.tensor([[query]], device='cuda:0')
        tensor_query = tensor_query.item()
        tensor_query = torch.tensor(tensor_query, device='cuda:0')
        updated_data, removed_edge = attack_method.attack(data=data, selected_nodes=tensor_query)
        removed_edges_list.append(removed_edge)
        print(f"Attack on Node: {query}. Remove edges:{removed_edge}")
    
    removed_edges_list_stat(removed_edges_list)

    # Apply all removed edges to the final data 
    for edge in removed_edges_list:
        attacked_data = remove_edges(attacked_data, edge)

    compare_original_vs_updated(data, attacked_data)

    # Initialize the GCN search method after attack
    gcn_after = get_search_method('gcn', data=attacked_data, hidden_channels=64, epochs=200, lr=0.01)

    # Initialize the Evaluator
    evaluator = Evaluator(
        search_method_before=gcn_before,
        search_method_after=gcn_after,
        k_values=[5, 10, 20]
    )

    # Perform evaluation and save results to CSV
    evaluator.evaluate(
        queries=queries,
        gold_sets=gold_sets,
        results_dir='evaluation_results',
        filename='gcn_attack_evaluation.csv'
    )

    # Optionally, save results to JSON or Pickle
    results = evaluator.get_all_results()  # Assuming you have a method to retrieve all results
    save_results_to_json(results, 'evaluation_results/gcn_attack_evaluation.json')
    save_results_to_pickle(results, 'evaluation_results/gcn_attack_evaluation.pkl')

if __name__ == '__main__':
    """
    Entry point for the script. Executes the `main` function.
    """
    main()




