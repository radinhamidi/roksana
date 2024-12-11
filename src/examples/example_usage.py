# examples/example_evaluation.py

from roksana.datasets import load_dataset, prepare_search_set
from roksana.search_methods import get_search_method
from roksana.attack_methods import get_attack_method
from roksana.evaluation import Evaluator, save_results_to_pickle, save_results_to_json

def main():
    # Load the Cora dataset
    dataset = load_dataset(dataset_name='cora', root='data/')
    data = dataset[0]

    # Prepare the test set with 10% of nodes as queries
    queries, gold_sets = prepare_search_set(data, percentage=0.1, seed=123)

    # Initialize the GCN search method before attack
    gcn_before = get_search_method('gcn', data=data, hidden_channels=64, epochs=200, lr=0.01)

    # Initialize the attack method
    attack_method = get_attack_method('predefined_attack1', data=data, perturbations=2)

    # Perform attacks on all query nodes
    for query in queries:
        attack_details = attack_method.attack(query_node=query, perturbations=2)
        print(f"Attack on Node {query}: {attack_details}")

    # Initialize the GCN search method after attack
    gcn_after = get_search_method('gcn', data=data, hidden_channels=64, epochs=200, lr=0.01)

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
    main()