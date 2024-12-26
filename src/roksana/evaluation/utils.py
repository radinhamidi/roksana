# roksana/evaluation/utils.py

import json
import csv
import pickle
import torch
from typing import Dict, Any, List

def tensor_to_serializable(obj: Any) -> Any:
    """
    Recursively convert Tensor objects in a data structure to serializable types.

    Args:
        obj: The data structure containing potentially Tensors.

    Returns:
        The data structure with Tensors converted to lists or scalars.
    """
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        else:
            return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(tensor_to_serializable(v) for v in obj)
    else:
        return obj

def save_results_to_json(results: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save evaluation results to a JSON file.

    Args:
        results (List[Dict[str, Any]]): List of evaluation result dictionaries.
        filepath (str): Path to the JSON file where results will be saved.
    """
    # Convert Tensors to serializable types
    serializable_results = tensor_to_serializable(results)

    with open(filepath, 'w') as json_file:
        json.dump(serializable_results, json_file, indent=4)
    print(f"Results successfully saved to {filepath} in JSON format.")

def save_results_to_csv(results: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save evaluation results to a CSV file.

    Args:
        results (List[Dict[str, Any]]): List of evaluation result dictionaries.
        filepath (str): Path to the CSV file where results will be saved.
    """
    if not results:
        print("No results to save.")
        return

    # Extract header from the first result dictionary
    header = results[0].keys()

    with open(filepath, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results successfully saved to {filepath} in CSV format.")

def save_results_to_pickle(results: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save evaluation results to a Pickle file.

    Args:
        results (List[Dict[str, Any]]): List of evaluation result dictionaries.
        filepath (str): Path to the Pickle file where results will be saved.
    """
    with open(filepath, 'wb') as pickle_file:
        pickle.dump(results, pickle_file)
    print(f"Results successfully saved to {filepath} in Pickle format.")