"""
Attack Methods Registry
------------------------
This module provides a registry for managing attack methods. It includes predefined attack methods and utilities
to register custom attack methods dynamically.

Classes:
    - DegreeAttack: Implements degree-based edge removal.
    - PageRankAttack: Implements PageRank-based edge removal.
    - RandomAttack: Implements random edge removal.
    - VikingAttack: Implements Viking perturbation attack.

Functions:
    - register_attack_method: Decorator to register a custom attack method.
    - get_attack_method: Retrieve an instance of a registered attack method.
"""

from typing import Type, Dict, Any
from .base_attack import BaseAttack
from .degree import DegreeAttack
from .pagerank import PageRankAttack
from .random import RandomAttack
from .viking import VikingAttack

# Registry dictionary to hold attack method classes
ATTACK_METHODS: Dict[str, Type[BaseAttack]] = {
    'degree': DegreeAttack,
    'pagerank': PageRankAttack,
    'random': RandomAttack,
    'viking': VikingAttack,
    # Add more predefined attacks here
}

def register_attack_method(name: str):
    """
    Decorator to register an attack method class with a given name.

    Args:
        name (str): The name to register the attack method under.

    Returns:
        Callable: A decorator function that registers the attack method.
    """
    def decorator(cls: Type[BaseAttack]):
        if not issubclass(cls, BaseAttack):
            raise ValueError("Can only register subclasses of BaseAttack")
        ATTACK_METHODS[name.lower()] = cls
        return cls
    return decorator

def get_attack_method(name: str, data: Any, **kwargs) -> BaseAttack:
    """
    Retrieve an instance of the specified attack method.

    Args:
        name (str): Name of the attack method (e.g., 'degree', 'pagerank', 'random', 'viking').
        data (Any): The graph dataset.
        **kwargs: Additional keyword arguments for initializing the attack method.

    Returns:
        BaseAttack: An instance of the requested attack method.

    Raises:
        ValueError: If the specified attack method is not registered.

    Example:
        >>> from roksana.attack_methods.registry import get_attack_method
        >>> attack = get_attack_method('degree', data=my_graph, param1=value1)
    """
    name = name.lower()
    if name not in ATTACK_METHODS:
        raise ValueError(f"Attack method '{name}' not found. Available methods: {list(ATTACK_METHODS.keys())}")
    attack_method_class = ATTACK_METHODS[name]
    return attack_method_class(data, **kwargs)
