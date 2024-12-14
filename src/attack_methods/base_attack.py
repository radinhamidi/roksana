"""
BaseAttack Module
------------------
This module provides an abstract base class for implementing graph attack methods. All attack methods must
inherit from `BaseAttack` and implement the required methods.

Classes:
    - BaseAttack: Abstract base class for all attack methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAttack(ABC):
    """
    Abstract BaseAttack Class
    --------------------------
    Defines the interface for attack methods. All attack methods must inherit from this class and implement
    the `__init__` and `attack` methods.

    Attributes:
        data (Any): The graph dataset used by the attack method.
        params (dict): Additional parameters for the attack method.
    """

    @abstractmethod
    def __init__(self, data: Any, **kwargs):
        """
        Initialize the attack method with the given dataset.

        Args:
            data (Any): The graph dataset.
            **kwargs: Additional keyword arguments specific to the attack method.
        """
        pass

    @abstractmethod
    def attack(self, query_node: int, perturbations: int = 1) -> Dict[str, Any]:
        """
        Perform an attack on the specified query node.

        Args:
            query_node (int): Index of the node to attack.
            perturbations (int, optional): Number of perturbations to apply. Defaults to 1.

        Returns:
            Dict[str, Any]: A dictionary containing details of the attack, such as removed edges or modifications made.
        """
        pass
