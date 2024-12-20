"""
Attack Methods Package
-----------------------
This package provides various attack methods for adversarial modifications of graph datasets. It includes both
predefined attack methods and utilities for registering and retrieving custom attacks.

Modules:
    - base_attack: Defines the abstract base class for all attack methods.
    - registry: Manages registration and retrieval of attack methods.
    - degree: Implements degree-based edge removal.
    - pagerank: Implements PageRank-based edge removal.
    - random: Implements random edge removal.
    - viking: Implements Viking perturbation attack.

Attributes:
    - BaseAttack: Abstract base class for all attack methods.
    - get_attack_method: Retrieve a registered attack method by name.
    - ATTACK_METHODS: Dictionary of registered attack methods.
    - DegreeAttack: Implements degree-based attack logic.
    - PageRankAttack: Implements PageRank-based attack logic.
    - RandomAttack: Implements random attack logic.
    - VikingAttack: Implements Viking perturbation attack logic.
"""

from .base_attack import BaseAttack
from .registry import get_attack_method, ATTACK_METHODS
from .degree import DegreeAttack
from .pagerank import PageRankAttack
from .random import RandomAttack
from .viking import VikingAttack


__all__ = [
    'BaseAttack',
    'get_attack_method',
    'ATTACK_METHODS',
    'DegreeAttack',
    'PageRankAttack',
    'RandomAttack',
    'VikingAttack',
]