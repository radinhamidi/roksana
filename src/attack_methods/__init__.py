# roksana/attack_methods/__init__.py

from .base_attack import BaseAttack
from .registry import get_attack_method, ATTACK_METHODS
from .degree import DegreeAttack
from .pagerank import PageRankAttack
from .random import RandomAttack
from .viking import VikingAttack


__all__ = [
    'AttackMethod',
    'get_attack_method',
    'ATTACK_METHODS',
    'degree',
    'pagerank',
    'random',
    'viking',
]