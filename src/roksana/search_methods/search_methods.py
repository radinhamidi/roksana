# roksana/search_methods.py

from abc import ABC, abstractmethod
from typing import List, Optional
import torch
from torch_geometric.data import Data


# Define the abstract base class for search methods
class SearchMethod(ABC):
    """
    Abstract base class for search methods.
    """

    @abstractmethod
    def __init__(self, data: Data, device: Optional[str] = None):
        """
        Initialize the search method with the given dataset.

        Args:
            data (Data): The graph dataset.
            device (str, optional): Device to run the computations on ('cpu' or 'cuda').
        """
        pass

    @abstractmethod
    def search(self, query_features: torch.Tensor, top_k: int = 10) -> List[int]:
        """
        Perform a search with the given query features.

        Args:
            query_features (torch.Tensor): Feature vector of the query node.
            top_k (int, optional): Number of top similar nodes to retrieve.

        Returns:
            List[int]: List of node indices sorted by similarity to the query.
        """
        pass

