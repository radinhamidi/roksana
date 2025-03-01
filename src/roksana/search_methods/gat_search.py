# roksana/search_methods/gat_search.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch.optim as optim
from typing import Any, List
from .base_search import SearchMethod

class GATNet(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 8):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Obtain node embeddings from the model.

        Args:
            x (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            torch.Tensor: Node embeddings.
        """
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x

class GATSearch(SearchMethod):
    """
    Search method using Graph Attention Networks (GAT).
    """

    def __init__(
        self,
        data: Any,
        device: str = None,
        hidden_channels: int = 64,
        heads: int = 8,
        epochs: int = 200,
        lr: float = 0.005
    ):
        """
        Initialize and train the GAT model.

        Args:
            data (Any): The graph dataset.
            device (str, optional): Device to run the computations on ('cpu' or 'cuda').
            hidden_channels (int, optional): Number of hidden channels in GAT layers.
            heads (int, optional): Number of attention heads in GAT layers.
            epochs (int, optional): Number of training epochs.
            lr (float, optional): Learning rate for the optimizer.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = data.to(self.device)
        self.model = GATNet(
            in_channels=self.data.num_features,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            heads=heads
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs

        # Train the model
        self.train_model()

        # Generate embeddings
        self.embeddings = self.get_node_embeddings()

    def train_model(self):
        """
        Train the GAT model on the dataset.
        Assumes that the dataset has a 'y' attribute for node labels.
        """
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            self.optimizer.zero_grad()
            out = self.model(self.data.x, self.data.edge_index)
            if hasattr(self.data, 'y') and self.data.y is not None:
                loss = F.cross_entropy(out[self.data.train_mask], self.data.y[self.data.train_mask])
                loss.backward()
                self.optimizer.step()
                if epoch % 10 == 0 or epoch == 1:
                    acc = self.evaluate()
                    print(f"GAT Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {acc:.4f}")
            else:
                # If no labels are provided, use a dummy loss (e.g., reconstruction loss)
                loss = torch.tensor(0.0, requires_grad=True).to(self.device)
                loss.backward()
                self.optimizer.step()

    def evaluate(self) -> float:
        """
        Evaluate the model's accuracy on the training set.

        Returns:
            float: Training accuracy.
        """
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            pred = out.argmax(dim=1)
            correct = (pred[self.data.train_mask] == self.data.y[self.data.train_mask]).sum()
            acc = int(correct) / int(self.data.train_mask.sum())
        return acc

    def get_node_embeddings(self) -> torch.Tensor:
        """
        Generate node embeddings by passing the data through the model.

        Returns:
            torch.Tensor: Node embeddings.
        """
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.get_embeddings(self.data.x, self.data.edge_index)
        return embeddings.cpu()

    def search(self, query_features: torch.Tensor, top_k: int = 10) -> List[int]:
        """
        Perform a search with the given query features using GAT embeddings.

        Args:
            query_features (torch.Tensor): Feature vector of the query node.
            top_k (int, optional): Number of top similar nodes to retrieve.

        Returns:
            List[int]: List of node indices sorted by similarity to the query.
        """
        self.model.eval()
        query_features = query_features.to(self.device)

        # Ensure query_features is 2D
        if query_features.dim() == 1:
            query_features = query_features.unsqueeze(0)  # Shape: [1, feature_dim]

        with torch.no_grad():
            # Append query features to the node features
            x = torch.cat([self.data.x, query_features], dim=0)
            edge_index = self.data.edge_index
            # Get embeddings for all nodes including the query nodes
            embeddings = self.model.get_embeddings(x, edge_index)
            # Number of queries
            num_queries = query_features.size(0)
            # Extract the query embeddings (last num_queries nodes)
            query_embeddings = embeddings[-num_queries:]  # Shape: [num_queries, embedding_dim]
            # Embeddings for all other nodes
            node_embeddings = embeddings[:-num_queries]  # Shape: [num_nodes, embedding_dim]

        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        node_embeddings = F.normalize(node_embeddings, p=2, dim=1)

        # Compute cosine similarity
        similarities = torch.mm(query_embeddings, node_embeddings.t())  # Shape: [num_queries, num_nodes]

        # Get top_k indices for each query
        top_k_indices = similarities.topk(top_k, dim=1).indices.tolist()

        return top_k_indices