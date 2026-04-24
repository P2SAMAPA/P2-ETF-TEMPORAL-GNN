"""
Temporal Graph Network using GCN + GRU.
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch_geometric.utils import to_dense_batch

class TemporalGNN(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, num_layers):
        super().__init__()
        self.recurrent = GConvGRU(node_feat_dim, hidden_dim, num_layers)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, graph_sequence):
        """
        graph_sequence: list of Data objects, each with x (num_nodes, feat) and edge_index.
        Returns: (seq_len, batch_size, num_nodes, 1) predictions for each time step.
        """
        outputs = []
        h = None
        for graph in graph_sequence:
            h = self.recurrent(graph.x, graph.edge_index, h)
            # h shape: (num_nodes, hidden_dim)
            out = self.linear(h)  # (num_nodes, 1)
            outputs.append(out.unsqueeze(0))  # (1, num_nodes, 1)
        return torch.cat(outputs, dim=0)  # (T, num_nodes, 1)

class TGNRunner:
    def __init__(self, node_feat_dim, hidden_dim=64, num_layers=2, lr=0.001, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = TemporalGNN(node_feat_dim, hidden_dim, num_layers)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_sequence(self, graphs, epochs=80):
        self.model.train()
        n_graphs = len(graphs)
        # Use a rolling window of training: we'll feed the whole sequence every epoch
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            preds = self.model(graphs)  # (T, N, 1)
            targets = torch.stack([g.y for g in graphs], dim=0)  # (T, N)
            loss = self.criterion(preds.squeeze(-1), targets)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")

    def predict_latest(self, graphs):
        """Predict next-day return using the most recent graph."""
        self.model.eval()
        with torch.no_grad():
            # Pass the last graph only (or sequence? better to pass last for inference)
            h = None
            # We need to run through the whole sequence to get the final hidden state
            for graph in graphs:
                h = self.model.recurrent(graph.x, graph.edge_index, h)
            # h is the final state after the last graph
            out = self.model.linear(h)  # (N, 1)
        return out.squeeze(-1).numpy()
