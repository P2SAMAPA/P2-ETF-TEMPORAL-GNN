"""
Temporal Graph Network – GCN + GRU built from torch and torch_geometric.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GConvGRUManual(nn.Module):
    """GCN followed by GRU for temporal graphs, implemented manually."""
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        # GCN layers for updating candidate state
        self.gcn_z = GCNConv(in_channels + hidden_channels, hidden_channels)
        self.gcn_r = GCNConv(in_channels + hidden_channels, hidden_channels)
        self.gcn_h = GCNConv(in_channels + hidden_channels, hidden_channels)

    def forward(self, x, edge_index, h=None):
        # x: (num_nodes, in_channels)
        # h: (num_nodes, hidden_channels) or None
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_channels, device=x.device)
        h_in = torch.cat([x, h], dim=-1)

        z = torch.sigmoid(self.gcn_z(h_in, edge_index))
        r = torch.sigmoid(self.gcn_r(h_in, edge_index))

        h_cand = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.gcn_h(h_cand, edge_index))

        new_h = z * h + (1 - z) * h_tilde
        return new_h

class TemporalGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.recurrent = GConvGRUManual(in_channels, hidden_channels, num_layers)
        self.linear = nn.Linear(hidden_channels, 1)

    def forward(self, graph_sequence):
        outputs = []
        h = None
        for graph in graph_sequence:
            h = self.recurrent(graph.x, graph.edge_index, h)
            out = self.linear(h)  # (N, 1)
            outputs.append(out.unsqueeze(0))  # (1, N, 1)
        return torch.cat(outputs, dim=0)  # (T, N, 1)

class TGNRunner:
    def __init__(self, node_feat_dim, hidden_dim=64, num_layers=2, lr=0.001, seed=42):
        torch.manual_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TemporalGNN(node_feat_dim, hidden_dim, num_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_sequence(self, graphs, epochs=80):
        self.model.train()
        # Move all graphs to device
        for g in graphs:
            g.x = g.x.to(self.device)
            g.edge_index = g.edge_index.to(self.device)
            g.y = g.y.to(self.device)
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
        self.model.eval()
        for g in graphs:
            g.x = g.x.to(self.device)
            g.edge_index = g.edge_index.to(self.device)
        with torch.no_grad():
            h = None
            for g in graphs:
                h = self.model.recurrent(g.x, g.edge_index, h)
            out = self.model.linear(h)
        return out.cpu().squeeze(-1).numpy()
