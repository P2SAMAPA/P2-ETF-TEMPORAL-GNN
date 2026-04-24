"""
Temporal Graph Attention Network (TGAT) using TransformerConv + GRU.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv

class GATConvGRUManual(nn.Module):
    """Graph Attention followed by GRU for temporal graphs."""
    def __init__(self, in_channels, hidden_channels, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_channels = hidden_channels

        # Use TransformerConv as attention layer
        self.gat_z = TransformerConv(in_channels + hidden_channels, hidden_channels // num_heads,
                                     heads=num_heads, dropout=dropout, concat=True)
        self.gat_r = TransformerConv(in_channels + hidden_channels, hidden_channels // num_heads,
                                     heads=num_heads, dropout=dropout, concat=True)
        self.gat_h = TransformerConv(in_channels + hidden_channels, hidden_channels // num_heads,
                                     heads=num_heads, dropout=dropout, concat=True)

    def forward(self, x, edge_index, h=None):
        # x: (N, in_channels), h: (N, hidden_channels) or None
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_channels, device=x.device)
        h_in = torch.cat([x, h], dim=-1)

        z = torch.sigmoid(self.gat_z(h_in, edge_index))
        r = torch.sigmoid(self.gat_r(h_in, edge_index))

        h_cand = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.gat_h(h_cand, edge_index))

        new_h = z * h + (1 - z) * h_tilde
        return new_h

class TGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.recurrent = GATConvGRUManual(in_channels, hidden_channels, num_heads, dropout)
        self.linear = nn.Linear(hidden_channels, 1)

    def forward(self, graph_sequence):
        outputs = []
        h = None
        for graph in graph_sequence:
            h = self.recurrent(graph.x, graph.edge_index, h)
            out = self.linear(h).unsqueeze(0)  # (1, N, 1)
            outputs.append(out)
        return torch.cat(outputs, dim=0)  # (T, N, 1)

class TGATRunner:
    def __init__(self, node_feat_dim, hidden_dim=64, num_heads=4, dropout=0.1, lr=0.001, seed=42):
        torch.manual_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TGAT(node_feat_dim, hidden_dim, num_heads, 1, dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_sequence(self, graphs, epochs=80):
        self.model.train()
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
                print(f"    TGAT Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")

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
