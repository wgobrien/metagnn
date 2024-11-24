# encoder.py

import torch
from torch_geometric.nn import GCNConv, SAGPooling, global_mean_pool
import pyro

class GraphEncoder(pyro.nn.PyroModule):
    def __init__(
        self, 
         input_dim: int, 
         hidden_dim: int, 
         latent_dim: int, 
         num_layers: int, 
         edge_attr_dim: int,
    ):
        super().__init__()
        
        self.convs = torch.nn.ModuleList([
            GCNConv(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
            )
            for i in range(num_layers)
        ])
        
        self.skip_connections = torch.nn.ModuleList()
        for i in range(0, num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.skip_connections.append(
                torch.nn.Linear(in_dim, hidden_dim)
            )

        self.norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.pool = SAGPooling(hidden_dim)
        
        self.fc_mu = torch.nn.Linear(hidden_dim, latent_dim)
        self.fc_var = torch.nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, edge_weight, batch):
        for i, conv in enumerate(self.convs):
            x_skip = self.skip_connections[i](x)
            x = conv(x, edge_index, edge_weight) + x_skip
            x = self.norms[i](x)
            x = torch.relu(x)

        x, edge_index, edge_weight, batch, perm, score = self.pool(
            x, edge_index, edge_attr=edge_weight, batch=batch
        )
        x = global_mean_pool(x, batch)
        
        return self.fc_mu(x), torch.exp(self.fc_var(x))
