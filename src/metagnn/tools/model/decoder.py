# decoder.py

import torch
from torch_geometric.data import Data, Batch
import pyro

class GraphDecoder(pyro.nn.PyroModule):
    def __init__(self, kmer_map, latent_dim, hidden_dim, num_layers):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.kmer_map = kmer_map
        self.num_nodes = len(kmer_map.keys())

        self.node_projection = torch.nn.Linear(latent_dim, self.num_nodes * self.hidden_dim)
        
        self.edge_layers = torch.nn.ModuleList()
        self.edge_skips = torch.nn.ModuleList()
        self.edge_norms = torch.nn.ModuleList()
        for i in range(num_layers):
            in_dim = self.hidden_dim * 2 if i == 0 else hidden_dim
            self.edge_layers.append(torch.nn.Linear(in_dim, hidden_dim))
            self.edge_skips.append(torch.nn.Linear(in_dim, hidden_dim))
            self.edge_norms.append(torch.nn.LayerNorm(hidden_dim))
        
        self.output_layer = torch.nn.Linear(hidden_dim, 1)

    def generate_successors(self, k_minus_1_mer):
        return [
            self.kmer_map[k_minus_1_mer[1:] + n] for n in 'ACGT'
        ]

    def decoded_batch(self, edge_indices, edge_probas, batch_size):
        data_list = []
        for b in range(batch_size):
            batch_edge_index = edge_indices.clone()
            batch_edge_probas = edge_probas[b]
            data = Data(
                edge_index=batch_edge_index,
                edge_proba=batch_edge_probas,
                num_nodes=self.num_nodes,
            )
            data_list.append(data)

        return Batch.from_data_list(data_list)
    
    def forward(self, z):
        batch_size, latent_dim = z.size()
        x_batch = self.node_projection(z).view(batch_size, self.num_nodes, self.hidden_dim)

        edge_indices = []
        for k_mer in self.kmer_map.keys():
            successors = self.generate_successors(k_mer)
            edge_indices.extend([[self.kmer_map[k_mer], v] for v in successors])
        edge_indices = torch.tensor(edge_indices, dtype=torch.long).T

        u, v = edge_indices

        u_embeddings = x_batch[:, u, :]  # [batch_size, num_nodes, hidden_dim]
        v_embeddings = x_batch[:, v, :]  # [batch_size, num_nodes, hidden_dim]
        h = torch.cat([u_embeddings, v_embeddings], dim=-1)  # [batch_size, num_edges, hidden_dim * 2]
        
        for layer, skip, norm in zip(self.edge_layers, self.edge_skips, self.edge_norms):
            h = torch.relu(norm(layer(h) + skip(h)))

        edge_probas = torch.sigmoid(self.output_layer(h).squeeze(-1))
        return self.decoded_batch(edge_indices, edge_probas, batch_size)
