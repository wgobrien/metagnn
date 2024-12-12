import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool, global_max_pool, SAGPooling
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Dict, Optional
import numpy as np
from debruijin import DeBruijnGraph
import pandas as pd
from umap.umap_ import UMAP  
import plotly.express as px
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class ViralGenomeEncoder(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 512,
        embedding_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        pool_ratio: float = 0.5  
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.node_encoder = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        self.convs.append(GATConv(
            hidden_dim, 
            hidden_dim // 8,
            heads=8, 
            dropout=dropout
        ))
        self.pools.append(SAGPooling(hidden_dim, ratio=pool_ratio))
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(
                hidden_dim,
                hidden_dim // 8,
                heads=8,
                dropout=dropout
            ))
            self.pools.append(SAGPooling(hidden_dim, ratio=pool_ratio))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.convs.append(GATConv(
            hidden_dim,
            hidden_dim // 8,
            heads=8,
            dropout=dropout
        ))
        self.pools.append(SAGPooling(hidden_dim, ratio=pool_ratio))
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),  # Concatenated pooled features
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
            
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        batch = data.batch if hasattr(data, 'batch') else None
        
        x = self.node_encoder(x)
        
        pooled_representations = []
        
        for conv, pool, norm in zip(self.convs, self.pools, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
            
            x, edge_index, _, batch, _, _ = pool(
                x, edge_index, None, batch
            )
            
            pooled = global_mean_pool(x, batch)
            pooled_representations.append(pooled)
        
        x = torch.cat(pooled_representations, dim=-1)
        
        x = self.projection(x)
        
        normalized_emb = F.normalize(x, p=2, dim=1)
        
        return normalized_emb

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, normalize_embeddings: bool = True):
        super().__init__()
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings
        self.mse = nn.MSELoss()
    
    def forward(self, embeddings: torch.Tensor, similarity_matrix: torch.Tensor) -> torch.Tensor:
        
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        distances = torch.cdist(embeddings, embeddings, p=2)
        
        distances_normalized = (distances - distances.min()) / (distances.max() - distances.min())
        
        target_distances = 1 - similarity_matrix
        
        loss = self.mse(distances_normalized, target_distances)
        
        return loss
import torch
import numpy as np
import pickle

def save_embeddings(embeddings: torch.Tensor, labels: list, filepath: str = "embeddings.pt"):
    
    embeddings_cpu = embeddings.cpu().detach()
    
    torch.save(embeddings_cpu, f"{filepath}_vectors.pt")
    
    with open(f"{filepath}_labels.pkl", "wb") as f:
        pickle.dump(labels, f)
    
    print(f"Saved embeddings to {filepath}_vectors.pt and {filepath}_labels.pkl!!")

def load_embeddings(filepath: str = "embeddings") -> tuple:
    
    embeddings = torch.load(f"{filepath}_vectors.pt")
    
    with open(f"{filepath}_labels.pkl", "rb") as f:
        labels = pickle.load(f)
    
    return embeddings, labels

def save_model(model, optimizer, epoch, loss, save_path="model_save"):
    "
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'model_config': {
            'num_node_features': model.node_encoder[0].in_features, 
            'hidden_dim': model.convs[0].in_channels,
            'embedding_dim': model.embedding_dim,  
            'num_layers': len(model.convs)
        }
    }
    
    torch.save(save_dict, f"{save_path}_negative.pt")

def load_model(load_path="model_save.pt", device=None):
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(load_path, map_location=device)
    
    model = ViralGenomeEncoder(
        num_node_features=checkpoint['model_config']['num_node_features'],
        hidden_dim=checkpoint['model_config']['hidden_dim'],
        embedding_dim=checkpoint['model_config']['embedding_dim'],
        num_layers=checkpoint['model_config']['num_layers']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']

def get_embeddings_with_labels(
    model: ViralGenomeEncoder,
    graphs: List[Data],
    labels: List[str],
    batch_size: int = 128,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[torch.Tensor, List[str]]:
    
    model = model.to(device)
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(graphs), batch_size):
            batch = Batch.from_data_list(
                graphs[i:i+batch_size]
            ).to(device)
            batch_embeddings = model(batch)
            embeddings.append(batch_embeddings.cpu())
    
    all_embeddings = torch.cat(embeddings, dim=0)
    
    return all_embeddings, labels


def read_fasta(file_path):
    """
    Reads a FASTA file and returns a list of sequences.

    :param file_path: Path to the FASTA file
    :return: List of sequences from the FASTA file
    """
    sequences = []
    
    try:
        with open(file_path, 'r') as file:
            sequence = ""
            for line in file:
                if line.startswith('>'):
                    if sequence:  
                        sequences.append(sequence)
                        sequence = ""
                else:
                    sequence += line.strip()
            if sequence:
                sequences.append(sequence)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"Errorrrr")

    return sequences

import pickle

def save_graphs(graphs, filepath="processed_graphs.pkl"):
    with open(filepath, 'wb') as f:
        pickle.dump(graphs, f)

def load_graphs(filepath="processed_graphs.pkl"):
    with open(filepath, 'rb') as f:
        graphs = pickle.load(f)
    return graphs

class EmbeddingSimilarityMonitor:
    def __init__(self, log_freq: int = 10):
        self.log_freq = log_freq
        self.similarity_history = []
        self.epoch_metrics = defaultdict(list)
        
    def compute_similarity_stats(self, embeddings: torch.Tensor, similarity_matrix: torch.Tensor) -> Dict[str, float]:
        embeddings_normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        cos_sim = torch.matmul(embeddings_normalized, embeddings_normalized.t())
        
        mask = ~torch.eye(cos_sim.size(0), dtype=torch.bool, device=cos_sim.device)
        embedding_similarities = cos_sim[mask].cpu().detach().numpy()
        
        sim_matrix_values = similarity_matrix[mask].cpu().detach().numpy()
        
        return {
            'emb_mean_sim': float(np.mean(embedding_similarities)),
            'emb_std_sim': float(np.std(embedding_similarities)),
            'emb_max_sim': float(np.max(embedding_similarities)),
            'emb_min_sim': float(np.min(embedding_similarities)),
            'emb_median_sim': float(np.median(embedding_similarities)),
            
            'target_mean_sim': float(np.mean(sim_matrix_values)),
            'target_std_sim': float(np.std(sim_matrix_values)),
            'target_max_sim': float(np.max(sim_matrix_values)),
            'target_min_sim': float(np.min(sim_matrix_values)),
            'target_median_sim': float(np.median(sim_matrix_values)),
            
            'mean_diff': float(np.mean(np.abs(embedding_similarities - sim_matrix_values)))
        }
    
    def log_batch_stats(self, embeddings: torch.Tensor, similarity_matrix: torch.Tensor, 
                       epoch: int, batch_idx: int) -> Dict[str, float]:
        stats = self.compute_similarity_stats(embeddings, similarity_matrix)
        
        for key, value in stats.items():
            self.epoch_metrics[key].append(value)
        
        return stats

def train_model_with_monitoring(
    model: ViralGenomeEncoder,
    graphs: List[Data],
    similarity_data: Tuple[torch.Tensor, List[str]],
    num_epochs: int = 250,
    batch_size: int = 32,
    learning_rate: float = 0.0001,
    patience: int = 15,
    min_delta: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = ContrastiveLoss()
    monitor = EmbeddingSimilarityMonitor()
    
    best_loss = float('inf')
    epochs_without_improvement = 0
    best_epoch = 0
    loss_history = []
    
    similarity_matrix, labels = similarity_data
    if not isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = torch.tensor(similarity_matrix, dtype=torch.float32)
    
    dataset_size = len(graphs)
    print(f"Starting training with {dataset_size} samples...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        indices = torch.randperm(dataset_size)
        num_batches = (dataset_size + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_indices = indices[start_idx:end_idx]
            
            batch = Batch.from_data_list([graphs[idx] for idx in batch_indices.numpy()]).to(device)
            batch_sim = similarity_matrix[batch_indices][:, batch_indices].to(device)
            
            optimizer.zero_grad()
            embeddings = model(batch)
            loss = criterion(embeddings, batch_sim)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            stats = monitor.log_batch_stats(embeddings, batch_sim, epoch, batch_idx)
            
            if (batch_idx + 1) % monitor.log_freq == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{num_batches}")
                print(f"Loss: {loss.item():.4f}")
                print("Stats:")
                for key, value in stats.items():
                    print(f"  {key}: {value:.4f}")
        
        avg_loss = total_loss / num_batches
        loss_history.append(avg_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
                
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            epochs_without_improvement = 0
            best_epoch = epoch
        else:
            epochs_without_improvement += 1
            print(f"Epochs without improvement: {epochs_without_improvement}")
        
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping, not improving")
            print(f"Best loss was {best_loss:.4f} at epoch {best_epoch+1}")
            break
    
    save_model(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        loss=avg_loss,
        save_path="final"
    )
    
    return model, monitor

def load_similarity_matrix(file_path):
    
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    
    labels = df.index.tolist()
    
    tensor = torch.tensor(df.values, dtype=torch.float32)
    
    assert torch.all((tensor >= 0) & (tensor <= 1)), "Similarity matrix values must be between 0 and 1"
    
    return tensor, labels


sequences = read_fasta("sampled_fasta.fna")
dbg = DeBruijnGraph(k=8)
print(dbg)
graphs = dbg.batch_process_sequences(sequences)
# save_graphs(graphs, "viral_graphs_sampled.pkl")

graphs = load_graphs("viral_graphs_sampled.pkl")

model = ViralGenomeEncoder(
    num_node_features=1,
    hidden_dim=512,
    embedding_dim=64,
    num_layers=3
)

# Train the model

similarity_matrix_path="norm_distance_matrix_sampled.tsv"
similarity_matrix, labels = load_similarity_matrix(similarity_matrix_path)

trained_model, monitor = train_model_with_monitoring(
    model=model,
    graphs=graphs,
    similarity_data=(similarity_matrix, labels),
    num_epochs=250,
    batch_size=32
)
# Get final embeddings
embeddings, labels = get_embeddings_with_labels(trained_model, graphs, labels)
print(embeddings.shape)
save_embeddings(embeddings, labels, filepath="final")