import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool, global_max_pool, SAGPooling
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Dict, Optional
import numpy as np
import pickle
import os
import faiss
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import json
from datetime import datetime
from debruijin import DeBruijnGraph

class ViralGenomeKNN:
    def __init__(
        self,
        encoder: ViralGenomeEncoder,
        label_encoder: LabelEncoder = None,
        n_neighbors: int = 5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.encoder = encoder.to(device)  # Make sure encoder is on correct device
        self.device = device
        self.n_neighbors = n_neighbors
        self.label_encoder = label_encoder or LabelEncoder()
        self.index = None
        self.stored_labels = None
    
    def _prepare_batch(self, graphs: List[Data]) -> Data:

        graphs = [g.cpu() for g in graphs]
        
        num_nodes = [g.x.shape[0] for g in graphs]
        cum_nodes = torch.tensor([0] + num_nodes).cumsum(0)
        
        x = torch.cat([g.x for g in graphs], dim=0)
        
        edge_index = torch.cat([
            g.edge_index + cum_nodes[i] for i, g in enumerate(graphs)
        ], dim=1)
        
        batch = torch.cat([
            torch.full((num_nodes[i],), i, dtype=torch.long)
            for i in range(len(graphs))
        ])
        
        return Data(
            x=x.to(self.device),
            edge_index=edge_index.to(self.device),
            batch=batch.to(self.device)
        )
    
    def _compute_embeddings(self, graphs: List[Data], batch_size: int = 32) -> np.ndarray:
        self.encoder.eval()
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(graphs), batch_size):
                batch_graphs = graphs[i:i + batch_size]
                batch_data = self._prepare_batch(batch_graphs)
                batch_embeddings = self.encoder(batch_data).cpu().numpy()
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def fit(self, graphs: List[Data], labels: List[str]):
        if not hasattr(self.label_encoder, 'classes_'):
            self.stored_labels = self.label_encoder.fit_transform(labels)
        else:
            self.stored_labels = self.label_encoder.transform(labels)
        
        embeddings = self._compute_embeddings(graphs)
        
        print("FAISS index building ")
        self.index = faiss.IndexFlatL2(self.encoder.embedding_dim)
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        self.index.add(normalized_embeddings.astype('float32'))
    
    def predict(self, graphs: List[Data]) -> List[str]:
        embeddings = self._compute_embeddings(graphs)
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        
        print("Predicting")
        distances, indices = self.index.search(normalized_embeddings.astype('float32'), self.n_neighbors)
        
        predictions = []
        for idx_list in indices:
            neighbor_labels = self.stored_labels[idx_list]
            pred = np.bincount(neighbor_labels).argmax()
            predictions.append(pred)
        
        return self.label_encoder.inverse_transform(predictions)

def predict_new_sequences(
    model_path: str,
    fasta_path: str,
    k: int = 8,
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    print(f"Using device: {device}")
    classifier = load_knn_classifier(model_path, device=device)
    
    sequences, headers = read_fasta_with_headers(fasta_path)
    
    dbg = DeBruijnGraph(k=k)
    new_graphs = dbg.batch_process_sequences(sequences)
    
    predictions = classifier.predict(new_graphs)
    
    results = []
    for header, pred in zip(headers, predictions):
        results.append({
            'header': header,
            'predicted_class': pred
        })
    
    return results

def load_knn_classifier(filepath: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> ViralGenomeKNN:
    save_dict = torch.load(filepath, map_location=device)
    
    encoder = ViralGenomeEncoder(num_node_features=1).to(device)
    encoder.load_state_dict(save_dict['encoder_state'])
    
    label_encoder = LabelEncoder()
    label_encoder.classes_ = save_dict['label_encoder_classes']
    
    classifier = ViralGenomeKNN(encoder=encoder, label_encoder=label_encoder, device=device)
    classifier.stored_labels = save_dict['stored_labels']
    classifier.index = faiss.deserialize_index(save_dict['faiss_index'])
    
    return classifier