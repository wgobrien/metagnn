# metagnn.tools.common.py

import torch
import dataclasses
from dataclasses import dataclass, field

@dataclass(unsafe_hash=True)
class MetaGNNConfig:
    ## DATA PARAMETERS
    node_feature_dim: int = 1 # batch.x.size(1)
    max_length: int = 8000
    k: int = 7

    ## TRAINING PARAMETERS
    num_epochs: int = 100
    batch_size: int = 16
    contrastive_only: bool = False
    learning_rate: float = .05
    val_split: float = 0.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: bool = True
    save_interval: int = 5
    improvement_threshold: float = .05
    num_workers: int = 8

    ## MODEL HYPERPARAMETERS
    hidden_dim: int = 256
    latent_dim: int = 16
    num_layers: int = 3
    num_components: int = 10
    margin: float = 1.
    
    def to_dict(self):
        res = dataclasses.asdict(self)
        return res
    
    @classmethod
    def from_dict(cls, config_dict):
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

def get_pair_subset(decoded_edges, observed_edges):
    decoded_edges_t = decoded_edges.T
    observed_edges_t = observed_edges.T

    matches = (decoded_edges_t.unsqueeze(1) == observed_edges_t.unsqueeze(0)).all(dim=2)
    return matches.any(dim=1)