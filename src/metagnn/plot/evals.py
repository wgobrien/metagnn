import random
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import pyro
import pyro.distributions as dist
from torch.utils.data import Subset, DataLoader

from metagnn.tools.common import get_pair_subset
from metagnn.plot.utils import activate_plot_settings
from metagnn.utils import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def sample_edges(model, batch, num_samples=(10,)):
    alpha_q = pyro.param("alpha_q")
    beta_q = pyro.param("beta_q")
    model.encoder.eval()
    
    z_loc, z_scale = model.encoder(
        batch["graphs"].x,
        batch["graphs"].edge_index,
        batch["graphs"].edge_weight,
        batch["graphs"].batch,
    )
    z_samples = dist.Normal(z_loc, z_scale).sample(num_samples).mean(0)
    decoded_mtx = model.decoder(z_samples)
    edge_proba_sparsity_posterior = dist.Beta(alpha_q, beta_q).sample(num_samples).mean(0)
    sns.histplot(decoded_mtx.edge_proba.detach().numpy())
    true_non_edge_probas, true_edge_probas = [], []
    for j in range(len(decoded_mtx.ptr) - 1):
        overlap_mask = get_pair_subset(
            decoded_mtx.get_example(j).edge_index,
            batch["graphs"].get_example(j).edge_index,
        )
        cur_example = decoded_mtx.get_example(j)
        cur_example.edge_proba *= (edge_proba_sparsity_posterior / ((1 - edge_proba_sparsity_posterior) + cur_example.edge_proba))
        
        true_edge_probas.append(
            cur_example.edge_proba[overlap_mask].detach().numpy()
        )
        true_non_edge_probas.append(
            cur_example.edge_proba[~overlap_mask].detach().numpy()
        )

    return true_non_edge_probas, true_edge_probas

def reconstruction(vae, batch, num_samples: int=(10,)):
    activate_plot_settings()

    true_non_edge_probas, true_edge_probas = sample_edges(vae, batch, num_samples=num_samples)
    plt.figure(figsize=(10, 6))
    for i in range(len(true_non_edge_probas)):
        sns.kdeplot(true_non_edge_probas[i], color="r", alpha=0.5, label="Non-edge probabilities" if i == 0 else "")
        sns.kdeplot(true_edge_probas[i], color="b", alpha=0.5, label="Edge probabilities" if i == 0 else "")
    plt.title("Reconstruction Probability Distributions")
    plt.xlabel("Edge Probability")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
