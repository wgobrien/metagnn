import random
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Subset, DataLoader

from metagnn.tools.common import get_pair_subset
from metagnn.plot.utils import activate_plot_settings
from metagnn.utils import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def edge_reconstruction(dataset, vae, sample_size):
    if vae.config.batch_size > sample_size:
        raise ValueError(
            f"`sample_size`={sample_size} must be greater "
            f"than `batch_size`={vae.config.batch_size}."
        )

    vae.encoder.eval()
    vae.decoder.eval()

    decoded_outputs, true_edge_probas, true_non_edge_probas = [], [], []
    
    sampled_indices = random.sample(range(len(dataset)), sample_size)
    subset = Subset(dataset, sampled_indices)

    sampled_dataloader = DataLoader(
        subset,
        batch_size=vae.config.batch_size,
        shuffle=False,
        collate_fn=dataset.metagenome_collate_fn,
    )

    iterator = tqdm(sampled_dataloader, desc="Reconstruction progress")
    with torch.no_grad():
        for batch in iterator:
            decoded_mtx = vae(batch)[1]

            for j in range(len(decoded_mtx.ptr) - 1):
                overlap_mask = get_pair_subset(
                    decoded_mtx.get_example(j).edge_index, batch["graphs"].get_example(j).edge_index
                )
                true_edge_probas.append(decoded_mtx.get_example(j).edge_proba[overlap_mask].detach().numpy())
                true_non_edge_probas.append(decoded_mtx.get_example(j).edge_proba[~overlap_mask].detach().numpy())

            decoded_outputs.append(decoded_mtx)

    return true_edge_probas, true_non_edge_probas

def reconstruction(dataset, vae, sample_size: int=None):
    activate_plot_settings()

    if sample_size is None:
        sample_size = min(len(dataset), 32)
        sample_size = max(vae.config.batch_size + 1, sample_size)
    
    true_edge_probas, true_non_edge_probas = edge_reconstruction(
        dataset, vae, sample_size
    )
    for i in range(len(true_edge_probas)):
        sns.kdeplot(true_non_edge_probas[i], color="r")
        sns.kdeplot(true_edge_probas[i], color="b")
    plt.show()