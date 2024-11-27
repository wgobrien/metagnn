from torch_geometric.utils import batched_negative_sampling
from torch.utils.data import random_split, DataLoader
import torch

import numpy as np
from typing import List

import pyro
import pyro.optim as optim
from pyro.infer import SVI, TraceEnum_ELBO

import seaborn as sns
import matplotlib.pyplot as plt

from metagnn.utils import is_notebook, get_logger
from metagnn.tools.common import get_pair_subset, MetaGNNConfig
from metagnn.tools.utils import save_model, load_model
from metagnn.tools.model.vae import MixtureVAE
from metagnn.tools.data.loaders import MetagenomeDataset
from metagnn.plot.utils import activate_plot_settings, plot_loss

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

logger = get_logger(__name__)

def get_num_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_metagnn(
    fasta: str,
    config: MetaGNNConfig=None,
    run_id: str=None,
):
    if run_id is not None:
        # if continuing from prev run, load old config
        vae = load_model(run_id)

        if config is not None:
            # if updating config, update mutable params only
            vae.config.num_epochs = config.num_epochs
            vae.config.batch_size = config.batch_size
            vae.config.learning_rate = config.learning_rate
            vae.config.val_split = config.val_split
            vae.config.device = config.device
            vae.config.verbose = config.verbose
            vae.config.save_interval = config.save_interval
            vae.config.improvement_threshold = config.improvement_threshold
            vae.config.num_workers = config.num_workers
        
        config = vae.config

    if config is None:
        config = MetaGNNConfig()
  
    dataset = MetagenomeDataset(fasta, config, train=True)

    val_dataloader = None
    if config.val_split > 0:
        val_size = int(config.val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            collate_fn=dataset.metagenome_collate_fn,
            shuffle=False,
            drop_last=True,
        )
    else:
        train_dataset = dataset

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=dataset.metagenome_collate_fn,
        shuffle=True,
        drop_last=True,
    )

    if run_id is None:
        pyro.clear_param_store()
        vae = MixtureVAE(
            kmer_map=dataset.debruijn.kmer_map,
            config=config,
        )
    logger.info(f"{get_num_model_params(vae)} paramter model")
    vae.to(config.device)
    training_loop(vae, train_dataloader, val_dataloader)

def training_loop(
    vae,
    train_dataloader,
    val_dataloader=None,
):
    activate_plot_settings()
    config = vae.config
    device = config.device

    optimizer = optim.ClippedAdam({"lr": config.learning_rate})
    num_epochs = config.num_epochs
    total_iterations = num_epochs * len(train_dataloader)
    progress_bar = tqdm(total=total_iterations, desc="Training Progress")
    
    epoch_losses, batch_losses, val_losses = [], [], []
    loss_fn = loss_fn_factory(config, progress_bar)
    svi = SVI(vae.model, vae.guide, optimizer, loss=loss_fn)

    best_loss = torch.inf
    save_interval = config.save_interval
    improvement_threshold = config.improvement_threshold
    
    for epoch in range(num_epochs):
        vae.train()

        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_dataloader):
            batch_loss = svi.step(batch, epoch)
            epoch_loss += batch_loss
            batch_losses.append(batch_loss)

            avg_loss = epoch_loss / ((epoch * len(train_dataloader)) + batch_idx + 1)
            progress_bar.set_description(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")
            progress_bar.update(1)

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        epoch_losses.append(avg_epoch_loss)

        if val_dataloader is not None:
            vae.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_loss += svi.evaluate_loss(val_batch)
            avg_val_loss = val_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)
            progress_bar.set_description(
                f"Epoch {epoch} | Train Loss: {avg_epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
            )

            if avg_val_loss < best_loss and (best_loss - avg_val_loss) / best_loss > improvement_threshold:
                best_loss = avg_val_loss
                save_model(vae, overwrite=True, verbose=False)
                logger.info(f"Model saved at epoch {epoch} with val loss: {avg_val_loss:.4f}")
        
        if epoch % save_interval == 0:
            save_model(vae, overwrite=True, verbose=False)
            logger.info(f"Checkpoint model saved at epoch {epoch}")  
    
    save_model(vae, overwrite=True, verbose=False)
    if config.verbose is True:
        plot_loss(epoch_losses, label="epoch train", log_loss=True)
        plot_loss(batch_losses, label="batch train", log_loss=True)
        if len(val_losses) > 0:
            plot_loss(validation_loss, label="validation", log_loss=True)
    progress_bar.close()

def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = torch.norm(anchor - positive, p=2, dim=1)
    neg_dist = torch.norm(anchor - negative, p=2, dim=1)

    loss = torch.clamp(pos_dist - neg_dist + margin, min=0)
    return loss.mean()

def reconstruction_loss(decoded_batch, observed_batch):
    device = observed_batch.edge_index.device
    batch_size = len(observed_batch.ptr) - 1
    total_loss = 0.0
    
    for i in range(batch_size):
        # obs_start = observed_batch.ptr[i].item()
        # mask_obs = (
        #     (observed_batch.edge_index[0] >= obs_start)
        #     &
        #     (observed_batch.edge_index[0] < observed_batch.ptr[i + 1].item())
        # )
        # observed_edges = observed_batch.edge_index[:, mask_obs]
        # observed_edges -= obs_start
        # num_pos_edges = observed_edges.size(1)

        # decoded_start = decoded_batch.ptr[i].item()
        # mask_decoded = (
        #     (decoded_batch.edge_index[0] >= decoded_start)
        #     &
        #     (decoded_batch.edge_index[0] < decoded_batch.ptr[i + 1].item())
        # )
        # decoded_edges = decoded_batch.edge_index[:, mask_decoded]
        # decoded_edges -= decoded_start
        # edge_proba = decoded_batch.edge_proba[mask_decoded]
        # num_negative_samples = num_pos_edges
        
        obs_i = observed_batch.get_example(i)
        dec_i = decoded_batch.get_example(i)
        edge_proba = dec_i.edge_proba
        
        pos_edge_mask = get_pair_subset(dec_i.edge_index, obs_i.edge_index)
        num_pos_samples = pos_edge_mask.sum()
        
        pos_probas = edge_proba[pos_edge_mask]
        neg_probas = edge_proba[~pos_edge_mask]
        
        perm = torch.randperm(neg_probas.size(0))#[:num_pos_samples]
        neg_probas = neg_probas[perm]

        pos_target = torch.ones_like(pos_probas)
        neg_target = torch.zeros_like(neg_probas)
    
        pos_loss = torch.nn.functional.binary_cross_entropy(
            pos_probas, pos_target, reduction='mean'
        )
        neg_loss = torch.nn.functional.binary_cross_entropy(
            neg_probas, neg_target, reduction='mean'
        )
    
        total_loss += (pos_loss + neg_loss)
    return total_loss / batch_size

def contrastive_loss(z_mu, similarity_matrix, weights_matrix=None, margin=1.0):
    batch_size = z_mu.size(0)
    
    pairwise_distances = torch.cdist(z_mu, z_mu, p=2) ** 2
    positive_loss = similarity_matrix * pairwise_distances
    negative_loss = (1 - similarity_matrix) * torch.clamp(margin - pairwise_distances.sqrt(), min=0) ** 2
    contr_loss = (positive_loss + negative_loss)
    
    if weights_matrix is not None:
        contr_loss *= weights_matrix
        return contr_loss.sum() / (weights_matrix.sum() + 1e-8)
    
    return contr_loss.mean()

def loss_fn_factory(config, progress_bar=None):
    steps = torch.arange(config.num_epochs)
    scale = steps / 10
    shift = config.num_epochs / 2.5
    kl_annealing = torch.sigmoid((steps - shift) / scale)
    
    def loss(model, guide, batch, epoch):
        batch_fft = batch["fft_mtx"]
        batch_graphs = batch["graphs"]
        batch_size = config.batch_size

        anneal = kl_annealing[epoch]
        elbo_loss = TraceEnum_ELBO(max_plate_nesting=1).differentiable_loss(model, guide, batch) * anneal
        
        reco_loss = reconstruction_loss(
            observed_batch=batch_graphs,
            decoded_batch=model(batch)[1],
        ) * (4 * 4**(config.k-1))
        
        cont_loss = contrastive_loss(
            guide(batch)[0],
            batch["ani_mtx"], 
            batch["wgt_mtx"],
            margin=config.margin,
        ) * 4 * (4**(config.k-1))

        batch_loss = cont_loss + reco_loss + elbo_loss
        
        if progress_bar is not None:
            progress_bar.set_postfix(
                batch_loss=batch_loss.item(),
                elbo_loss=elbo_loss.item(),
                contrastive_loss=cont_loss.item(),
                reconstruction_loss=reco_loss.item(),
                anneal=anneal.item(),
            )
        
        return batch_loss
    return loss