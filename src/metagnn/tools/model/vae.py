import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import config_enumerate

from datetime import datetime

from metagnn.tools.model.decoder import GraphDecoder
from metagnn.tools.model.encoder import GraphEncoder
from metagnn.tools.common import MetaGNNConfig, get_pair_subset

def batch_to_obs(decoded_batch, observed_batch):
    batch_size = len(observed_batch.ptr)-1
    num_edges = decoded_batch.get_example(0).edge_proba.shape[0]

    edge_observations = torch.zeros(batch_size, num_edges)
    edge_probas = torch.zeros(batch_size, num_edges)
    for i in range(batch_size):
        obs_edges = observed_batch.get_example(i).edge_index
        dec_edges = decoded_batch.get_example(i).edge_index
        pairs_mask = get_pair_subset(dec_edges, obs_edges)
    
        edge_probas[i] = decoded_batch.get_example(i).edge_proba
        edge_observations[i] = pairs_mask.to(torch.long)
    return edge_probas, edge_observations

class MixtureVAE(pyro.nn.PyroModule):
    def __init__(
        self,
        kmer_map: dict,
        config: MetaGNNConfig,
    ):
        super().__init__()

        self.config = config
        self.beta_strength = config.beta_strength
        self.encoder = GraphEncoder(
            input_dim=config.node_feature_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            num_layers=config.num_layers,
            edge_attr_dim=config.edge_attr_dim,
        )

        self.decoder = GraphDecoder(
            kmer_map=kmer_map,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        )
        
        self.num_components = config.num_components
        self.latent_dim = config.latent_dim
        self.eps = 1e-6

        self.model = poutine.scale(self._model, scale=1/(config.batch_size))
        self.guide = poutine.scale(self._guide, scale=1/(config.batch_size))

        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    def forward(self, x): return self.model(x)

    @config_enumerate
    def _model(self, batch):
        pyro.module("mixture_vae", self)

        fft_mtx = batch["fft_mtx"]
        x = batch["graphs"]
        batch_size = len(x.ptr)-1
        
        weights = pyro.sample(
            "weights", dist.Dirichlet(.5 * torch.ones(self.num_components))
        )

        with pyro.plate("components", self.num_components):
            locs = pyro.sample(
                "eta_locs",
                dist.Normal(0, 10).expand([self.latent_dim]).to_event(1),
            )
            scales = pyro.sample(
                "eta_scales",
                dist.LogNormal(0, 2).expand([self.latent_dim]).to_event(1),
            )
            
        with pyro.plate("data", batch_size):
            z = pyro.sample("z", dist.Categorical(weights))
            eta_loc = locs[z]
            eta_scale = scales[z]

            if len(z.shape) == 1:
                i, j = torch.triu_indices(batch_size, batch_size, offset=1)
                observed_beta = fft_mtx[i, j]
                same_component = (z[i] == z[j]).float()
                pyro.sample(
                    "similarity_penalty",
                    dist.Beta(
                        torch.clamp(self.beta_strength * same_component, min=self.eps),
                        torch.clamp(self.beta_strength * (1 - same_component), min=self.eps)
                    ).to_event(1),
                    obs=observed_beta
                )
            
            eta = pyro.sample(
                "eta",
                dist.Normal(eta_loc, eta_scale).to_event(1),
            )

            decoded_edges = self.decoder(eta)
            edge_probas, edge_observations = batch_to_obs(decoded_edges, x)
            edge_samples = pyro.sample(
                "obs",
                dist.Bernoulli(edge_probas).to_event(1),
                obs=edge_observations,
            )

            return edge_samples, decoded_edges

    @config_enumerate
    def _guide(self, batch):
        pyro.module("mixture_vae", self)
        
        fft_mtx = batch["fft_mtx"]
        x = batch["graphs"]
        batch_size = len(x.ptr) - 1

        # global vars
        weights_q = pyro.param(
            "weights_q", torch.ones(self.num_components) / self.num_components, constraint=dist.constraints.simplex
        )
        weights = pyro.sample("weights", dist.Dirichlet(weights_q))
        
        eta_locs_q = pyro.param(
            "eta_locs_q", 
            torch.randn(self.num_components, self.latent_dim),
        )
        eta_scales_q = pyro.param(
            "eta_scales_q", 
            torch.ones(self.num_components, self.latent_dim),
            constraint=dist.constraints.positive
        )
        
        with pyro.plate("components", self.num_components):
            pyro.sample("eta_locs", dist.Delta(eta_locs_q).to_event(1))
            pyro.sample("eta_scales", dist.Delta(eta_scales_q).to_event(1))
        
        # local vars
        z_loc, z_scale = self.encoder(
            x.x, x.edge_index, x.edge_weight, x.batch
        )
        z_scale = torch.nn.functional.softplus(z_scale)
    
        with pyro.plate("data", batch_size):
            z_probs = pyro.param(
                "z_probs",
                torch.ones(batch_size, self.num_components) / self.num_components,
                constraint=dist.constraints.simplex,
            )
            pyro.sample("z", dist.Categorical(z_probs))
            
            eta_loc = torch.sum(z_probs.unsqueeze(-1) * eta_locs_q, dim=1)
            eta_scale = torch.sum(z_probs.unsqueeze(-1) * (eta_scales_q ** 2), dim=1) + self.eps
            pyro.sample("eta", dist.Normal(eta_loc, torch.sqrt(eta_scale)).to_event(1))
        return z_loc, z_scale

"""

"""