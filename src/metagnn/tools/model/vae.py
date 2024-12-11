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
        self.encoder = GraphEncoder(
            input_dim=config.node_feature_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            num_layers=config.num_layers,
        )

        self.decoder = GraphDecoder(
            kmer_map=kmer_map,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        )
        
        self.num_components = config.num_components
        self.latent_dim = config.latent_dim

        self.model = poutine.scale(self._model, scale=1/(config.batch_size))
        self.guide = poutine.scale(self._guide, scale=1/(config.batch_size))

        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    def forward(self, x, graph_sim): return self.model(x, graph_sim)

    @config_enumerate
    def _model(self, x=None, graph_sim=None):
        pyro.module("mixture_vae", self)
        batch_size = self.config.batch_size
        num_edges = 4*(4**(self.config.k-1))
        
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

        with pyro.plate("edges", num_edges):
            edge_proba_prior = pyro.sample("rho", dist.Beta(1., 10.))

        with pyro.plate("data", batch_size):
            z = pyro.sample("z", dist.Categorical(weights), infer={"enumerate": "parallel"})
            eta_loc = locs[z]
            eta_scale = scales[z]

            eta = pyro.sample(
                "eta",
                dist.Normal(eta_loc, eta_scale).to_event(1),
            )

            i, j = torch.triu_indices(batch_size, batch_size, offset=1)
            b_kernel = torch.exp(-0.5 * torch.sum((eta[i] - eta[j]) ** 2, dim=-1))
            b_scale = pyro.param("B_scale", torch.tensor(10.), dist.constraints.greater_than_eq(0.01))
            obs_graph_sim = graph_sim[i,j] if graph_sim is not None else None
            pyro.sample(
                "B",
                dist.Beta(1 + b_kernel * b_scale, 1 + (1 - b_kernel) * b_scale).to_event(1),
                obs=obs_graph_sim,
            )
        
            decoded_edges = self.decoder(eta)
            edge_probas, edge_observations = batch_to_obs(decoded_edges, x)
            
            edge_probas = edge_proba_prior * edge_probas / ((1 - edge_proba_prior) + edge_probas)
            edge_samples = pyro.sample(
                "obs",
                dist.Bernoulli(edge_probas).to_event(1),
                obs=edge_observations,
            )
        
        return edge_samples, decoded_edges

    @config_enumerate
    def _guide(self, x=None, graph_sim=None):
        pyro.module("mixture_vae", self)
        batch_size = self.config.batch_size
        num_edges = 4*(4**(self.config.k-1))
        
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
            pyro.sample("eta_locs", dist.Normal(eta_locs_q, 10).to_event(1))
            pyro.sample("eta_scales", dist.LogNormal(eta_scales_q, 2).to_event(1))
        
        # local vars
        z_loc, z_scale = self.encoder(
            x.x, x.edge_index, x.edge_weight, x.batch
        )
        z_scale = torch.nn.functional.softplus(z_scale)

        z_probs = pyro.param(
            "z_probs",
            torch.ones(batch_size, self.num_components) / self.num_components,
            constraint=dist.constraints.simplex,
        )
        z_probs = torch.clamp(z_probs, min=1e-6)
        alpha_q = pyro.param(
            "alpha_q",
            torch.ones(num_edges),
            constraint=dist.constraints.positive,
        )
        beta_q = pyro.param(
            "beta_q",
            torch.ones(num_edges),
            constraint=dist.constraints.positive,
        )
        with pyro.plate("edges", num_edges):
            pyro.sample("rho", dist.Beta(alpha_q, beta_q))

        with pyro.plate("data", batch_size):
            pyro.sample("z", dist.Categorical(z_probs),infer={"enumerate": "parallel"})
            
            eta_loc = torch.sum(z_probs.unsqueeze(-1) * eta_locs_q, dim=1)
            eta_scale = torch.sum(z_probs.unsqueeze(-1) * (eta_scales_q ** 2), dim=1)
            pyro.sample("eta", dist.Normal(eta_loc, torch.sqrt(eta_scale)).to_event(1))
        
        return z_loc, z_scale
