import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

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
    
    def _model(self, batch):
        pyro.module("mixture_vae", self)

        # fft_batch = batch["fft"]
        x = batch["graphs"]
        batch_size = len(x.ptr)-1
        
        weights = pyro.sample(
            "weights", dist.Dirichlet(torch.ones(self.num_components))
        )

        with pyro.plate("components", self.num_components):
            mixture_means = pyro.sample(
                "mixture_means",
                dist.Normal(0, 1).expand([self.latent_dim]).to_event(1),
            )
            mixture_scales = pyro.sample(
                "mixture_scales",
                dist.LogNormal(0, 1).expand([self.latent_dim]).to_event(1),
            )

        # similarity_matrix = compute_similarity(fft_batch)
        # i, j = torch.triu_indices(batch_size, batch_size, offset=1)
        # observed_beta = similarity_matrix[i, j]
    
        with pyro.plate("data", batch_size):
            z = pyro.sample("z", dist.Categorical(probs=weights))
            
            # same_component = (z[i] == z[j]).float()
            # pyro.sample(
            #     "similarity_penalty",
            #     dist.Beta(
            #         torch.clamp(self.beta_strength * same_component + self.eps, min=self.eps),
            #         torch.clamp(self.beta_strength * (1 - same_component) + self.eps, min=self.eps)
            #     ).to_event(1),
            #     obs=observed_beta
            # )
    
            mu = mixture_means[z]
            scale = mixture_scales[z]
        
            latent = pyro.sample(
                "latent",
                dist.Normal(mu, scale).to_event(1),
            )

            decoded_edges = self.decoder(latent)
            edge_probas, edge_observations = batch_to_obs(decoded_edges, x)
            edge_samples = pyro.sample(
                "obs",
                dist.Bernoulli(edge_probas).to_event(1),
                obs=edge_observations,
            )

        return edge_samples, decoded_edges

    def _guide(self, batch):
        pyro.module("mixture_vae", self)

        x = batch["graphs"]
        batch_size = len(x.ptr) - 1
    
        mixture_means_posterior = pyro.param(
            "mixture_means_posterior",
            torch.randn(self.num_components, self.latent_dim)
        )
        mixture_scales_posterior = pyro.param(
            "mixture_scales_posterior",
            torch.ones(self.num_components, self.latent_dim),
            constraint=dist.constraints.positive
        )
        mixture_means_posterior_var = pyro.param(
            "mixture_means_posterior_var",
            torch.ones(self.num_components, self.latent_dim),
            constraint=dist.constraints.positive,
        )
        mixture_scales_posterior_var = pyro.param(
            "mixture_scales_posterior_var",
            torch.ones(self.num_components, self.latent_dim)*.1,
            constraint=dist.constraints.positive
        )
        z_logits = pyro.param(
            "z_logits",
            torch.zeros(batch_size, self.num_components)+self.eps,
            constraint=dist.constraints.simplex,
        )
        # temperature = pyro.param(
        #     "temperature",
        #     torch.ones(self.num_components),
        #     constraint=dist.constraints.positive
        # )

        weights_posterior = pyro.param(
            "weights_posterior",
            torch.ones(self.num_components),
            constraint=dist.constraints.positive
        )
        weights = pyro.sample("weights", dist.Dirichlet(weights_posterior))
        
        with pyro.plate("components", self.num_components):
            pyro.sample(
                "mixture_means",
                dist.Normal(mixture_means_posterior, mixture_means_posterior_var).to_event(1)
            )
            pyro.sample(
                "mixture_scales",
                dist.LogNormal(mixture_scales_posterior, mixture_scales_posterior_var).to_event(1)
            )
    
        z_mu, z_var = self.encoder(
            x.x,
            x.edge_index,
            x.edge_weight,
            x.batch,
        )
    
        with pyro.plate("data", batch_size):
            pyro.sample(
                "latent",
                dist.Normal(z_mu, z_var).to_event(1)
            )
            pyro.sample("z", dist.Categorical(probs=z_logits))
        
        return z_mu, z_var
