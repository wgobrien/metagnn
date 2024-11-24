import argparse, os

from metagnn.tools.pipeline import train_metagnn
from metagnn.tools.common import MetaGNNConfig

def main(args):
    config = MetaGNNConfig(
        node_feature_dim=args.node_feature_dim,
        edge_attr_dim=args.edge_attr_dim,
        max_length=args.max_length,
        k=args.k,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        device=args.device,
        verbose=args.verbose,
        save_interval=args.save_interval,
        improvement_threshold=args.improvement_threshold,
        num_workers=args.num_workers,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        num_components=args.num_components,
        beta_strength=args.beta_strength,
    )

    input_fasta = os.path.abspath(args.input_fasta)
    if not os.path.exists(input_fasta):
        raise FileNotFoundError(f"Input FASTA file not found: {input_fasta}")

    train_metagnn(input_fasta, config, run_id=args.run_id)
