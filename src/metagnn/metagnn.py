import argparse

from metagnn.scripts.build import main as build_main
from metagnn.tools.common import MetaGNNConfig

def main():
    parser = argparse.ArgumentParser(prog="metagnn", description="MetaGNN CLI tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build MetaGNN models")
    build_parser.add_argument("--input_fasta", type=str, required=True, help="Path to the input FASTA file.")
    build_parser.add_argument("--run_id", type=str, help="Optional run ID.")

    build_parser.add_argument("--node_feature_dim", type=int, default=MetaGNNConfig.node_feature_dim, help="Node feature dimension.")
    build_parser.add_argument("--edge_attr_dim", type=int, default=MetaGNNConfig.edge_attr_dim, help="Edge attribute dimension.")
    build_parser.add_argument("--max_length", type=int, default=MetaGNNConfig.max_length, help="Maximum sequence length.")
    build_parser.add_argument("--k", type=int, default=MetaGNNConfig.k, help="Value of k for k-mers.")
    build_parser.add_argument("--num_epochs", type=int, default=MetaGNNConfig.num_epochs, help="Number of training epochs.")
    build_parser.add_argument("--batch_size", type=int, default=MetaGNNConfig.batch_size, help="Batch size.")
    build_parser.add_argument("--learning_rate", type=float, default=MetaGNNConfig.learning_rate, help="Learning rate.")
    build_parser.add_argument("--val_split", type=float, default=MetaGNNConfig.val_split, help="Validation split ratio.")
    build_parser.add_argument("--device", type=str, default=MetaGNNConfig.device, help="Device to run on ('cuda' or 'cpu').")
    build_parser.add_argument("--verbose", action="store_true", default=MetaGNNConfig.verbose, help="Enable verbose output.")
    build_parser.add_argument("--save_interval", type=int, default=MetaGNNConfig.save_interval, help="Interval for saving checkpoints.")
    build_parser.add_argument("--improvement_threshold", type=float, default=MetaGNNConfig.improvement_threshold, help="Threshold for improvement-based saving.")
    build_parser.add_argument("--num_workers", type=int, default=MetaGNNConfig.num_workers, help="Number of data loader workers.")
    build_parser.add_argument("--hidden_dim", type=int, default=MetaGNNConfig.hidden_dim, help="Hidden dimension size.")
    build_parser.add_argument("--latent_dim", type=int, default=MetaGNNConfig.latent_dim, help="Latent dimension size.")
    build_parser.add_argument("--num_layers", type=int, default=MetaGNNConfig.num_layers, help="Number of GNN layers.")
    build_parser.add_argument("--num_components", type=int, default=MetaGNNConfig.num_components, help="Number of Gaussian components.")
    build_parser.add_argument("--beta_strength", type=float, default=MetaGNNConfig.beta_strength, help="Beta strength for loss balancing.")

    args = parser.parse_args()

    if args.command == "build":
        build_main(args)

if __name__ == "__main__":
    main()