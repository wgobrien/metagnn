# Metagenomic Classification with Deep Variational Graph Neural Networks and Mixture Modeling

## Installation

The package can be downloaded with `poetry`. Create a conda environment and install poetry:

`conda create --name metagnn python=3.10`

`conda activate metagnn`

`pip install poetry biopython faiss`

`conda install bioconda::mash`

Poetry allows you to install package dependencies and adds the package to your path. Use `poetry install` to complete the build.

## Usage

With the package installed, you can use the package in a variety of ways. Typical usage would install the package with `import metagnn as mg`. Then, you have the set of functions available in `tools` imported as `.tl`. To train a model, run `mg.tl.train_metagnn`. The config can be created with `mg.tl.MetaGNNConfig`, see the table below for a table of configuration settings.

To work with the package from the command line, several commands are available or are works in progress. Training can be performed with `metagnn build`, with options to pick up trainig on a model by passing a `run_id`.

Future commands `metagnn --run_id` will perform classification using FAISS(Facebook AI Similarity Search). Additional functions are under development.

## Data

To download the refseq dataset, run the following command with curl:

`curl -O ftp://ftp.ncbi.nlm.nih.gov/refseq/release/viral/viral.*.genomic.fna.gz`

## Configuration
| Setting                | Default (Type)            | Description                                                                 |
|------------------------|---------------------------|-----------------------------------------------------------------------------|
| **DATA PARAMETERS**    |                           |                                                                             |
| `node_feature_dim`     | 1 (int)                  | Dimension of node features, e.g., `batch.x.size(1)`.                        |
| `max_length`           | 8000 (int)               | Maximum sequence length.                                                   |
| `k`                    | 7 (int)                  | Value of \( k \) for k-mer embedding or processing.                         |
| **TRAINING PARAMETERS**|                           |                                                                             |
| `num_epochs`           | 100 (int)                | Number of training epochs.                                                 |
| `batch_size`           | 16 (int)                 | Size of each training batch.                                               |
| `learning_rate`        | 0.05 (float)             | Learning rate for optimization.                                            |
| `val_split`            | 0.0 (float)              | Fraction of data used for validation split.                                 |
| `device`               | `"cuda"` (str)           | Device for computation (`"cuda"` or `"cpu"`).                               |
| `verbose`              | True (bool)              | Whether to display detailed logs during training.                           |
| `save_interval`        | 5 (int)                  | Interval (in epochs) at which to save the model checkpoint.                 |
| `improvement_threshold`| 0.05 (float)             | Minimum improvement threshold for saving the model.                         |
| `num_workers`          | 8 (int)                  | Number of workers for data loading.                                         |
| **MODEL HYPERPARAMETERS**|                        |                                                                             |
| `hidden_dim`           | 256 (int)                | Dimension of hidden layers for the encoder and decoder architectures.                                    |
| `latent_dim`           | 16 (int)                 | Dimension of the latent representation.                                    |
| `num_layers`           | 3 (int)                  | Number of encoder/decoder layers.                                     |
| `num_components`       | 10 (int)                 | Number of components for the mixture model.                      |