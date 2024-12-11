# metagnn.tools.data.loaders.py

import re, os, json, gzip
import pandas as pd

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import degree
from torch.utils.data import Dataset, random_split, DataLoader

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

from metagnn.tools.data.debruijn import DeBruijnGraph
from metagnn.tools.data.mash import MashRunner
from metagnn.tools.common import MetaGNNConfig
from metagnn.utils import is_notebook, METAGNN_GLOBALS, get_logger

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import seaborn as sns

logger = get_logger(__name__)

def process_sequence(args):
    debruijn, sequence = args
    return debruijn.generate_pyg_data(sequence, debruijn.k)

def extract_accession_id(header):
    match = re.search(
        r"([A-Z]+_?[A-Z]*\d+\.\d+)",
        header,
    )
    return match.group(0) if match else "NA"

class MetagenomeDataset(Dataset):
    def __init__(
        self,
        fasta_file: str,
        config: MetaGNNConfig,
    ):
        self.k = config.k
        self.max_length = config.max_length
        self.num_workers = config.num_workers
        self.device = config.device
        # self.train = train

        self.headers, self.sequences = self.load_fasta(fasta_file)
        
        # For training dataloader only
        # if train is True:
        #     self.mash = self.build_or_get_mash(fasta_file)
        
        self.debruijn = DeBruijnGraph(k=self.k, num_workers=self.num_workers)
        if config.num_workers > 1:
            self.graphs = self.parallel_generate_graphs(self.sequences)
        else:
            self.graphs = self.debruijn.batch_process_sequences(self.sequences)
    
    def __len__(self):
        return len(self.headers)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # if len(sequence) > self.max_length:
        #     max_start = len(sequence) - self.max_length
        #     sample_window = torch.randint(
        #         low=0, high=max_start + 1, size=(1,)
        #     ).item()
        #     seq_window = sequence[sample_window:sample_window + self.max_length]
        # else:
        #     seq_window = sequence
    
        one_hot_window = self.one_hot_encode(sequence)
        fft_seq = torch.fft.fft(one_hot_window, dim=0)
    
        if len(sequence) < self.max_length:
            pad_size = self.max_length - len(sequence)
            fft_seq = torch.cat(
                [fft_seq, torch.zeros((pad_size, 4), dtype=torch.complex64)],
                dim=0,
            )
        else:
            fft_seq = fft_seq[:self.max_length]

        return {
            "fft": fft_seq.to(self.device),
            "graphs": self.graphs[idx].to(self.device),
            "headers":  self.headers[idx],
            "seq_len": len(self.sequences[idx])
        }

    def metagenome_collate_fn(self, batch):
        seq_lens = torch.tensor([item["seq_len"] for item in batch], dtype=torch.float32)
        fft_mtx = self.fft_similarity(
            torch.stack([item["fft"] for item in batch], dim=0),
            seq_lens,
        )
        graphs = [item["graphs"] for item in batch]
        batched_graphs = Batch.from_data_list(graphs)
        headers = [[item["headers"] for item in batch]]

        graph_sim = self.graph_similarity(batched_graphs)
        return {
            "graphs": batched_graphs,
            "fft_mtx": fft_mtx,
            "graph_sim": graph_sim,
            "headers": headers,
        }

    def graph_similarity(self, batch):
        num_graphs = batch.batch.max().item() + 1
        degrees_per_graph = torch.zeros((num_graphs, self.debruijn.num_nodes), device=batch.edge_index.device)
    
        for i in range(num_graphs):
            node_mask = batch.batch == i
            edge_mask = node_mask[batch.edge_index[0]] & node_mask[batch.edge_index[1]]
            edge_index = batch.edge_index[:, edge_mask]
            
            global_to_local = torch.zeros_like(node_mask, dtype=torch.long)
            global_to_local[node_mask] = torch.arange(node_mask.sum())
            local_edge_index = global_to_local[edge_index]
    
            # Compute node degrees for graph i
            degrees = degree(local_edge_index[0], num_nodes=node_mask.sum())
            # normalized_degrees = degrees / (degrees.sum() + 1e-8)  # Normalize by the total degree sum
            degrees_per_graph[i] = degrees
            # sns.histplot(normalized_degrees)
        similarity_matrix = torch.eye(num_graphs)
        for i in range(num_graphs):
            for j in range(i + 1, num_graphs):
                sim = torch.nn.functional.cosine_similarity(
                    degrees_per_graph[i].unsqueeze(0), degrees_per_graph[j].unsqueeze(0)
                ).item()
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        return similarity_matrix
    
    def mash_similarity(self, batch_ids):
        batch_size = len(batch_ids)
        mash_matrix = torch.zeros((batch_size, batch_size), dtype=torch.float32)
        weights_matrix = torch.zeros((batch_size, batch_size), dtype=torch.float32)
    
        for i, ref_id in enumerate(batch_ids):
            for j, query_id in enumerate(batch_ids):
                if ref_id != query_id:
                    mash_data = self.mash.get((query_id + "|" + ref_id), {})
                    mash_matrix[i, j] = mash_data.get("distance", 0.0)
                    weights_matrix[i, j] = mash_data.get("p_val", 0.0)

        # flip the magnitude - similiar should be close to 1, and high p val that match is random shoulded be weighted less (ie p_val=1 means we dont consider the pair) 
        return 1-mash_matrix, 1-weights_matrix
    
    def fft_similarity(self, fft_batch, seq_lens):
        # self-similarity
        self_product = fft_batch * torch.conj(fft_batch)
        self_correlation = torch.fft.ifft(self_product, dim=1).real
        self_similarity = torch.max(self_correlation.sum(dim=-1), dim=1).values
    
        # cross-similarity
        sequences_1 = fft_batch.unsqueeze(1)
        sequences_2 = fft_batch.unsqueeze(0)
        product = sequences_1 * torch.conj(sequences_2)
        cross_correlation = torch.fft.ifft(product, dim=2).abs().sum(dim=-1)
        similarity_matrix = torch.max(cross_correlation, dim=2).values
    
        # normalize by self-similarity
        normalization_factor = torch.sqrt(self_similarity.unsqueeze(1) * self_similarity.unsqueeze(0))
        similarity_matrix = similarity_matrix / (normalization_factor + 1e-8)
    
        # sequence length ratio
        lengths_i = seq_lens.unsqueeze(1)
        lengths_j = seq_lens.unsqueeze(0)
        length_ratio = 2 * torch.min(lengths_i, lengths_j) / (lengths_i + lengths_j + 1e-8)
    
        # regress out effect of sequence length ratio
        X = torch.stack([length_ratio.flatten(), torch.ones_like(length_ratio.flatten())], dim=1)
        y = similarity_matrix.flatten()
        beta = torch.linalg.lstsq(X, y).solution
        predicted_similarity = (beta[0] * length_ratio + beta[1]).view_as(similarity_matrix)
        similarity_matrix -= predicted_similarity
    
        # min-max normalization
        similarity_matrix_min = similarity_matrix.min()
        similarity_matrix_max = similarity_matrix.max()
        similarity_matrix = (similarity_matrix - similarity_matrix_min) / (similarity_matrix_max - similarity_matrix_min + 1e-8)
    
        return similarity_matrix

    def parallel_generate_graphs(self, sequences):
        args = [(self.debruijn, seq) for seq in sequences]

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            graphs = list(tqdm(
                executor.map(process_sequence, args),
                total=len(sequences),
                desc="Generating graphs"
            ))
        return graphs

    def one_hot_encode(self, sequence):
        mapping = {
            "A": [1, 0, 0, 0],
            "C": [0, 1, 0, 0],
            "G": [0, 0, 1, 0],
            "T": [0, 0, 0, 1],
        }
        encoded = [mapping.get(base, [0, 0, 0, 0]) for base in sequence]
        return torch.tensor(encoded, dtype=torch.float32)
    
    def load_fasta(self, fasta_file):
        headers = []
        sequences = []
        with open(fasta_file, "r") as file:
            header = None
            sequence = []
    
            file_iterator = tqdm(file, desc="Loading fasta")
            for line in file_iterator:
                line = line.strip()
                if line.startswith(">"):
                    if header and sequence:
                        headers.append(extract_accession_id(header))
                        sequences.append("".join(sequence))
                    header = line[1:]  # Store header without ">"
                    sequence = []      # Reset sequence list for new header
                else:
                    sequence.append(line)  # Add sequence line to current sequence
            
            # Add the last header and sequence
            if header and sequence:
                headers.append(extract_accession_id(header))
                sequences.append("".join(sequence))
        
        return headers, sequences

    def build_or_get_mash(self, fasta_file: str):
        file_name = re.search(r"([^/\\]+)\.fasta$", fasta_file).group(1)
        json_path = os.path.join(
            METAGNN_GLOBALS["save_folder"],
            f"{file_name}_mash.json.gz"
        )
        if not os.path.exists(json_path):
            output_path = os.path.join(
                METAGNN_GLOBALS["save_folder"],
                f"{file_name}_mash.tsv"
            )
            if not os.path.exists(output_path):
                os.makedirs(METAGNN_GLOBALS["save_folder"], exist_ok=True)
                runner = MashRunner()
                runner.run(
                    reference_path=fasta_file,
                    query_paths=fasta_file,
                    kmer_size=14,
                    sketch_size=10_000,
                    output_path=output_path,
                    num_threads=self.num_workers,
                )
            else:
                logger.info(f"Loading stored pairs from {output_path}...")
    
            pattern = re.compile(r"([A-Z]+_?[A-Z]*\d+\.\d+)")
            mash_dict = {}
    
            chunk_iter = pd.read_csv(
                output_path,
                sep="\t",
                dtype={"reference": str, "query": str, "distance": float, "p_val": float, "counts": str},
                chunksize=1_000_000,
                header=None,
                names=["reference", "query", "distance", "p_val", "counts"]
            )
    
            for chunk in chunk_iter:
                chunk["query"] = chunk["query"].str.extract(pattern)
                chunk["reference"] = chunk["reference"].str.extract(pattern)
                chunk.dropna(subset=["query", "reference"], inplace=True)
                mash_dict.update({
                    f"{row['query']}|{row['reference']}": {
                        "distance": row["distance"],
                        "p_val": row["p_val"]
                    }
                    for _, row in chunk.iterrows()
                })
    
            with gzip.open(json_path, "wt", encoding="utf-8") as f:
                json.dump(mash_dict, f)
        else:
            logger.info(f"Loading precomputed mash JSON from {json_path}...")
            with gzip.open(json_path, "rt", encoding="utf-8") as f:
                mash_dict = json.load(f)
    
        return mash_dict
