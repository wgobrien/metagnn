# metagnn.tools.data.loaders.py

import re, os
import pandas as pd

import torch
from torch_geometric.data import Batch, Data
from torch.utils.data import Dataset, random_split, DataLoader

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

from metagnn.tools.data.debruijn import DeBruijnGraph
from metagnn.tools.data.lzani import LzAniRunner
from metagnn.tools.common import MetaGNNConfig
from metagnn.utils import is_notebook, METAGNN_GLOBALS

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def process_sequence(args):
    debruijn, sequence = args
    return debruijn.generate_pyg_data(sequence, debruijn.k)

def extract_accession_id(header):
    match = re.search(r"\b[A-Z]{2}\d{6,7}\.\d\b", header)
    return match.group(0) if match else "NA"

class MetagenomeDataset(Dataset):
    def __init__(
        self,
        fasta_file: str,
        config: MetaGNNConfig,
        train: bool,
    ):
        self.k = config.k
        self.max_length = config.max_length
        self.num_workers = config.num_workers

        self.headers, self.sequences = self.load_fasta(fasta_file)
        
        # For training dataloader only
        if train is True:
            self.ani = self.build_or_get_ani(fasta_file)
        
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
    
        # one_hot_window = self.one_hot_encode(seq_window)
    
        # fft_seq = torch.fft.fft(one_hot_window, dim=0)
    
        # if len(sequence) < self.max_length:
        #     pad_size = self.max_length - len(sequence)
        #     fft_seq = torch.cat(
        #         [fft_seq, torch.zeros((pad_size, 4), dtype=torch.complex64)],
        #         dim=0,
        #     )

        return {
            # "fft": fft_seq,
            "graphs": self.graphs[idx],
            "headers":  self.headers[idx],
        }

    def metagenome_collate_fn(self, batch):
        # fft_mtx = self.fft_similarity(
        #     torch.stack([item["fft"] for item in batch], dim=0)
        # )
        graphs = [item["graphs"] for item in batch]
        batched_graphs = Batch.from_data_list(graphs)
        headers = [[item["headers"] for item in batch]]
        ani_mtx, wgt_mtx = self.ani_similarity([item["headers"] for item in batch])
        return {
            "graphs": batched_graphs,
            # "fft_mtx": fft_mtx,
            "ani_mtx": ani_mtx,
            "wgt_mtx": wgt_mtx,
            "headers": headers,
        }

    def ani_similarity(self, batch_ids):
        batch_size = len(batch_ids)
        ani_matrix = torch.eye(batch_size, dtype=torch.float32)
        weights_matrix = torch.eye(batch_size, dtype=torch.float32)
    
        for i, ref_id in enumerate(batch_ids):
            for j, query_id in enumerate(batch_ids):
                if ref_id != query_id:
                    try:
                        ani_score, len_ratio = self.ani.loc[(query_id, ref_id), ["ani", "len_ratio"]]
                        ani_matrix[i, j] = ani_score
                        weights_matrix[i, j] = len_ratio
                    except KeyError:
                        pass
    
        return ani_matrix, weights_matrix
    
    def fft_similarity(self, fft_batch):
        num_non_zeros = (fft_batch.abs().sum(dim=-1) > 0).sum(dim=1) # Effective lengths
        length_differences = torch.abs(num_non_zeros.unsqueeze(1) - num_non_zeros.unsqueeze(0))  
        length_sums = num_non_zeros.unsqueeze(1) + num_non_zeros.unsqueeze(0)
    
        self_product = fft_batch * torch.conj(fft_batch)
        self_correlation = torch.fft.ifft(self_product, dim=1)
        self_correlation = self_correlation.real + self_correlation.imag
        self_similarity = torch.max(self_correlation.sum(dim=-1), dim=1).values
    
        sequences_1 = fft_batch.unsqueeze(1)
        sequences_2 = fft_batch.unsqueeze(0)
        product = sequences_1 * torch.conj(sequences_2)
        cross_correlation = torch.fft.ifft(product, dim=2)
        cross_correlation = cross_correlation.real.sum(dim=-1) + cross_correlation.imag.sum(dim=-1)
        similarity_matrix = torch.max(cross_correlation, dim=2).values
    
        # Normalize by the relative length difference
        length_ratio = (1 - (length_differences / (length_sums + 1e-8)))
        similarity_matrix *= length_ratio
    
        normalization_factor = torch.sqrt(self_similarity.unsqueeze(1) * self_similarity.unsqueeze(0))
        similarity_matrix = similarity_matrix / (normalization_factor + 1e-8)
    
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

    def build_or_get_ani(self, fasta_file: str):
        file_name = re.search(r"([^/\\]+)\.fasta$", fasta_file).group(1)
        output_path = os.path.join(
            METAGNN_GLOBALS["save_folder"],
            f"{file_name}_ani.tsv"
        )

        if not os.path.exists(output_path):
            os.makedirs(METAGNN_GLOBALS["save_folder"], exist_ok=True)
            runner = LzAniRunner()
            runner.run(
                fasta_paths=fasta_file,
                output_path=output_path,
                num_threads=self.num_workers,
                verbose=False,
            )
        ani_df = pd.read_csv(output_path, sep="\t")
        for c in ["query", "reference"]:
            ani_df[c] = ani_df[c].astype(str).str.extract(
                r"(\b[A-Z]{2}\d{6,7}\.\d\b)"
            )
        ani_df.dropna(subset=["query", "reference"], inplace=True)
        ani_df.set_index(["query", "reference"], inplace=True)
        return ani_df[["ani", "len_ratio"]]
