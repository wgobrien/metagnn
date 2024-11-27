# metagnn.tools.data.loaders.py

import re, os, json, gzip
import pandas as pd

import torch
from torch_geometric.data import Batch, Data
from torch.utils.data import Dataset, random_split, DataLoader

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

from metagnn.tools.data.debruijn import DeBruijnGraph
from metagnn.tools.data.lzani import LzAniRunner
from metagnn.tools.common import MetaGNNConfig
from metagnn.utils import is_notebook, METAGNN_GLOBALS, get_logger

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

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
        train: bool,
    ):
        self.k = config.k
        self.max_length = config.max_length
        self.num_workers = config.num_workers
        self.train = train

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
            "fft": fft_seq,
            "graphs": self.graphs[idx],
            "headers":  self.headers[idx],
        }

    def metagenome_collate_fn(self, batch):
        fft_mtx = self.fft_similarity(
            torch.stack([item["fft"] for item in batch], dim=0)
        )
        graphs = [item["graphs"] for item in batch]
        batched_graphs = Batch.from_data_list(graphs)
        headers = [[item["headers"] for item in batch]]
        if self.train is True:
            ani_mtx, wgt_mtx = self.ani_similarity([item["headers"] for item in batch])
            return {
                "graphs": batched_graphs,
                "fft_mtx": fft_mtx,
                "ani_mtx": ani_mtx,
                "wgt_mtx": wgt_mtx,
                "headers": headers,
            }
        return {
            "graphs": batched_graphs,
            "fft_mtx": fft_mtx,
            "headers": headers,
        }

    def ani_similarity(self, batch_ids):
        batch_size = len(batch_ids)
        ani_matrix = torch.eye(batch_size, dtype=torch.float32)
        weights_matrix = torch.eye(batch_size, dtype=torch.float32)
    
        for i, ref_id in enumerate(batch_ids):
            for j, query_id in enumerate(batch_ids):
                if ref_id != query_id:
                    ani_data = self.ani.get((query_id + "|" + ref_id), {})
                    ani_matrix[i, j] = ani_data.get("ani", 0.0)
                    weights_matrix[i, j] = ani_data.get("len_ratio", 0.0)
    
        return ani_matrix, weights_matrix
    
    def fft_similarity(self, fft_batch):
        num_non_zeros = (fft_batch.abs().sum(dim=-1) > 0).sum(dim=1)  # Effective lengths
        length_differences = torch.abs(num_non_zeros.unsqueeze(1) - num_non_zeros.unsqueeze(0))
        length_sums = num_non_zeros.unsqueeze(1) + num_non_zeros.unsqueeze(0)
    
        self_product = fft_batch * torch.conj(fft_batch)
        self_correlation = torch.fft.ifft(self_product, dim=1)
        self_correlation = self_correlation.real
        self_similarity = torch.max(self_correlation.sum(dim=-1), dim=1).values
    
        sequences_1 = fft_batch.unsqueeze(1)
        sequences_2 = fft_batch.unsqueeze(0)
        product = sequences_1 * torch.conj(sequences_2)
        cross_correlation = torch.fft.ifft(product, dim=2)
        cross_correlation = torch.abs(cross_correlation).sum(dim=-1)
        similarity_matrix = torch.max(cross_correlation, dim=2).values
    
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
        json_path = os.path.join(
            METAGNN_GLOBALS["save_folder"],
            f"{file_name}_ani.json.gz"
        )
        if not os.path.exists(json_path):
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
            else:
                logger.info(f"Loading stored pairs from {output_path}...")
    
            pattern = re.compile(r"([A-Z]+_?[A-Z]*\d+\.\d+)")
            ani_dict = {}
        
            chunk_iter = pd.read_csv(
                tsv_path,
                sep="\t",
                usecols=["query", "reference", "ani", "len_ratio"],
                dtype={"query": str, "reference": str, "ani": float, "len_ratio": float},
                chunksize=1_000_000
            )
        
            for chunk in chunk_iter:
                chunk["query"] = chunk["query"].str.extract(pattern)
                chunk["reference"] = chunk["reference"].str.extract(pattern)
                chunk.dropna(subset=["query", "reference"], inplace=True)
                ani_dict.update({
                    f"{row['query']}|{row['reference']}": {"ani": row["ani"], "len_ratio": row["len_ratio"]}
                    for _, row in chunk.iterrows()
                })
        
            with gzip.open(json_path, "wt", encoding="utf-8") as f:
                json.dump(ani_dict, f)
        else:
            logger.info(f"Loading precomputed ANI JSON from {json_path}...")
            with gzip.open(json_path, "rt", encoding="utf-8") as f:
                ani_dict = json.load(f)
        
        # ani_dict = {tuple(k.split("|")): v for k, v in ani_dict.items()}
        return ani_dict
