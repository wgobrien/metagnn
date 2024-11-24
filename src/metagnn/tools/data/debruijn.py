import torch
from torch_geometric.data import Data

from typing import List, Tuple, Dict, Optional
from itertools import product
from collections import defaultdict

from metagnn.utils import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

class DeBruijnGraph:
    def __init__(self, k: int=8, num_workers=1):
        self.k = k
        self.kmer_map = self._generate_kmer_map(self.k)
        self.idx_to_kmer = {idx: kmer for kmer, idx in self.kmer_map.items()}
        self.valid_chars = 'ACGT'
        self.num_nodes = self.get_num_nodes(k)
        self.num_workers = num_workers

    def _generate_kmer_map(self, k: int) -> Dict[str, int]:
        kmers = [''.join(p) for p in product('ACGT', repeat=k-1)]
        return {kmer: i for i, kmer in enumerate(kmers)}

    def generate_successors(self, k_minus_1_mer):
        return [
            self.kmer_map[k_minus_1_mer[1:] + n] for n in self.valid_chars
        ]

    def get_num_nodes(self, k: int) -> int:
        return 4 ** (k-1)

    def count_kmers(self, sequence: str, k: int) -> torch.Tensor:
        kmer_len = k - 1
        kmer_to_idx = self.kmer_map

        kmer_counts = defaultdict(int)
        for i in range(len(sequence) - kmer_len + 1):
            kmer = sequence[i:i+kmer_len]
            if all(c in self.valid_chars for c in kmer):
                kmer_counts[kmer] += 1

        counts = torch.zeros(self.num_nodes, dtype=torch.float32)
        for kmer, count in kmer_counts.items():
            if kmer in kmer_to_idx:
                counts[kmer_to_idx[kmer]] = count

        total_counts = counts.sum()
        if total_counts > 0:
            counts = counts / total_counts

        return counts

    def generate_pyg_data(self, sequence: str, k: int) -> Data:
        edge_counts = defaultdict(int)

        x = self.count_kmers(sequence, k).unsqueeze(1)

        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            if all(c in self.valid_chars for c in kmer):
                prefix_idx = self.kmer_map[kmer[:-1]]
                suffix_idx = self.kmer_map[kmer[1:]]
                edge_counts[(prefix_idx, suffix_idx)] += 1

        if edge_counts:
            edges, weights = zip(*edge_counts.items())
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            log_weights = torch.log(weights_tensor + 1e-6)
            edge_weight = (log_weights / log_weights.sum()).unsqueeze(1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weight = torch.zeros((0, 1), dtype=torch.float32)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

    def batch_process_sequences(self, sequences: List[str], show_progress: bool = True) -> List[Data]:
        graphs = []
        iterator = tqdm(sequences, desc="Building graphs") if show_progress else sequences

        for seq in iterator:
            graphs.append(self.generate_pyg_data(seq, self.k))

        return graphs

    def parallel_generate_graphs(self, sequences: List[str]):
        args = [(self, seq) for seq in sequences]

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            graphs = list(tqdm(
                executor.map(process_sequence, args),
                total=len(sequences),
                desc="Generating graphs"
            ))
        return graphs