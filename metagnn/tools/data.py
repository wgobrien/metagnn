import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from typing import List, Tuple, Dict, Optional
from itertools import product
from collections import defaultdict

from echidna.utils import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else 
    from tqdm import tqdm

class DeBruijnGraphs:
    def __init__(self, k_values: List[int] = [4, 5, 6]):
        """
        Initialize DeBruijn graph generator for multiple k values.
        """
        self.k_values = [k_values] if isinstance(k_values, int) else k_values
        self.kmer_maps = {k: self._generate_kmer_map(k) for k in k_values}
        # Create reverse mapping for node feature generation
        self.idx_to_kmer = {
            k: {idx: kmer for kmer, idx in kmer_map.items()}
            for k, kmer_map in self.kmer_maps.items()
        }
    
    def _generate_kmer_map(self, k: int) -> Dict[str, int]:
        """Generate mapping of (k-1)-mers to indices."""
        return {
            ''.join(kmer): i 
            for i, kmer in enumerate([''.join(p) for p in product('ACGT', repeat=k-1)])
        }
    
    def get_num_nodes(self, k: int) -> int:
        """Get number of nodes for a given k value."""
        return 4 ** (k-1)
    
    def count_kmers(self, sequence: str, k: int) -> torch.Tensor:
        """
        Count frequencies of (k-1)-mers in the sequence.
        
        Args:
            sequence: Input DNA sequence
            k: k-mer size (will count (k-1)-mers for nodes)
            
        Returns:
            Tensor of shape (4^(k-1),) containing normalized frequencies
        """
        kmer_len = k - 1
        num_nodes = self.get_num_nodes(k)
        kmer_to_idx = self.kmer_maps[k]
        
        # Count (k-1)-mers
        kmer_counts = defaultdict(int)
        for i in range(len(sequence) - kmer_len + 1):
            kmer = sequence[i:i+kmer_len]
            if set(kmer).issubset({'A', 'C', 'G', 'T'}):
                kmer_counts[kmer] += 1
        
        # Convert to tensor
        counts = torch.zeros(num_nodes, dtype=torch.float32)
        for kmer, count in kmer_counts.items():
            if kmer in kmer_to_idx:
                counts[kmer_to_idx[kmer]] = count
        
        # Normalize frequencies
        total_counts = counts.sum()
        if total_counts > 0:
            counts = counts / total_counts
            
        return counts
    
    def generate_pyg_data(self, sequence: str, k: int) -> Data:
        """
        Generate PyG Data object with k-mer frequency node features.
    
        Args:
            sequence: Input DNA sequence
            k: k-mer size
    
        Returns:
            PyG Data object containing:
                - x: Node features (normalized k-mer frequencies)
                - edge_index: Edge connectivity
                - edge_attr: Log-normalized edge weights
                - kmer_map: Dictionary mapping node indices to their k-mer sequences
        """
        kmer_to_idx = self.kmer_maps[k]
        edge_counts = defaultdict(int)
        num_nodes = self.get_num_nodes(k)
        
        # Generate node features from k-mer frequencies
        x = self.count_kmers(sequence, k)
        
        # Count edges
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            if set(kmer).issubset({'A', 'C', 'G', 'T'}):
                prefix_idx = kmer_to_idx[kmer[:-1]]
                suffix_idx = kmer_to_idx[kmer[1:]]
                edge_counts[(prefix_idx, suffix_idx)] += 1
        
        # Convert to tensors
        if edge_counts:
            edges, weights = zip(*((edge, count) for edge, count in edge_counts.items()))
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            
            # Log-normalize edge weights
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            log_weights = torch.log(weights_tensor + 1e-8)  # Avoid log(0) by adding epsilon
            edge_attr = (log_weights / log_weights.sum()).unsqueeze(1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float32)
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
    
    def batch_process_sequences(self, 
                              sequences: List[str],
                              show_progress: bool = True) -> Dict[int, List[Data]]:
        """Process multiple sequences for all k values."""
        graphs = {k: [] for k in self.k_values}
        iterator = tqdm(sequences, desc="Building graphs") if show_progress else sequences
        
        for seq in iterator:
            for k in self.k_values:
                graphs[k].append(self.generate_pyg_data(seq, k))
                
        return graphs

class MetagenomeDataset(Dataset):
    def __init__(self, fasta_file: str, max_length: int=5000, k: List[int]=[4,5]):
        self.headers, self.sequences = self.load_fasta(fasta_file)
        self.max_length = max_length
        
        self.fft = self.encode_sequences(self.sequences)
        self.debruijn = DeBruijnGraphs(k_values=k)
        self.graphs = self.debruijn.batch_process_sequences(self.sequences)
    
    def __len__(self):
        return len(self.headers)
    
    def __getitem__(self, idx):
        """
        Retrieve the FFT-transformed sequence and DeBruijn graph data for a given index,
        supporting multiple k values.
        
        Args:
            idx: Index of the sample to retrieve
        
        Returns:
            A dictionary containing:
                - 'fft_seq': FFT-transformed sequence data
                - 'graphs': A dictionary mapping each k to its corresponding DeBruijn graph data
        """
        fft = self.fft[idx, ...]
        
        # Gather graphs for all k values
        graphs = {k: self.graphs[k][idx] for k in self.debruijn.k_values}
        
        return {'fft': fft, 'graphs': graphs}
    
    def load_fasta(self, fasta_file):
        headers = []
        sequences = []
        with open(fasta_file, 'r') as file:
            header = None
            sequence = []
    
            file_iterator = tqdm(file, desc="Loading fasta")
            for line in file_iterator:
                line = line.strip()
                if line.startswith(">"):
                    if header and sequence:
                        headers.append(header)
                        sequences.append("".join(sequence))
                    header = line[1:]  # Store header without '>'
                    sequence = []      # Reset sequence list for new header
                else:
                    sequence.append(line)  # Add sequence line to current sequence
            
            # Add the last header and sequence
            if header and sequence:
                headers.append(header)
                sequences.append("".join(sequence))
        
        return headers, sequences

    def encode_sequences(self, sequences):
        # Mapping for encoding nucleotides
        mapping = {'A': [1, 0, 0, 0], 
                   'C': [0, 1, 0, 0], 
                   'G': [0, 0, 1, 0], 
                   'T': [0, 0, 0, 1]}
        
        mapping_tensor = torch.tensor([mapping.get(char, [0, 0, 0, 0]) for char in 'ACGT'], dtype=torch.float32)
        mapping_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}  # Indexes in mapping_tensor for fast lookup
        
        # Initialize a list to store encoded sequences
        encoded_sequences = []
    
        iterator = tqdm(sequences, desc="Encoding sequences")
        for sequence in iterator:
            # Convert each nucleotide to its respective encoding index
            indices = [mapping_dict[char] for char in sequence if char in mapping_dict]
            
            # Use indices to gather rows from the mapping tensor
            encoding = mapping_tensor[indices]
            encoding = torch.fft.fft(encoding, dim=1)
            
            # Pad if the encoded sequence is shorter than max_length
            if encoding.shape[0] < self.max_length:
                padding = torch.zeros((self.max_length - encoding.shape[0], 4), dtype=torch.float32)
                encoding = torch.cat([encoding, padding], dim=0)
            else:
                encoding = encoding[:self.max_length]  # Truncate if longer than max_length
                
            encoded_sequences.append(encoding)
        
        # Stack all encoded sequences to create a 3D tensor
        return torch.stack(encoded_sequences)