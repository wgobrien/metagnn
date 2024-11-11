import torch
import networkx as nx

def generate_random_dna(length=1000, samples=1, device='cpu'):
    """
    Generates random DNA sequences using PyTorch.

    Parameters:
    - length (int): The length of each DNA sequence.
    - samples (int): The number of DNA sequences to generate.
    - device (str): The device to store the sequences on ('cpu' or 'cuda').

    Returns:
    - list of str: A list of random DNA sequences.
    """
    bases = ['A', 'C', 'T', 'G']
    # Generate random indices for the bases
    random_indices = torch.randint(0, 4, (samples, length), device=device)
    random_sequences = [[bases[idx] for idx in row] for row in random_indices]
    joined_sequences = [''.join(seq) for seq in random_sequences]

    return joined_sequences

def get_k_mers(sequence, k):
    """
    Generates all k-mers of length k from a given sequence.

    Parameters:
    - sequence (str): DNA sequence.
    - k (int): Length of k-mers.

    Returns:
    - list of str: List of k-mers of length k.
    """
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]

def get_minimizers(sequences: list, k=31, l=15):
    """
    Finds the minimizers for a list of sequences.

    Parameters:
    - sequences (list of str): List of DNA sequences.
    - k (int): Length of k-mers.
    - l (int): Length of minimizer windows.

    Returns:
    - set of str: Set of unique minimizers found in the sequences.
    """
    minimizers = set()
    sequence_mins = []
    
    for s in sequences:
        kmers = get_k_mers(s, k)
        minimizers_tmp = set()
        sequence_min_tmp = []
        for mer in kmers:
            # Find all l-mers within the k-mer to determine the minimizer
            minimizer_set = set(get_k_mers(mer, l))
            # if len(minimizer_set) > 0:
            min_mer = min(minimizer_set)  # Lexicographically smallest l-mer
            minimizers_tmp.add(min_mer)
            sequence_min_tmp.append(min_mer)
            
        sequence_mins.append(sequence_min_tmp)
        minimizers.update(minimizers_tmp)

    return minimizers, sequence_mins

def de_bruijn_adjacency_matrices(sequences, k=31, l=8, device='cpu'):
    """
    Constructs a 3D adjacency matrix for the de Bruijn graphs of multiple sequences using minimizers.

    Parameters:
    - sequences (list of str): List of DNA sequences to build the graphs from.
    - k (int): The length of k-mers to use.
    - l (int): The length of the minimizers to use.
    - device (str): Device to store the adjacency matrix ('cpu' or 'cuda').

    Returns:
    - adj_matrices (torch.Tensor): A 3D matrix of shape (num_sequences, num_nodes, num_nodes).
    - nodes (list): List of unique nodes (minimizers) across all sequences in the order they appear in the adjacency matrices.
    """
    minimizers, sequence_minimizers = get_minimizers(sequences, k=k, l=l)

    # Sort the minimizers to have a consistent node order
    nodes = sorted(minimizers)

    node_index = {node: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)
    num_sequences = len(sequences)

    # Initialize the adjacency matrix
    adj_matrices = torch.zeros((num_sequences, num_nodes, num_nodes), dtype=torch.int, device=device)
    
    # Build adjacency matrices based on minimizers
    for seq_idx, min_list in enumerate(sequence_minimizers):
        G = nx.DiGraph()

        # Build edges between consecutive minimizers
        for i in range(len(min_list) - 1):
            start_min = min_list[i]
            end_min = min_list[i + 1]
            G.add_edge(start_min, end_min)

        # Populate the adjacency matrix for this sequence
        for start_node, end_node in G.edges():
            i, j = node_index[start_node], node_index[end_node]
            adj_matrices[seq_idx, i, j] += 1  # Set 1 for directed edge from start to end

    return adj_matrices, nodes
