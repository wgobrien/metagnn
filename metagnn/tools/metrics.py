# metrics.py

import torch

def compute_similarity_matrix(fft_seqs):
    """
    Compute a similarity matrix for a batch of sequences using cross-correlation of FFTs.
    
    Parameters:
        fft_seqs (torch.Tensor): Tensor of shape [n_obs, max_length, 4].
        
    Returns:
        torch.Tensor: Similarity matrix of shape [n_obs, n_obs].
    """
    # Equivalent to torch.einsum('ilk,jlk->ij', fft_seqs, fft_seqs_conj)
    fft_seqs_flat = fft_seqs.view(fft_seqs.shape[0], -1)  # [n_obs, length * 4]
    similarity_matrix = torch.abs(fft_seqs_flat @ torch.conj(fft_seqs_flat).T) # [n_obs, n_obs]

    # Normalize
    magnitudes = torch.sqrt(torch.sum(torch.abs(fft_seqs_flat) ** 2, dim=1, keepdim=True))
    similarity_matrix /= (magnitudes @ magnitudes.T + 1e-8)

    return similarity_matrix

def frobenius_norm(matrix):
    """
    Compute the Frobenius norm of a matrix (zero out diagonal).
    
    Parameters:
        matrix (torch.Tensor): Input tensor.
        
    Returns:
        torch.Tensor: The Frobenius norm, a scalar value.
    """
    matrix = matrix.clone().fill_diagonal_(0)
    return torch.sqrt(torch.sum(torch.abs(matrix) ** 2) / matrix.size(0))

def frobenius_norm_cross_correlation(fft_seqs):
    """
    Compute the Frobenius norm of the off-diagonal elements using block operations
    for better space efficiency.
    
    Parameters:
        fft_seqs (torch.Tensor): Tensor of shape [n_obs, max_length, 4]
    Returns:
        torch.Tensor: Frobenius norm of the off-diagonal elements
    """
    # Flatten fft_seqs for correlation
    fft_seqs_flat = fft_seqs.view(fft_seqs.shape[0], -1)
    
    # Compute magnitudes once
    magnitudes = torch.sqrt(torch.sum(torch.abs(fft_seqs_flat) ** 2, dim=1))
    
    # Compute matrix-vector products in chunks and accumulate squared values
    block_size = 1  # Adjust based on available memory
    n_blocks = (fft_seqs_flat.shape[0] + block_size - 1) // block_size
    
    frobenius_sum = torch.zeros(1, device=fft_seqs.device)
    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, fft_seqs_flat.shape[0])
        
        # Compute block of similarity matrix
        block = torch.abs(fft_seqs_flat[start_idx:end_idx] @ torch.conj(fft_seqs_flat).T)
        block = block / (magnitudes[start_idx:end_idx, None] * magnitudes[None, :] + 1e-8)
        
        # Zero out diagonal elements in this block
        if start_idx < end_idx:
            block.diagonal()[max(-start_idx, 0):] = 0
            
        # Accumulate squared values
        frobenius_sum += torch.sum(block ** 2)

    return torch.sqrt(frobenius_sum / fft_seqs.shape[0])