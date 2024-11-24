from Bio import SeqIO, pairwise2
from Bio.Seq import Seq
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from tqdm import tqdm
import pandas as pd
import time
from datetime import datetime, timedelta

def load_sequences(fasta_file):
    """Load sequences from FASTA file into a dictionary with progress bar."""
    print("Loading sequences from FASTA file...")
    sequences = {}
    total_sequences = sum(1 for _ in open(fasta_file)) // 2  # Rough estimate for FASTA
    
    with tqdm(total=total_sequences, desc="Loading sequences") as pbar:
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences[record.id] = str(record.seq)
            pbar.update(1)
    
    print(f"Loaded {len(sequences)} sequences")
    return sequences

def align_pair(args):
    """Perform pairwise alignment for a single pair of sequences."""
    seq1_id, seq1, seq2_id, seq2 = args
    # Get alignment score
    score = pairwise2.align.globalxx(seq1, seq2, score_only=True)
    # Normalize by length of shorter sequence
    min_length = min(len(seq1), len(seq2))
    normalized_score = score / min_length
    return (seq1_id, seq2_id, normalized_score)

def parallel_pairwise_alignment(fasta_file, num_threads=32, chunk_size=10000):
    """
    Perform all-to-all pairwise alignment using parallel processing.
    Shows detailed progress information.
    """
    # Load sequences
    sequences = load_sequences(fasta_file)
    seq_ids = list(sequences.keys())
    n_seqs = len(seq_ids)
    
    # Calculate total number of alignments
    total_alignments = (n_seqs * (n_seqs - 1)) // 2
    print(f"\nTotal alignments to perform: {total_alignments:,}")
    
    # Create pairs for alignment
    print("\nGenerating sequence pairs...")
    pairs = list(itertools.combinations(seq_ids, 2))
    
    # Prepare input for parallel processing
    print("Preparing alignment arguments...")
    alignment_args = [(id1, sequences[id1], id2, sequences[id2]) 
                     for id1, id2 in pairs]
    
    # Initialize results matrix
    result_matrix = np.zeros((n_seqs, n_seqs))
    
    # Calculate number of chunks
    num_chunks = (len(alignment_args) + chunk_size - 1) // chunk_size
    print(f"\nProcessing in {num_chunks} chunks of size {chunk_size}")
    
    # Initialize counters
    completed_alignments = 0
    start_time = time.time()
    
    # Process alignments in parallel with nested progress bars
    with tqdm(total=total_alignments, desc="Total progress", position=0) as pbar_total:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for chunk_start in range(0, len(alignment_args), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(alignment_args))
                chunk = alignment_args[chunk_start:chunk_end]
                
                # Submit chunk of alignments
                futures = [executor.submit(align_pair, args) for args in chunk]
                
                # Process results as they complete
                for future in as_completed(futures):
                    seq1_id, seq2_id, score = future.result()
                    
                    # Update matrix
                    i = seq_ids.index(seq1_id)
                    j = seq_ids.index(seq2_id)
                    result_matrix[i, j] = score
                    result_matrix[j, i] = score
                    
                    # Update progress
                    completed_alignments += 1
                    pbar_total.update(1)
                    
                    # Calculate and display statistics every 1000 alignments
                    if completed_alignments % 1000 == 0:
                        elapsed_time = time.time() - start_time
                        alignments_per_second = completed_alignments / elapsed_time
                        remaining_alignments = total_alignments - completed_alignments
                        estimated_remaining_time = remaining_alignments / alignments_per_second
                        
                        # Clear previous line and print statistics
                        print(f"\rSpeed: {alignments_per_second:.1f} alignments/sec | "
                              f"Completed: {completed_alignments:,}/{total_alignments:,} | "
                              f"Est. remaining time: {timedelta(seconds=int(estimated_remaining_time))}", 
                              end='')
    
    print("\n\nFilling diagonal with self-alignment scores...")
    np.fill_diagonal(result_matrix, [len(sequences[id]) for id in seq_ids])
    
    print("Converting to DataFrame...")
    result_df = pd.DataFrame(result_matrix, index=seq_ids, columns=seq_ids)
    
    # Print final statistics
    total_time = time.time() - start_time
    print(f"\nCompleted {total_alignments:,} alignments in {timedelta(seconds=int(total_time))}")
    print(f"Average speed: {total_alignments/total_time:.1f} alignments/second")
    
    return result_df

def save_results(result_df, output_file):
    """Save results to file with progress bar."""
    print(f"\nSaving results to {output_file}...")
    result_df.to_csv(output_file)
    print("Results saved successfully!")

if __name__ == "__main__":
    # Example usage
    fasta_file = "/shares/swabseq/metagnn/data/combined.fasta"
    output_file = "/shares/swabseq/metagnn/data/alignment_matrix_combined.csv"
    
    print(f"Starting alignment process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run the alignment
    result_df = parallel_pairwise_alignment(fasta_file, num_threads=32)
    
    # Save results
    save_results(result_df, output_file)
    
    print(f"\nProcess completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")