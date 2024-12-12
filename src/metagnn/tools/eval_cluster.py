import torch
import pickle
import numpy as np
from Bio import SeqIO
from collections import defaultdict
import pandas as pd
from sklearn import metrics
from sklearn.mixture import GaussianMixture
import re
from tqdm import tqdm

def load_taxonomy_mapping(taxonomy_file):
    
    acc_to_genus = {}
    
    with open(taxonomy_file, 'r') as f:
        next(f)  
        for line in tqdm(f, desc="Reading taxonomy file"):
            accession, taxid, rank, genus = line.strip().split('\t')
            acc_to_genus[accession] = genus
            
    print(f"Loaded {len(acc_to_genus)} taxonomy mappings")
    return acc_to_genus

def extract_accession(header):
    
    match = re.match(r'(\w+\.\d+)', header)
    return match.group(1) if match else None

def create_ground_truth_clusters(fasta_path, taxonomy_file):
    
    acc_to_genus = load_taxonomy_mapping(taxonomy_file)
    
    print("Now reading fasta")
    genus_clusters = defaultdict(list)
    sequence_to_genus = {}
    
    total_sequences = sum(1 for _ in SeqIO.parse(fasta_path, 'fasta'))
    
    unmapped_sequences = []
    for idx, record in tqdm(enumerate(SeqIO.parse(fasta_path, 'fasta')), 
                           total=total_sequences, 
                           desc="Processing sequences"):
        accession = extract_accession(record.description)  
        
        if accession and accession in acc_to_genus:
            genus = acc_to_genus[accession]
            genus_clusters[genus].append(idx)
            sequence_to_genus[idx] = genus
        else:
            unmapped_sequences.append((idx, record.description, accession))
    
    print(f"\nFound {len(genus_clusters)} unique genera")
    if unmapped_sequences:
        print(f"\nWarning: {len(unmapped_sequences)} sequences could not be mapped to a genus")
    
    genus_to_label = {genus: idx for idx, genus in enumerate(genus_clusters.keys())}
    ground_truth_labels = np.zeros(len(sequence_to_genus), dtype=int)
    for idx, genus in sequence_to_genus.items():
        ground_truth_labels[idx] = genus_to_label[genus]
    
    return ground_truth_labels, genus_clusters, genus_to_label

def analyze_clusters(embeddings_path, labels_path, n_clusters=200):
    
    embeddings = torch.load(embeddings_path)
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)
    
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()
    
    gmm = GaussianMixture(n_components=n_clusters, n_init=20, random_state=42)
    cluster_assignments = gmm.fit_predict(embeddings)
    
    cluster_contents = defaultdict(list)
    for idx, (cluster_id, label) in tqdm(enumerate(zip(cluster_assignments, labels)), 
                                       total=len(labels),
                                       desc="Assigning clusters"):
        cluster_contents[int(cluster_id)].append({
            'sequence_idx': idx,
            'label': label,
            'cluster': cluster_id
        })
    
    print("Calculating cluster statistics...")
    cluster_stats = {}
    label_distribution = defaultdict(lambda: defaultdict(int))
    
    for cluster_id in tqdm(range(n_clusters), desc="Processing clusters"):
        sequences = cluster_contents[cluster_id]
        cluster_stats[cluster_id] = {
            'size': len(sequences),
            'sequences': sequences
        }
        
        cluster_labels = [seq['label'] for seq in sequences]
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            label_distribution[cluster_id][label] = count
    
    df_stats = pd.DataFrame.from_dict(cluster_stats, orient='index')
    df_distribution = pd.DataFrame.from_dict(dict(label_distribution), orient='index').fillna(0)
    
    return cluster_contents, df_stats, df_distribution

def evaluate_clustering(predicted_clusters, ground_truth_labels):

    metrics_dict = {
        'normalized_mutual_info': metrics.normalized_mutual_info_score(ground_truth_labels, predicted_clusters),
        'homogeneity': metrics.homogeneity_score(ground_truth_labels, predicted_clusters),
        'completeness': metrics.completeness_score(ground_truth_labels, predicted_clusters),
        'v_measure': metrics.v_measure_score(ground_truth_labels, predicted_clusters)
    }
    
    return metrics_dict

def analyze_cluster_purity(cluster_contents, ground_truth_labels, genus_to_label):

    cluster_purity = {}
    label_to_genus = {v: k for k, v in genus_to_label.items()}
    
    for cluster_id, sequences in tqdm(cluster_contents.items(), desc="Calculating cluster purity"):
        sequence_indices = [seq['sequence_idx'] for seq in sequences]
        cluster_genera = [label_to_genus[ground_truth_labels[idx]] for idx in sequence_indices]
        
        genus_counts = pd.Series(cluster_genera).value_counts()
        total_sequences = len(sequence_indices)
        
        cluster_purity[cluster_id] = {
            'size': total_sequences,
            'dominant_genus': genus_counts.index[0],
            'dominant_genus_count': genus_counts.iloc[0],
            'purity': genus_counts.iloc[0] / total_sequences,
            'genus_distribution': genus_counts.to_dict()
        }
    
    return pd.DataFrame.from_dict(cluster_purity, orient='index')

if __name__ == "__main__":    
    fasta_path = "sampled_fasta.fna"
    taxonomy_file = "sequence_taxonomy.txt"
    embeddings_path = "final_vectors.pt"
    labels_path = "final_labels.pkl"
    
    ground_truth_labels, genus_clusters, genus_to_label = create_ground_truth_clusters(
        fasta_path, taxonomy_file
    )
    
    cluster_contents, cluster_stats, label_distribution = analyze_clusters(
        embeddings_path, 
        labels_path,
        n_clusters=766
    )
    
    predicted_clusters = np.array([seq['cluster'] for sequences in cluster_contents.values() 
                                 for seq in sequences])
    
    evaluation_metrics = evaluate_clustering(predicted_clusters, ground_truth_labels)
    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value:.3f}")
    
    purity_analysis = analyze_cluster_purity(cluster_contents, ground_truth_labels, genus_to_label)
    print(purity_analysis.describe())
    
    purity_analysis.to_csv("cluster_purity_analysis.csv")
    
