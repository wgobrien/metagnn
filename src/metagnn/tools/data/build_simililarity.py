import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
import re

file_path = 'small/distance_matrix_small.txt'
data = []

import os
num_cores = os.cpu_count()
with open(file_path, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        seq1, seq2, mash_dist, p_value, shared_fraction = parts
        data.append((seq1, seq2, shared_fraction))

def parse_shared_fraction(entry):
    seq1, seq2, shared_fraction = entry
    shared_fraction_value = float(Fraction(shared_fraction))
    return seq1, seq2, shared_fraction_value

with ThreadPoolExecutor(max_workers=32) as executor:
    parsed_data = list(executor.map(parse_shared_fraction, data))

df = pd.DataFrame(parsed_data, columns=['Sequence1', 'Sequence2', 'SharedFraction'])

sequences = pd.unique(df[['Sequence1', 'Sequence2']].values.ravel())
seq_to_idx = {seq: idx for idx, seq in enumerate(sequences)}

n = len(sequences)
similarity_matrix = np.zeros((n, n))

def populate_matrix(row):
    i, j = seq_to_idx[row['Sequence1']], seq_to_idx[row['Sequence2']]
    return i, j, row['SharedFraction']

with ThreadPoolExecutor() as executor:
    results = list(executor.map(populate_matrix, [row for _, row in df.iterrows()]))

for i, j, value in results:
    similarity_matrix[i, j] = value
    similarity_matrix[j, i] = value  

scaler = MinMaxScaler()
similarity_matrix_normalized = scaler.fit_transform(similarity_matrix)

pca = PCA(n_components=3)
pca_results = pca.fit_transform(similarity_matrix_normalized)

tsne = TSNE(n_components=3, perplexity=5, random_state=42)
tsne_results = tsne.fit_transform(similarity_matrix_normalized)

fig = plt.figure(figsize=(30, 20))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(pca_results[:, 0], pca_results[:, 1], pca_results[:, 2], c='blue', alpha=0.7)
ax1.set_title("PCA Projection")
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
ax1.set_zlabel("PC3")

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c='green', alpha=0.7)
ax2.set_title("t-SNE Projection")
ax2.set_xlabel("Dim1")
ax2.set_ylabel("Dim2")
ax2.set_zlabel("Dim3")

plt.tight_layout()

plt.show()
plt.savefig("small/3D_multi.png")

similarity_df = pd.DataFrame(similarity_matrix_normalized, index=sequences, columns=sequences)
similarity_df.to_csv('small/similarity_matrix_small.csv')
