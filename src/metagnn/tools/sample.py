from Bio import SeqIO
import random
from collections import defaultdict
from tqdm import tqdm

def get_sequence_family_mapping(gbff_file):
    id_to_family = {}
    
    for record in tqdm(SeqIO.parse(gbff_file, "genbank")):
        try:
            seq_id = record.id
            
            taxonomy = record.annotations.get('taxonomy', [])
            
            family = next((tax for tax in taxonomy if tax.lower().endswith('viridae')), None)
            
            if family:
                id_to_family[seq_id] = family
                
        except Exception as e:
            print(f"Errorrr")
    
    return id_to_family

def count_families(id_to_family):
    family_counts = defaultdict(int)
    for family in id_to_family.values():
        family_counts[family] += 1
    return family_counts

def sample_sequences(fasta_file, id_to_family, min_count=100, max_samples=100):
    family_counts = count_families(id_to_family)
    
    eligible_families = {family: count for family, count in family_counts.items() 
                        if count >= min_count}
    
    print(f"\nFound {len(eligible_families)} families with {min_count}+ sequences")
    
    sequences_by_family = defaultdict(list)
    
    for record in tqdm(SeqIO.parse(fasta_file, "fasta")):
        family = id_to_family.get(record.id)
        if family in eligible_families:
            sequences_by_family[family].append(record)
    
    sampled_records = []
    labels = []
    
    for family in tqdm(eligible_families):
        family_sequences = sequences_by_family[family]
        sampled = random.sample(family_sequences, 
                              min(max_samples, len(family_sequences)))
        
        sampled_records.extend(sampled)
        labels.extend([family] * len(sampled))
    
    return sampled_records, labels

def main():
    gbff_file = "/u/flashscratch/ophoff/seungmo6/test_gnn2/new_real/dataset/refseq_viral_sequences/viral.1.genomic.gbff"
    fasta_file = "/u/flashscratch/ophoff/seungmo6/test_gnn2/new_real/dataset/refseq_viral_sequences/viral.1.1.genomic.fna"
    output_fasta = "sampled_fasta.fna"
    output_labels = "labels.txt"
    
    id_to_family = get_sequence_family_mapping(gbff_file)
    
    sampled_records, labels = sample_sequences(fasta_file, id_to_family)
    
    print(f"\nWriting {len(sampled_records)} sampled sequences to {output_fasta}!")
    SeqIO.write(sampled_records, output_fasta, "fasta")
    
    print(f"Writing labels to {output_labels}!")
    with open(output_labels, 'w') as f:
        for record, label in zip(sampled_records, labels):
            f.write(f"{record.id}\t{label}\n")
    
    print("\nSampling Summary:")
    print("-" * 50)
    family_counts = defaultdict(int)
    for label in labels:
        family_counts[label] += 1
    
    print(f"{'Family':<30} {'Sampled Count':>15}")
    print("-" * 50)
    for family, count in sorted(family_counts.items()):
        print(f"{family:<30} {count:>15}")
    print("-" * 50)

if __name__ == "__main__":
    random.seed(42)  
    main()