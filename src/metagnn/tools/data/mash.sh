#!/bin/bash

INPUT_FASTA="complete_genomes.fasta"

#make sure to activate mash_env
SKETCH_FILE="sequences_16.msh"
DISTANCE_MATRIX="distance_matrix_16.txt"

KMER_SIZE=16
SKETCH_SIZE=10000

check_error() {
    if [ $? -ne 0 ]; then
        echo "Error: $1"
        exit 1
    fi
}

if [ ! -f "$INPUT_FASTA" ]; then
    echo "Error: Input file $INPUT_FASTA not found!"
    exit 1
fi

seq_count=$(grep -c "^>" "$INPUT_FASTA")
echo "Found $seq_count sequences in $INPUT_FASTA"

if [ $seq_count -lt 2 ]; then
    echo "Error: Need at least 2 sequences to calculate distances"
    exit 1
fi

echo "Creating sketch file..."
mash sketch -i -p 224 -k $KMER_SIZE -s $SKETCH_SIZE -o ${SKETCH_FILE%.*} $INPUT_FASTA
check_error "Failed to create sketch file"

echo "Calculating pairwise distances..."
mash dist -p 224 $SKETCH_FILE $SKETCH_FILE > $DISTANCE_MATRIX
check_error "Failed to calculate distances"

echo "Done Done!"
