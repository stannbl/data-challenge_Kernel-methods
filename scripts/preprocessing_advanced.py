import numpy as np


def reverse_complement(sequence):
    """Compute the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

    # ✅ Filter out invalid bases (keep only valid DNA characters)
    clean_sequence = "".join(base if base in complement else "N" for base in sequence)

    return "".join(complement.get(base, "N") for base in reversed(clean_sequence))  # 'N' handles unknown bases


def extract_kmers(sequence, k):
    """Extract canonical k-mers (lexicographically smaller between k-mer and its reverse complement)."""
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        rev_kmer = reverse_complement(kmer)
        canonical_kmer = min(kmer, rev_kmer)  # Keep only the lexicographically smaller one
        kmers.append(canonical_kmer)
    return kmers

def transform_sequences(X, k):
    """Transform sequences using canonical k-mers without gaps."""
    transformed = np.array([extract_kmers(seq, k) for seq in X if len(seq) >= k], dtype=object)
    return transformed
