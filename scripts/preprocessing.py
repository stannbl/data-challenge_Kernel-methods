import numpy as np


def one_hot_encode(sequence, max_length):
    """Convert a DNA sequence into a one-hot encoded vector of fixed length."""
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}

    encoded = [mapping.get(base, [0, 0, 0, 0]) for base in sequence]  # Encode each base

    # Pad or truncate to max_length
    while len(encoded) < max_length:
        encoded.append([0, 0, 0, 0])  # Pad with zeros
    encoded = encoded[:max_length]  # Truncate if too long

    return np.array(encoded).flatten()  # Convert to 1D vector


def transform_sequences(X, max_length):
    """Transform all DNA sequences into feature vectors of the same length."""
    transformed = np.array([one_hot_encode(seq, max_length) for seq in X])
    return transformed
