import numpy as np
from collections import Counter
from itertools import combinations_with_replacement, product, combinations
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scripts.preprocessing_advanced import reverse_complement
from itertools import product, combinations
from collections import Counter


def linear_kernel(X1, X2):
    """Compute the Linear Kernel"""
    return np.dot(X1, X2.T)

def gaussian_kernel(X1, X2, gamma=0.1):
    """Compute the RBF (Gaussian) Kernel"""
    sq_dist = np.sum(X1**2, axis=1, keepdims=True) - 2 * np.dot(X1, X2.T) + np.sum(X2**2, axis=1)
    return np.exp(-gamma * sq_dist)

def rbf_kernel(X1, X2, gamma=0.1):
    """Compute the Gaussian (RBF) Kernel."""
    X1 = np.array(X1, dtype=np.float64)  # Convert to NumPy array (float)
    X2 = np.array(X2, dtype=np.float64)

    sq_dist = np.sum(X1**2, axis=1, keepdims=True) - 2 * np.dot(X1, X2.T) + np.sum(X2**2, axis=1)
    return np.exp(-gamma * sq_dist)



def extract_kmers(seq, k):
    """Extract k-mers from a DNA sequence and count occurrences."""
    if isinstance(seq, np.ndarray):  # Convert NumPy array to string
        seq = "".join(seq.astype(str))  # Ensure it's a string

    return Counter(["".join(seq[i:i+k]) for i in range(len(seq) - k + 1)])  # Convert k-mers to strings

def spectrum_kernel(X1, X2, k=3, normalize=True):
    """Compute the Spectrum Kernel efficiently."""
    K = np.zeros((len(X1), len(X2)))

    # Extract raw k-mer counts
    X1_kmers = [extract_kmers(seq, k) for seq in X1]
    X2_kmers = [extract_kmers(seq, k) for seq in X2]

    # Get unique k-mers
    unique_kmers = set(kmer for d in X1_kmers + X2_kmers for kmer in d.keys())
    kmer_to_index = {kmer: i for i, kmer in enumerate(unique_kmers)}

    # Convert k-mer dictionaries to feature vectors
    def vectorize(kmer_counts):
        vec = np.zeros(len(unique_kmers))
        for kmer, count in kmer_counts.items():
            vec[kmer_to_index[kmer]] = count
        return vec

    X1_vectors = np.array([vectorize(kmer_counts) for kmer_counts in X1_kmers])
    X2_vectors = np.array([vectorize(kmer_counts) for kmer_counts in X2_kmers])

    # Compute the kernel matrix
    K = np.dot(X1_vectors, X2_vectors.T)

    if normalize:
        K /= np.linalg.norm(K, ord='fro')  # Normalize using Frobenius norm

    return K


def polynomial_kernel(X1, X2, degree=3, gamma=1, coef0=1):
    """
    Compute the Polynomial Kernel.

    Args:
        X1: Matrix of shape (n_samples_1, n_features)
        X2: Matrix of shape (n_samples_2, n_features)
        degree: Degree of the polynomial kernel (default: 3)
        gamma: Scaling factor for the dot product (default: 1)
        coef0: Independent term (default: 1)

    Returns:
        Kernel matrix of shape (n_samples_1, n_samples_2)
    """
    return (gamma * np.dot(X1, X2.T) + coef0) ** degree


import numpy as np
import scipy.sparse as sp
from collections import Counter
from itertools import product, combinations

# Cache for mismatch neighborhoods
mismatch_dict_cache = {}

def precompute_mismatch_neighborhood(k, m):
    """Precompute mismatch neighborhoods for k-mers."""
    alphabet = ["A", "C", "G", "T"]
    all_kmers = ["".join(p) for p in product(alphabet, repeat=k)]

    if (k, m) in mismatch_dict_cache:
        return mismatch_dict_cache[(k, m)]  # Use cached values if already computed

    print(f"🔄 Precomputing mismatch neighborhoods for k={k}, m={m}...")

    mismatch_dict = {}

    for kmer in all_kmers:
        mismatch_set = {kmer}  # Include original k-mer
        if m > 0:  # Only generate mismatches if m > 0
            for positions in combinations(range(k), m):  # Select 'm' positions for mismatches
                for replacements in product(alphabet, repeat=m):  # Generate possible substitutions
                    new_kmer = list(kmer)
                    for pos, repl in zip(positions, replacements):
                        new_kmer[pos] = repl
                    mismatch_set.add("".join(new_kmer))
        mismatch_dict[kmer] = mismatch_set

    # Store the result in cache
    mismatch_dict_cache[(k, m)] = mismatch_dict
    return mismatch_dict

def extract_kmer_counts(sequence, k, m, mismatch_dict):
    """Extract k-mer counts allowing up to 'm' mismatches."""
    kmer_dict = Counter()
    sequence = str(sequence)  # Ensure input is a string

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if kmer in mismatch_dict:
            for mismatch_kmer in mismatch_dict[kmer]:
                kmer_dict[mismatch_kmer] += 1

    return kmer_dict

def vectorize_kmers(kmer_counts, kmer_to_index, vector_size, normalize=True):
    """Convert k-mer counts to sparse vector format with optional normalization."""
    indices = [kmer_to_index[k] for k in kmer_counts.keys() if k in kmer_to_index]
    values = np.array([kmer_counts[k] for k in kmer_counts.keys() if k in kmer_to_index], dtype=np.float64)

    if normalize:
        norm = np.linalg.norm(values)
        if norm > 1e-8:  # ✅ Avoid division by zero
            values /= norm
        else:
            values.fill(1.0 / len(values))  # ✅ Assign uniform weight to avoid empty vectors

    return sp.csr_matrix((values, (np.zeros(len(indices)), indices)), shape=(1, vector_size))


def mismatch_kernel(X1, X2, k=3, m=1, normalize=True, is_train=True):
    """Compute the Mismatch Kernel using optimized k-mer frequency representations."""

    # Step 1: Precompute mismatch neighborhoods
    if (k, m) not in mismatch_dict_cache:
        print(f"⚡ Computing mismatch kernel: k={k}, m={m}")
        mismatch_dict_cache[(k, m)] = precompute_mismatch_neighborhood(k, m)
    mismatch_dict = mismatch_dict_cache[(k, m)]

    # Step 2: Extract k-mer counts
    X1_kmers = [extract_kmer_counts(seq, k, m, mismatch_dict) for seq in X1]
    X2_kmers = [extract_kmer_counts(seq, k, m, mismatch_dict) for seq in X2]

    unique_kmers = set(kmer for d in X1_kmers + X2_kmers for kmer in d.keys())
    kmer_to_index = {kmer: i for i, kmer in enumerate(unique_kmers)}
    vector_size = len(unique_kmers)

    # Step 3: Convert k-mer counts to sparse vector format (✅ Normalization Applied Here)
    X1_vectors = sp.vstack([vectorize_kmers(kmer_counts, kmer_to_index, vector_size, normalize=True) for kmer_counts in X1_kmers])
    X2_vectors = sp.vstack([vectorize_kmers(kmer_counts, kmer_to_index, vector_size, normalize=True) for kmer_counts in X2_kmers])

    # Step 4: Compute dot product for kernel matrix
    K = X1_vectors @ X2_vectors.T

    # ✅ Convert sparse matrix to dense NumPy array if needed
    if sp.issparse(K):
        K = K.toarray()

    # ✅ Debugging Info
    print(f"🔎 Kernel matrix statistics:")
    print(f"   ➤ Mean: {np.mean(K):.5f}, Std: {np.std(K):.5f}")
    print(f"   ➤ Min: {np.min(K):.5f}, Max: {np.max(K):.5f}")
    if K.shape[0] == K.shape[1]:
        print(f"   ➤ Diagonal mean: {np.mean(np.diag(K)):.5f}")  # Self-similarity check
    else:
        print(f"   ➤ Non-square kernel (Validation/Test)")

    return K


def di_mismatch_score(kmer1, kmer2):
    """Compute the number of matching dinucleotides between two k-mers."""
    return sum((kmer1[i] == kmer2[i] and kmer1[i + 1] == kmer2[i + 1]) for i in range(len(kmer1) - 1))


def di_mismatch_kernel(X1, X2, k=3, m=1):
    """Compute the Di-Mismatch Kernel by allowing gapped k-mers."""
    K = np.zeros((len(X1), len(X2)))

    for i, seq1 in enumerate(X1):
        for j, seq2 in enumerate(X2):
            match_count = 0
            for idx in range(len(seq1) - k + 1):
                sub1 = seq1[idx:idx + k]
                sub2 = seq2[idx:idx + k]
                if sum(c1 != c2 for c1, c2 in zip(sub1, sub2)) <= m:
                    match_count += 1
            K[i, j] = match_count

    return K


def weighted_spectrum_kernel(X1, X2, k=3, weight_factor=0.5, normalize=True):
    """Compute the Weighted Spectrum Kernel by prioritizing rare k-mers."""
    K = np.zeros((len(X1), len(X2)))

    X1_kmers = [{kmer: count / sum(counts.values()) for kmer, count in counts.items()} for counts in
                [extract_kmers(seq, k) for seq in X1]]
    X2_kmers = [{kmer: count / sum(counts.values()) for kmer, count in counts.items()} for counts in
                [extract_kmers(seq, k) for seq in X2]]
    all_kmers = set(kmer for d in X1_kmers + X2_kmers for kmer in d.keys())
    kmer_frequencies = {kmer: sum(d.get(kmer, 0) for d in X1_kmers + X2_kmers) for kmer in all_kmers}

    for i, kmer_counts_1 in enumerate(X1_kmers):
        for j, kmer_counts_2 in enumerate(X2_kmers):
            K[i, j] = sum(
                (kmer_counts_1[kmer] * kmer_counts_2.get(kmer, 0)) / (kmer_frequencies[kmer] ** weight_factor)
                for kmer in kmer_counts_1
            )

    if normalize:
        K /= np.linalg.norm(K, ord='fro')  # Normalize by Frobenius norm
    return K


def hybrid_kernel(X1, X2, alpha=0.5, beta=0.3, gamma=0.2, k=6, m=2):
    """Compute a Hybrid Kernel combining spectrum, mismatch, and RBF kernels efficiently."""
    K_spectrum = weighted_spectrum_kernel(X1, X2, k=k)
    K_mismatch = mismatch_kernel(X1, X2, k=k, m=m)
    K_di_mismatch = di_mismatch_kernel(X1, X2, k=k, m=m)

    return alpha * K_spectrum + beta * K_mismatch + gamma * K_di_mismatch

