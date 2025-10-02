import requests
from io import StringIO
from IPython.display import display
from random import shuffle
from itertools import combinations
import numpy as np

def shingle(text: str, k: int)->set:
    """
    Create a set of 'shingles' from the input text using k-shingling.

    Parameters:
        text (str): The input text to be converted into shingles.
        k (int): The length of the shingles (substring size).

    Returns:
        set: A set containing the shingles extracted from the input text.
    """
    shingle_set = []
    for i in range(len(text) - k+1):
        shingle_set.append(text[i:i+k])
    return set(shingle_set)
def build_vocab(shingle_sets: list)->dict:
    """
    Constructs a vocabulary dictionary from a list of shingle sets.

    This function takes a list of shingle sets and creates a unified vocabulary
    dictionary. Each unique shingle across all sets is assigned a unique integer
    identifier.

    Parameters:
    - shingle_sets (list of set): A list containing sets of shingles.

    Returns:
    - dict: A vocabulary dictionary where keys are the unique shingles and values
      are their corresponding unique integer identifiers.

    Example:
    sets = [{"apple", "banana"}, {"banana", "cherry"}]
    build_vocab(sets)
    {'apple': 0, 'cherry': 1, 'banana': 2}  # The exact order might vary due to set behavior
    """
    full_set = {item for set_ in shingle_sets for item in set_}
    vocab = {}
    for i, shingle in enumerate(list(full_set)):
        vocab[shingle] = i
    return vocab
def one_hot(shingles: set, vocab: dict):
    vec = np.zeros(len(vocab))
    for shingle in shingles:
        idx = vocab[shingle]
        vec[idx] = 1
    return vec


def get_minhash_arr(num_hashes:int,vocab:dict):
    """
    Generates a MinHash array for the given vocabulary.

    This function creates an array where each row represents a hash function and
    each column corresponds to a word in the vocabulary. The values are permutations
    of integers representing the hashed value of each word for that particular hash function.

    Parameters:
    - num_hashes (int): The number of hash functions (rows) to generate for the MinHash array.
    - vocab (dict): The vocabulary where keys are words and values can be any data
      (only keys are used in this function).

    Returns:
    - np.ndarray: The generated MinHash array with `num_hashes` rows and columns equal
      to the size of the vocabulary. Each cell contains the hashed value of the corresponding
      word for the respective hash function.

    Example:
    vocab = {'apple': 1, 'banana': 2}
    get_minhash_arr(2, vocab)
    # Possible output:
    # array([[1, 2],
    #        [2, 1]])
    """
    length = len(vocab.keys())
    arr = np.zeros((num_hashes,length))
    for i in range(num_hashes):
        permutation = np.random.permutation(len(vocab.keys())) + 1
        arr[i,:] = permutation.copy()
    return arr.astype(int)
def get_signature(minhash:np.ndarray, vector:np.ndarray):
    """
    Computes the signature of a given vector using the provided MinHash matrix.

    The function finds the nonzero indices of the vector, extracts the corresponding
    columns from the MinHash matrix, and computes the signature as the minimum value
    across those columns for each row of the MinHash matrix.

    Parameters:
    - minhash (np.ndarray): The MinHash matrix where each column represents a shingle
      and each row represents a hash function.
    - vector (np.ndarray): A vector representing the presence (non-zero values) or
      absence (zero values) of shingles.

    Returns:
    - np.ndarray: The signature vector derived from the MinHash matrix for the provided vector.

    Example:
    minhash = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])
    vector = np.array([0, 1, 0])
    get_signature(minhash, vector)
    output:array([3, 6, 9])
    """
    idx = np.nonzero(vector)[0].tolist()
    shingles = minhash[:,idx]
    signature = np.min(shingles,axis=1)
    return signature

def jaccard_similarity(set1, set2):
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))
    return intersection_size / union_size if union_size != 0 else 0.0

def compute_signature_similarity(signature_1, signature_2):
    """
    Calculate the similarity between two signature matrices using MinHash.

    Parameters:
    - signature_1: First signature matrix as a numpy array.
    - signature_matrix2: Second signature matrix as a numpy array.

    Returns:
    - Estimated Jaccard similarity.
    """
    # Ensure the matrices have the same shape
    if signature_1.shape != signature_2.shape:
        raise ValueError("Both signature matrices must have the same shape.")
    # Count the number of rows where the two matrices agree
    agreement_count = np.sum(signature_1 == signature_2)
    # Calculate the similarity
    similarity = agreement_count / signature_2.shape[0]

    return similarity



class LSH:
    """
    Implements the Locality Sensitive Hashing (LSH) technique for approximate
    nearest neighbor search.
    """
    buckets = []
    counter = 0

    def __init__(self, b: int):
        """
        Initializes the LSH instance with a specified number of bands.

        Parameters:
        - b (int): The number of bands to divide the signature into.
        """
        self.b = b
        for i in range(b):
            self.buckets.append({})

    def make_subvecs(self, signature: np.ndarray) -> np.ndarray:
        """
        Divides a given signature into subvectors based on the number of bands.

        Parameters:
        - signature (np.ndarray): The MinHash signature to be divided.

        Returns:
        - np.ndarray: A stacked array where each row is a subvector of the signature.
        """
        l = len(signature)
        assert l % self.b == 0
        r = int(l / self.b)
        subvecs = []
        for i in range(0, l, r):
            subvecs.append(signature[i:i+r])
        return np.stack(subvecs)

    def add_hash(self, signature: np.ndarray):
        """
        Adds a signature to the appropriate LSH buckets based on its subvectors.

        Parameters:
        - signature (np.ndarray): The MinHash signature to be hashed and added.
        """
        subvecs = self.make_subvecs(signature).astype(str)
        for i, subvec in enumerate(subvecs):
            subvec = ','.join(subvec)
            if subvec not in self.buckets[i].keys():
                self.buckets[i][subvec] = []
            self.buckets[i][subvec].append(self.counter)
        self.counter += 1

    def check_candidates(self) -> set:
        """
        Identifies candidate pairs from the LSH buckets that could be potential near duplicates.

        Returns:
        - set: A set of tuple pairs representing the indices of candidate signatures.
        """
        candidates = []
        for bucket_band in self.buckets:
            keys = bucket_band.keys()
            for bucket in keys:
                hits = bucket_band[bucket]
                if len(hits) > 1:
                    candidates.extend(combinations(hits, 2))
        return set(candidates)