"""
"""
import numpy as np
from typing import List


def pad_sequences(sequences: List[np.ndarray], max_length: int=None):
    """Pad each individual sequence of the given list of sequences with zeros.
    
    Parameters
    ----------
    squences : list or numpy.ndarray
        List of numpy arrays of shape (n_timestamps, n_features).
    max_length : int, optional, default: None
         Desired length of each sequence.
        
    Returns
    -------
    numpy.ndarray
        Padded sequences as array.
    """
    if max_length is None:
        max_length = max([len(sq) for sq in sequences])
    
    for i, fsq in enumerate(sequences):
        if len(fsq) != max_length:
            dim = (max_length - len(fsq), fsq.shape[1])
            padding = np.zeros(dim)
            padded = np.vstack((fsq, padding))

            sequences[i] = padded
    
    return np.stack(sequences)