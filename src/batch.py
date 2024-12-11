import pandas as pd
import numpy as np
import time

from torchrec import *
from torchrec.models import dlrm

import torch

from torchrec.datasets.utils import Batch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

class RecBatch:
    def __init__(
            self, data, cols_sparse,
            cols_dense, col_label, batch_size,
            num_generated_batches, seed, device
    ):
        """
        :param data: The data used for generating batches.
        :param cols_sparse: The list of column names representing sparse features.
        :param cols_dense: The list of column names representing dense features.
        :param col_label: The column name representing the label.
        :param batch_size: The size of each batch.
        :param num_generated_batches: The number of batches to be generated.
        :param seed: The seed used for random number generation.
        :param device: The device on which the tensors will be allocated (CPU or GPU).
        """

        self.data = data.reset_index(drop=True)
        self.cols_sparse = cols_sparse
        self.cols_dense = cols_dense
        self.col_label = col_label

        self.batch_size = batch_size
        self.num_generated_batches = num_generated_batches
        self.index = self.get_index(seed)
        self.device = device

        self.batches = [self._generate_batch(i) for i in range(len(self.index))]
        self.batch_index = 0

    def get_index(self, seed):
        """
        Generates the indices for creating batches.
        Depending on the value of num_generated_batches, it performs different operations:
        - If num_generated_batches is None, it calculates the number of batches based on batch_size and the size of the data.
        It then divides the data indices into sublists of size batch_size and returns the indices as a list.
        - If num_generated_batches has a specific value, it uses NumPy's random number generator to sample indices from the data.
          The sampling is done with replacement to avoid out-of-bounds indices.
        :param seed: for random number generation.
        :return: lists of indices of size batch_size.
        """

        data_size = self.data.shape[0]
        np.random.seed(seed=seed)

        if self.num_generated_batches is None:
            n = data_size // self.batch_size
            r = data_size % self.batch_size
            index = list(np.array(range(n * self.batch_size)).reshape(-1, self.batch_size))
            if r > 0:
                index.append(list(range(n * self.batch_size, n * self.batch_size + r)))
            self.num_generated_batches = len(index)
            print(f"Number of generated batches: {self.num_generated_batches}")
        else:
            # Always sample with replacement to prevent IndexError
            print('replace True')
            index = np.random.choice(
                range(data_size), self.batch_size * self.num_generated_batches, replace=True
            ).reshape(-1, self.batch_size)
            index = list(index)
            print(f"Number of generated batches: {self.num_generated_batches}")

        return index

    def _generate_batch(self, i):
        """
        Generate a single batch based on a given index.
        :param i: index.
        :return: batch.
        """
        batch_indices = self.index[i]
        if max(batch_indices) >= self.data.shape[0]:
            raise ValueError(f"Invalid index in batch {i}: max index is {max(batch_indices)}, but data only has {self.data.shape[0]} rows.")
        # Use positional indices with iloc
        sample = self.data.iloc[batch_indices].copy()

        # Concatenate all sparse feature lists
        values = []
        lengths = []
        for feat in self.cols_sparse:
            encoded_feat = sample[feat].tolist()
            # Flatten the list
            flat_feat = [item for sublist in encoded_feat for item in sublist]
            values.extend(flat_feat)
            # Record lengths for KeyedJaggedTensor
            lengths.append([len(sublist) for sublist in encoded_feat])

        # Flatten lengths for all features
        lengths_flat = [length for feat_lengths in lengths for length in feat_lengths]

        values_tensor = torch.tensor(values, dtype=torch.int64).to(self.device)
        lengths_tensor = torch.tensor(lengths_flat, dtype=torch.int32).to(self.device)

        dense_features = torch.tensor(sample[self.cols_dense].values, dtype=torch.float32).to(self.device)
        labels = torch.tensor(sample[self.col_label].values, dtype=torch.float32).to(self.device)
        if torch.isnan(dense_features).any():
            raise ValueError(f"NaN values found in dense features for batch {i}.")
        if torch.isnan(labels).any():
            raise ValueError(f"NaN values found in labels for batch {i}.")

        return values_tensor, lengths_tensor, dense_features, labels

def build_batch(batched_iterator, keys):
    """
    Builds a batch object from the given batched iterator.

    Args:
        batched_iterator (tuple): A tuple containing values, lengths, dense_features, and labels.
        keys (list): A list of keys representing the sparse feature names.

    Returns:
        Batch: A Batch object containing the batched data.
    """
    values, lengths, dense_features, labels = batched_iterator

    # Construct KeyedJaggedTensor for sparse features
    sparse_features = KeyedJaggedTensor.from_lengths_sync(
        keys=[k + '_enc' for k in keys],
        values=values,
        lengths=lengths
    )

    # Create a Batch object with the batched data
    batch = Batch(
        dense_features=dense_features,
        sparse_features=sparse_features,
        labels=labels,
    )
    if torch.isnan(batch.sparse_features.values()).any():
        raise ValueError("NaN values found in sparse features.")
    return batch