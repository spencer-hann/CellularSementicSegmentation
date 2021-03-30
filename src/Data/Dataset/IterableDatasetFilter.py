import torch

from torch import nn, Tensor
from torch.utils.data import IterableDataset

from .DatasetFilter import DatasetFilter

from typing import Iterable, Optional, Callable, List, Tuple


class IterableDatasetFilter(DatasetFilter, IterableDataset):
    def __iter__(self):
        validate = self.validate
        for datum in self.dataset:
            datum = validate(datum)
            if datum is None:
                continue
            yield datum

