import torch

from torch import nn, Tensor
from torch.utils.data import Dataset

from typing import Iterable, Optional, Callable, List, Tuple


class DatasetFilter(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        validate: Callable[[Tensor], Optional[Tensor]],
    ):
        super().__init__()

        self.dataset = dataset
        self.validate = validate

    def __getitem__(self, idx):
        return self.validate(self.dataset[idx])

    @property
    def __len__(self):
        return self.dataset.__len__

