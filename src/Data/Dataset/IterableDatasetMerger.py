import torch

import random

from torch.utils.data import IterableDataset
from itertools import accumulate

from typing import List


class IterableDatasetMerger(IterableDataset):
    def __init__(
        self,
        datasets: List[IterableDataset],
        weights: List[float] = None,
        cum_weights: List[float] = None,
    ):
        super().__init__()

        self.datasets = datasets

        if weights:
            weights = accumulate(weights)
        elif cum_weights:
            weights = cum_weights
        else:  # uniform probability
            weights = list(range(1, len(datasets)+1))

        sm = sum(weights)  # normalize
        weights = [w/sm for w in weights]

        self.weights = weights

    def __iter__(self):
        iterators = [iter(d) for d in self.datasets]
        while True:
            choice = random.choices(iterators, cum_weights=self.weights)[0]
            #print(choice, flush=True)
            try:
                yield next(choice)
            except StopIteration:
                #return None
                break

