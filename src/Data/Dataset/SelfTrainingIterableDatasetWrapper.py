from torch.utils.data import IterableDataset
from . import SelfTrainingDatasetWrapper


class SelfTrainingIterableDatasetWrapper(SelfTrainingDatasetWrapper, IterableDataset):
    def __iter__(self):
        for datum in self.dataset:
            yield self.wrap(datum)

