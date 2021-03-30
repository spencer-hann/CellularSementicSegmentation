import torch
import random

from typing import Iterable, Union
from pathlib import Path

from ..data import itostr, fname_from_index
from ..config import raw_image_size
from ...Device import device


class ImageIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        folders,
        transform,
        cropper,
        n_streams,
        min_streams=None,
        name=None,
        shuffle=True,
        indices=None,
    ):
        super().__init__()

        if isinstance(folders, Path):
            folders = (folders,)
        self.folders = folders
        self.transform = transform if transform is not None else lambda t: t
        self.cropper = cropper
        self.n_streams = n_streams
        self.min_streams = None
        self.shuffle = shuffle

        self.indices = indices
        if indices is None:
            self.indices = [*range(0, 205), *range(306, 2100)]

        if name is None:
            self.name = type(self).__name__
        else:
            self.name = name

    def __iter__(self):
        streams = self.make_streams()
        take = streams.take()

        while take is not None:
            yield take
            take = streams.take()

    def make_streams(self):
        # skip test range
        if self.shuffle:
            random.shuffle(self.indices)
        river = map(self.stack_by_index, self.indices)
        river = map(self.transform, river)
        river = map(self.cropper, river)
        streams = ImageStreamContainer(river, self.n_streams, self.min_streams)
        return streams

    def stack_by_index(
        self, index: Union[int, str], dtype: torch.dtype = torch.float,
    ) -> torch.Tensor:
        base_name = fname_from_index(index)

        def safe_load(folder):
            path = folder / base_name
            if path.exists():
                return torch.load(path).to(dtype).to(device)
            return torch.zeros((1, *raw_image_size), dtype=dtype, device=device)

        stack = [safe_load(f) for f in self.folders]

        return torch.cat(stack)


class ImageStreamContainer(list):
    def __init__(self, source, n_streams, min_streams=None):
        self.source = source
        while len(self) < n_streams:
            self.append(next(self.source))

        if min_streams is None:
            min_streams = max(1, n_streams // 2)
        self.min_streams = min_streams

    def take(self):
        if not self:  # empty
            return None

        i = random.randrange(len(self))
        try:
            item = next(self[i])
        except StopIteration:
            self.new_stream(i)
            item = self.take()

        return item

    def new_stream(self, i):
        del self[i]
        #print("re-sourcing stream #", i)
        #print("n streams", len(self))
        try:
            self.insert(i, next(self.source))
        except StopIteration:
            if len(self) < self.min_streams:
                self.clear()

