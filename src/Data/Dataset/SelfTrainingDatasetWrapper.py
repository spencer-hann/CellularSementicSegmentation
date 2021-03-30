import torch

from torch import nn, Tensor
from torch.utils.data import Dataset

from typing import Iterable, Optional, Callable, List, Tuple


class SelfTrainingDatasetWrapper(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        models: Iterable[nn.Module],
        add_original: bool = False,
        split: Optional[Callable[[Tensor], Tuple[Tensor,Tensor]]] = None,
        preproc: Optional[Callable[[Tensor], Tensor]] = None,
        postproc: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        super().__init__()

        self.models = models
        self.dataset = dataset

        if split is None:
            split = lambda t: (t[:1], t[1:])
        if preproc is None:
            preproc = lambda t: t
        if postproc is None:
            postproc = lambda t: t.mean(dim=0)

        self.add_original = add_original

        self.split = split
        self.preproc = preproc
        self.postproc = postproc

    def wrap(self, inp: Tensor) -> Tensor:
        inp, orig = self.split(inp)
        out = self.preproc(inp)
        if len(out.shape) < 4:
            out = out[None]  # add batch dim
        out = [m(out)[0] for m in self.models]
        for m in self.models: m.zero_grad()
        if self.add_original:
            out = torch.stack((*out, orig))
        else:
            out = torch.stack(out)
            #out += orig
            #out.clip_(0, 1)
        out = self.postproc(out)
        out = torch.cat((inp, out))
        return out.detach()

    def __getitem__(self, index: int):
        return self.wrap(self.dataset[index])

    def __len__(self):
        return len(self.dataset)

