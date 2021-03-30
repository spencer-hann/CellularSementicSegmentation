from torch import utils, nn, Tensor
from typing import Tuple, Any, Dict, Optional, Callable


class GenerativeDataset(utils.data.IterableDataset):
    def __init__(
        self,
        generator: nn.Module,
        *args: Tuple[Any],
        **kwargs: Dict[Any, Any],
    ):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.generator = generator

    def __iter__(self):
        args, kwargs = self.args, self.kwargs
        while True:
            x = self.generator(*args, **kwargs)
            self.generator.zero_grad()
            yield x

