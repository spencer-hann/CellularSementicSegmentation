from __future__ import annotations

import torch
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple, Generator, Callable
    from Types import TensorLike


class CropGenerator:
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, img_batch):
        return CropGenerator.iter_crop(img_batch, *self.args, **self.kwargs)

    @staticmethod
    def iter_crop(
        a: TensorLike,
        shape: Tuple[int, int],
        overlap: float = 0.4,
        shuffle: bool = True,
        final_overlap: bool = True,
        validate_crop: Callable[[TensorLike], bool] = lambda _: True,
    ) -> Generator[TensorLike]:
        slices = CropGenerator.iter_slice(shape, a.shape, overlap, final_overlap)

        if shuffle:
            slices = list(slices)
            np.random.shuffle(slices)

        for x, y in slices:
            crop = a[..., x, y]  # ..., H, W  (i.e. NCHW)
            valid = validate_crop(crop)
            # validate crop may return new tensor
            if torch.is_tensor(valid) and valid.dtype is not torch.bool:
                yield valid
            # or simply return a boolean
            elif valid:
                yield crop

    @staticmethod
    def iter_slice(shape, outershape, overlap=.0, final_overlap=True):
        x_crop, y_crop = shape
        x_shift = int(x_crop * (1 - overlap))  # overlap to crops
        y_shift = int(y_crop * (1 - overlap))

        if not final_overlap:
            outershape = list(outershape)
            outershape[1] -= (outershape[1] % x_shift)
            outershape[2] -= (outershape[2] % y_shift)

        x = outershape[1] - x_crop
        y_start = outershape[2] - y_crop
        del outershape

        while x > 0:
            y = y_start
            while y > 0:
                yield slice(x, x+x_crop), slice(y, y+y_crop)
                y -= y_shift

            yield slice(x, x+x_crop), slice(0, y_crop)
            x-= x_shift

        y = y_start
        while y > 0:
            yield slice(0, x_crop), slice(y, y+y_crop)
            y -= y_shift

        yield slice(0, x_crop), slice(0, y_crop)

