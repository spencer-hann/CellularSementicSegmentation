from __future__ import annotations

import torch
import numpy as np

from torchvision import transforms
from PIL import Image as PIL_Image

from . import dirs
from .config import DTYPE
from ..Device import device

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union
    from pathlib import Path
    from ..Types import PathLike, TensorLike


TEST_INDICES = set(range(204, 305+1))


def itostr(index: Union[int, str]):
    return str(index).zfill(4)


def fname_from_index(index: Union[int, str]) -> str:
    index = itostr(index)
    return index + ".pt"


def np_from_path(path: PathLike, dtype=DTYPE, **kwargs):
    #with silent_logger(logging.INFO):
    #    return np.array(PIL_Image.open(path), dtype=dtype, **kwargs)[:,:,None]
    return np.array(PIL_Image.open(path))


def tensor_from_path(path: PathLike, dtype=DTYPE, device=device, **kwargs):
    #with silent_logger(logging.INFO):
    return transforms.functional.to_tensor(PIL_Image.open(path)).to(device)


def plottable(arr: TensorLike):
    # TODO:  account for color imgs too
    return arr[0,:,:] if arr.ndim == 3 else arr[0,0,:,:]

