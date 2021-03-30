import torch

from torch import Tensor

from typing import Callable, Union


Float = Union[Tensor, float]
MetricFn = Callable[[Tensor, Tensor], Tensor]


def entropy(t: Tensor, epsilon: Float = 1e-7) -> Tensor:
    clipped = t.clip(epsilon, 1)
    return -(t * torch.log(clipped)).sum()


def _not(binary_tensor: Tensor) -> Tensor:
    return 1 - binary_tensor


def WeightedIOU(
    tp: Float = 1.0, fp: Float = 1.0, fn: Float = 1.0,
) -> MetricFn:

    tp_weight = tp
    fp_weight = fp
    fn_weight = fn

    def iou(inp: Tensor, target: Tensor) -> Tensor:
        tp = inp * target
        tp = tp.sum()
        tp *= tp_weight

        fp = inp * _not(target)
        fp = fp.sum()
        fp *= fp_weight

        fn = _not(inp) * target
        fn = fn.sum()
        fn *= fn_weight

        result = tp / (tp + fp + fn)
        if result.isnan():
            return torch.as_tensor(1.0, device=inp.device)
        return result

    return iou


def DiceCoefficient() -> MetricFn:
    return WeightedIOU(tp=2.0, fp=1.0, fn=1.0)


def IOU() -> MetricFn:
    return WeightedIOU(tp=1.0, fp=1.0, fn=1.0)


JaccardIndex = IOU


def Precision() -> MetricFn:
    return WeightedIOU(tp=1.0, fp=1.0, fn=0.0)


def Recall() -> MetricFn:
    return WeightedIOU(tp=1.0, fp=0.0, fn=1.0)

