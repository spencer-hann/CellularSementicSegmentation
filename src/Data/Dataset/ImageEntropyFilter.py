import torch

from ...Metrics.Segmentation import entropy


def ImageEntropyFilter(upperbound=0.0, inertia=.99):

    def image_entropy_filter(img):
        nonlocal upperbound

        e = entropy(img[1:,::4,::4]).cpu()

        if e > upperbound:
            img = None

        upperbound *= inertia
        upperbound += e * (1 - inertia)

        return img

    return image_entropy_filter

