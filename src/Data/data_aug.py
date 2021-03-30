import torch
import torchvision

import random
import PIL

from torchvision import transforms

from .config import raw_image_height, raw_image_width


class GaussianNoise(torch.nn.Module):
    def __init__(self, mu=0.0, sigma=1.0, clip_vals=(0., 1.)):
        super().__init__()
        self.mu = mu
        if isinstance(sigma, (float, int)):
            sigma = (sigma - .0001, sigma)
        self.sigma = sigma
        self.clip_vals = clip_vals

    def forward(self, t):
        sigma = random.uniform(*self.sigma)
        t += self.mu + torch.randn_like(t) * sigma
        t.clip_(*self.clip_vals)
        return t


def random_apply(transform_list, p=0.5):
    trasform_list = torch.nn.ModuleList(transform_list)
    return transforms.RandomApply(transform_list, p)


flip_pipeline = transforms.Compose((
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
))


def scale_pipeline(scale=0.5):
    h = int(raw_image_height*scale)
    w = int(raw_image_width*scale)
    return transforms.Resize((h, w))


def flip_scale_pipeline(scale=0.5):
    return transforms.Compose((scale_pipeline(scale), flip_pipeline,))


def noise_pipeline(p, sigma):
    return random_apply([
            transforms.GaussianBlur(3, sigma=sigma),
            GaussianNoise(0.0, (0.02, 0.08))
        ],
        p = p,
    )


def pipeline(
    scale=0.5, degrees=40, shear=5, translate=None, noise_p=.1, sigma=(.4,.6),
):
    return transforms.Compose((
        transforms.RandomAffine(
            degrees,
            translate=translate,
            scale=(.9, 1.1),
            shear=shear,
            resample=PIL.Image.BILINEAR,
        ),
        flip_scale_pipeline(scale),
        noise_pipeline(noise_p, sigma),
    ))

