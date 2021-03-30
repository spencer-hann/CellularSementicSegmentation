import torch

from torchvision import transforms

from .Augmentation import pipeline


class SaltAndPepper(torch.nn.Module):
    def __init__(self, salt_prob, pepper_prob, salt=1.0, pepper=0.0):
        self.salt_prob = salt_prob
        self.pep_prob = pepper_prob
        self.salt = salt
        self.pepper = pepper

    def forward(self, T: torch.Tensor):
        raise NotImplementedError

    @staticmethod
    def make_pipeline(self, base_pipeline=pipeline):
        return transforms.Compose((
            SaltAndPepper(.2, .2),
            pipeline,
        ))

