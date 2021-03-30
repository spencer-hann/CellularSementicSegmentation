import torch
import torchvision
import numpy as np

import random
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch import nn, optim

import style_main

from src.Device import device
from src.Data import dirs, Dataset
from src.Data.highlighted_image import highlighted_image

#from src.Models import LayerInfo; LayerInfo.on()


highlighted_image.RGB_override = True


##########################
##   Dataset Creation   ##
##########################

def create_celeba_dataset(size=128, batch_size=32,):
    celeba = torchvision.datasets.CelebA(
        '/home/spencer/data/',
        target_type="identity",
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda t: t.to(device)),
            torchvision.transforms.Resize(size),
            torchvision.transforms.CenterCrop(size),
            torchvision.transforms.RandomHorizontalFlip(0.5),
        ]),
    )

    print("n CelebA images", len(celeba))

    # only take image--label not needed.  shift into tanh range
    celeba = Dataset.DatasetFilter(celeba, lambda t: t[0])

    return torch.utils.data.DataLoader(
        celeba, batch_size=batch_size, shuffle=True,
    )


#######################
##       Main        ##
#######################

if __name__ == "__main__":

    path = dirs.models / "CelebAGenerator.pt"
    channels = 3

    G, optimG = style_main.init_generator(channels, path=path)
    D, optimD = style_main.init_discriminator(channels)

    print(G, '\n')
    print(D, '\n')

    images = create_celeba_dataset()

    style_main.train(D, G, optimD, optimG, images, "CelebA")

    print("exit.")

