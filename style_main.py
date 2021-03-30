import torch
import numpy as np

import random
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim

import generator_viewer
from src.Data import data, dirs, Dataset, data_aug
from src.Data.highlighted_image import highlighted_image, view_generated_images
from src.Training import GANs
from src.Models.GANs.Vanilla import Discriminator


#from src.Models import LayerInfo; LayerInfo.on()


highlighted_image.RGB_override = False


#############################
##   Model Initialization  ##
#############################

def init_discriminator(
    channels, lr=1e-3, size_scale=3, dropout_p=.2, weight_decay=0.1,
):
    D = (
        Discriminator(channels, size_scale=size_scale, dropout_p=dropout_p   , ),
        Discriminator(channels, size_scale=size_scale, dropout_p=dropout_p+.1, ),
        Discriminator(channels, size_scale=size_scale, dropout_p=dropout_p+.2, ),
    )

    p = (d.parameters() for d in D)
    optimD = (
        optim.AdamW(next(p), lr, betas=(.5,0.999), weight_decay=weight_decay),
        optim.AdamW(next(p), lr, betas=(.5,0.999), weight_decay=weight_decay),
        optim.AdamW(next(p), lr, betas=(.5,0.999), weight_decay=weight_decay),
    )

    return D, optimD


##########################
##   Dataset Creation   ##
##########################

def create_cell_dataset(
    folders,
    scale=0.125,
    crop=(128,128),
    n_streams=12,
    batch_size=16,
    cell_prominence_min=0.4,
    cell_prominence_max=float('inf'),
):
    def validate_crop(tensor):
        mean = tensor[1:].sum(dim=0).mean()
        if cell_prominence_min < mean < cell_prominence_max:
            return tensor[:1]
        return None

    images = Dataset.ImageIterableDataset(
        folders,
        transforms.Compose((
            transforms.GaussianBlur(3, (.01, 1.)),
            data_aug.flip_scale_pipeline(scale,),
            #data_aug.pipeline(scale, degrees=0, noise_p=0.01),
        )),
        Dataset.CropGenerator(crop, validate_crop=validate_crop),
        n_streams=n_streams,  # large memory impact
        indices = [*range(0, 204), *range(306, 2526)],
    )

    return DataLoader(images, batch_size=batch_size, drop_last=True)


##########################
##      Training        ##
##########################

def train(D, G, optimD, optimG, data, save_name, plot_save_dir, epochs=600):
    try:
        GANs.train(
            D,
            G,
            data,
            "wgan" if 0 else nn.BCEWithLogitsLoss(),
            optimD,
            optimG,
            epochs,
            plot_save_dir,
            n_batches=512,
            init_batches=16,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # View Results?
    try:
        generator_viewer.generator_view_loop(G, D[0])
    except Exception as e:
        print("EXCEPTION WHILE VIEWING GENERATED IMAGES")
        print(str(e))

    # Save Generator?
    ans = input("Save Generator? ([y]/n) ")
    if 'n' not in ans.lower():
        print("Saving...")
        torch.save(G, save_name)
        print(f"Saved as {save_name}")
    else:
        torch.save(G, "FAILSAFE_SAVE.pt")


#######################
##       Main        ##
#######################

if __name__ == "__main__":
    path = dirs.models / "CellGANGenerator.pt"
    channels = 1

    G, optimG = generator_viewer.init_generator(channels, path=path)
    generator_viewer.generator_view_loop(G)
    D, optimD = init_discriminator(channels)

    print(G, '\n')
    print(D, '\n')

    images = create_cell_dataset([dirs.images.images, dirs.images.modified_cell],)

    train(D, G, optimD, optimG, images, path, "StyleGANs", 600)

    print("exit.")

