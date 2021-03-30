import torch
import numpy as np

import random
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim

from pathlib import Path

from src.Models.Segmentation import UNet
from src.Data import data, dirs, Dataset, data_aug
from src.Training import Segmentation, SelfTraining


#############################
##   Model Initialization  ##
#############################

def create_unet(*args, **kwargs,):
    save_name = kwargs.pop("save_name", None)

    if save_name: ans = input(f"Load {save_name}? [y]/n ")
    if save_name and not ans.lower().startswith('n'):
        print("Loading...")
        net = torch.load(save_name)
        print("Loaded", save_name)
    else:
        net = UNet(*args, **kwargs)

    return net


##########################
##   Dataset Creation   ##
##########################

def create_ground_truth_dataset(lower_bound=.4, upper_bound=1.):
    validate_crop = lambda t: lower_bound <= t[1].mean() <= upper_bound
    images = Dataset.ImageIterableDataset(
        [dirs.images.images, dirs.images.cell],
        data_aug.pipeline(0.25),
        Dataset.CropGenerator((256, 256), validate_crop=validate_crop),
        n_streams=6,  # large memory impact
        indices = [*range(600, 2040)],
    )
    return images

def create_self_training_dataset(
    data, parents, add_original=False, prominence_filtering=True,
):
    dataset = Dataset.SelfTrainingIterableDatasetWrapper(
        data, parents, add_original=add_original
    )
    if prominence_filtering:
        dataset = Dataset.IterableDatasetFilter(  # filter result after parents
            dataset, lambda t: t if ( .1 < t[1].mean() < .99 ) else None
        )
    dataset = Dataset.IterableDatasetFilter(dataset, Dataset.ImageEntropyFilter())
    dataset = Dataset.IterableDatasetFilter(
        dataset, lambda t: torch.cat((t[:1], (t[1:] > .6).float()))
    )
    dataset = Dataset.RepeatBufferIterableDataset(dataset, buffer_size=2048, repeat_prob=.8)
    return dataset

def create_pseudolabel_dataset(parents, add_original=True, prominence_filtering=True):
    images = Dataset.ImageIterableDataset(
        [dirs.images.images, dirs.images.modified_cell],
        data_aug.pipeline(0.25, degrees=0),
        Dataset.CropGenerator((256, 256)),
        n_streams=4,  # large memory impact
        indices = [*range(0, 204), *range(650, 2526)],
    )
    if prominence_filtering:
        images = Dataset.IterableDatasetFilter(  # filter result before parents
            images, lambda t: t if ( .1 < t[1].mean() ) else None
        )
    return create_self_training_dataset(images, parents, add_original=add_original)

def create_generative_dataset(
    parents,
    im_size = 256,
    generator = dirs.models / "SelfTrainingGenerator.pt",
):
    if isinstance(generator, (str, Path)):
        generator = torch.load(generator)
        #generator.eval()

    dataset = Dataset.GenerativeDataset(generator, 1)
    dataset = Dataset.IterableDatasetFilter(dataset, lambda t: t[0])  # no batch
    dataset = Dataset.IterableDatasetFilter(dataset, lambda t: t[:1])  # drop mask
    dataset = Dataset.IterableDatasetFilter(dataset, transforms.Resize(im_size))
    dataset = create_self_training_dataset(dataset, parents, add_original=False)
    return dataset


def create_validation_dataloader(batch_size=8, lower_bound=.6, upper_bound=.99):
    validate_crop = lambda t: lower_bound <= t[1].mean() <= upper_bound
    val = Dataset.ImageIterableDataset(
        [dirs.images.images, dirs.images.cell],
        data_aug.flip_scale_pipeline(0.25),
        Dataset.CropGenerator((256, 256), validate_crop=validate_crop),
        n_streams=6,  # large memory impact
        indices=list(range(350, 550)),
    )
    return DataLoader(val, batch_size=batch_size, drop_last=True)


def create_full_size_dataset():
    return Dataset.ImageIterableDataset(
        [dirs.images.images, dirs.images.modified_cell],
        data_aug.scale_pipeline(0.25),
        lambda t: iter((t,)),  # dummy cropper
        n_streams=1,
        indices=[*range(100), *range(306, 600), *range(2300, 2525)]
    )


##########################
##      Training        ##
##########################

def train(net, train_loader, save_name, epochs, lr=1e-3, decay=0.01):
    try:
        Segmentation.train(
            net,
            train_loader,
            create_validation_dataloader(),
            nn.BCELoss(),
            optim.AdamW(net.parameters(), lr, weight_decay=decay),
            epochs,
            init_batches=12,
            full_size_images=create_full_size_dataset(),
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")


    save = input("Save Model? ([y]/n) ")
    if not save.lower().startswith('n'):
        print("Saving...")
        torch.save(net, save_name)
        print(f"Saved as {save_name}")


if __name__ == "__main__":

    ##  Set Options  ##
    use_raw_data   = 1
    use_parents    = 0
    use_generative = 0


    ##  Create Dataset  ##
    data = []
    if use_raw_data:
        data.append(create_ground_truth_dataset())

    if use_parents or use_generative:
        parents = SelfTraining.Parents.load_parents()
        if use_parents:  # use parents on unlabeled data
            data.append(create_pseudolabel_dataset(
                parents, add_original=True, prominence_filtering=False
            ))
        if use_generative:  # use parents on generative data
            data.append(create_generative_dataset(parents))

    if len(data) > 1:
        data = Dataset.IterableDatasetMerger(data)
    else:
        data = data[0]

    dataloader = DataLoader(data, batch_size=16, drop_last=True)


    ##  Create Main Network  ##
    save_name = dirs.models / "SegmentationUNet.pt"
    net = create_unet(scale_size=2, dropout_p=.4, save_name=save_name)
    print("Traning Network", net)

    train(net, dataloader, save_name, epochs=400, decay=.1)

    print("exit.")

