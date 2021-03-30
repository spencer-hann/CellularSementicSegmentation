import torch
import random

from torch import autograd

from matplotlib import pyplot as plt
from matplotlib import lines as mlines
from tqdm import tqdm

from typing import Optional, Union, Iterable, Tuple

from ..Device import device
from ..Data import dirs
from ..Data.highlighted_image import highlighted_image
from ..Metrics.Segmentation import DiceCoefficient, entropy


def print_summary(epoch, epochs, loss, val_loss, met, val_met):
    print(
        f"[{epoch} / {epochs}]",
        f"Train Loss {loss[-1]:.4f} :",
        f"Val Loss {val_loss[-1]:.4f} :",
        f"Train Metric {met[-1]:.4f} :",
        f"Val Metric {val_met[-1]:.4f} ",
    )


def train(
    net,
    data,
    val_data,
    lossfn,
    optim,
    epochs,
    metric=None,
    n_batches=1024,
    n_test_batches=256,
    init_batches=256,
    full_size_images=None,
):
    print("Training...")

    if metric is None:
        metric = DiceCoefficient()

    loss = []
    met = []

    val_loss = []
    val_met = []

    for epoch in range(epochs):
        l, m = train_epoch(
            net, data, lossfn, optim, metric,
            n_batches if epoch else init_batches,
        )

        loss.append(l)
        met.append(m)

        net.eval()
        l, m = train_epoch(
            net, val_data, lossfn, None, metric,
            n_test_batches if epoch else init_batches,
        )
        net.train()

        val_loss.append(l)
        val_met.append(m)

        print_summary(epoch, epochs, loss, val_loss, met, val_met)

        if epoch % 1 == 0:
            plot_progress(net, data, loss, val_loss, met, val_met, epoch, "Training Data")
            if full_size_images: full_size_segmentation(net, full_size_images)
            plot_progress(net, val_data, loss, val_loss, met, val_met, epoch, "Validation Data", preclear=False)

    plot_progress(net, data, loss, met, val_met, "Final", "Training Data", how=True,)
    if full_size_images: full_size_segmentation(net, full_size_images)
    plot_progress(net, val_data, loss, met, val_met, "Final", "Validation Data", show=True, block=True, preclear=False)

    return loss


def train_epoch(net, dataloader, lossfn, optim, metric, n_batches=256):
    loss = 0
    met = 0
    for i, data in tqdm(zip(range(n_batches), dataloader), total=n_batches):
        l, m = train_batch(net, data, lossfn, optim, metric)
        loss += l.item()
        met += m.item()

    return loss / n_batches, met / n_batches


def train_batch(net, data, lossfn, optim, metric):
    img  = data[:, :1, ...]
    mask = data[:, 1:, ...]

    out = net(img)

    loss = lossfn(out, mask)

    if optim:
        loss.backward()
        optim.step()
        optim.zero_grad()

    met = metric(out.detach(), mask.detach())

    return loss, met


def plot_progress(
    net, data, loss, val_loss, met, val_met, epoch, title,
    n=5, preclear=True, show=True, block=False,
):
    plt.style.use("dark_background")

    if preclear:
        plt.cla(); plt.clf(); plt.close('all');

    fig, ax = plt.subplots(2, n+1, figsize=(22,12))

    batch = next(iter(data))
    img  = batch[:, :1, ...]
    mask = batch[:, 1:, ...]
    batch_entr = list(map(entropy, mask))
    batch = highlighted_image(batch)

    net.eval()
    out = net(img)
    out_entr = list(map(entropy, out))
    out = highlighted_image(torch.cat((img, out), dim=1))
    net.train()

    for i in range(n):
        ax[0,i+1].imshow(out[i])
        ax[0,i+1].set_title(f"UNet Mask {out_entr[i]:.2f}")

        ax[1,i+1].imshow(batch[i])
        ax[1,i+1].set_title(f"Training Image {batch_entr[i]:.2f}")

    ax[0,0].set_title("Loss")
    ax[0,0].plot(loss, label="Train")
    ax[0,0].plot(val_loss, label="Val")
    ax[0,0].legend()

    ax[1,0].set_title("Metric")
    ax[1,0].plot(met, label="Train")
    ax[1,0].plot(val_met, label="Val")
    ax[1,0].legend()

    plt.suptitle(title)

    plt.tight_layout()

    subdir = "Segmentation"
    savepath = dirs.progress_report / subdir / f"e{epoch}_{title}.png"
    print(f"Saving progress report to {savepath.as_posix()}")
    plt.savefig(savepath)

    if show: plt.show(block=block); plt.pause(1.)

def full_size_segmentation(model, images, n=2, block=False,):
    plt.style.use("dark_background")

    images = iter(images)

    fig, ax = plt.subplots(n, n, figsize=(20,14))

    for i in range(n):
        for j in range(n):
            image = next(images)
            while len(image.shape) < 4:
                image = image[None]

            mask  = image[:, 1:]
            image = image[:, :1]

            out = model.segment_full_image(image)

            if j == n-1:
                out = (out > .6).float()

            entr = entropy(out)

            out = torch.cat((image, out, mask), dim=1)
            out = highlighted_image(out)
            out = out[0]

            ax[i,j].imshow(out)
            ax[i,j].set_title(f"Mask Entropy {entr:.2f}")

    plt.tight_layout()
    plt.show(block=block)

