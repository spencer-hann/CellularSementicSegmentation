import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Iterable, Union

from ..Data import dirs
from ..Data.highlighted_image import highlighted_image


def train(vae, dataloader, epochs, optim=None, lossfn=None, n_batches=1024):
    if optim is None:
        optim  = torch.optim.AdamW(vae.parameters(), lr=0.0002)
    if lossfn is None:
        lossfn = torch.nn.MSELoss()

    loss = []
    for e in range(epochs):
        l = train_epoch(vae, dataloader, optim, lossfn, n_batches=n_batches)
        l = l.item()
        loss.append(l)
        print(f"[{e}/{epochs}] Loss : {l:.4f}")

        if e % 4 == 0:
            progress_report(vae, next(iter(dataloader)), loss, e, lossfn)

    progress_report(vae, next(iter(dataloader)), loss, "final", lossfn, block=True)

    return loss


def train_epoch(vae, dataloader, optim, lossfn, n_batches=float('inf')):
    loss = 0.0
    for i, x in tqdm(enumerate(dataloader, 1), total=n_batches):
        loss += train_batch(vae, x, optim, lossfn)
        if i > n_batches: break
    return loss / i


def train_batch(vae, x, optim, lossfn):
    xhat = vae(x)
    loss = lossfn(x, xhat)
    loss.backward()
    optim.step()
    optim.zero_grad()
    return loss


def progress_report(
    vae: torch.nn.Module,
    x: torch.Tensor,
    loss: Iterable[float],
    epoch: Union[str, int],
    lossfn: torch.nn.Module,
    show: bool = True,
    block: bool = False,
):
    plt.cla()
    plt.clf()
    plt.close()

    n = max(1, min(6, len(x)+1))
    fig, ax = plt.subplots(2, n, figsize=(20,12))

    ax[0,0].plot(loss)
    ax[0,0].set_title(f"Loss by epoch, e{epoch}")
    flteloss = torch.FloatTensor(loss)
    std = torch.std(flteloss)
    mean = torch.mean(flteloss)
    ax[1,0].plot(torch.clip(flteloss, mean - std, mean + std))

    vae.eval()
    out = vae(x)
    vae.train()

    x = highlighted_image(x)
    out = highlighted_image(out)

    for i, orig, rec in zip(range(1, n), x, out):
        l = lossfn(orig[None], rec[None])

        ax[0,i].set_title("Original")
        ax[0,i].imshow(orig)

        ax[1,i].set_title(f"Reconstruction, ({l.item():.4f})")
        ax[1,i].imshow(rec,)

    plt.suptitle(f"Progress Report, epoch {epoch}")
    plt.tight_layout()

    savepath = dirs.progress_report / "VAE" / f"e{epoch}_progress_report.png"
    print(f"Saving progress report to {savepath.as_posix()}")
    plt.savefig(savepath)
    if show:
        plt.show(block=block)
        if not block:
            plt.pause(4.);

