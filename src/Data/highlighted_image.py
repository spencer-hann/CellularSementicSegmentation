import torch
import matplotlib.pyplot as plt

from typing import Tuple, Mapping, Optional

from . import dirs, data_aug
from .Dataset import ImageIterableDataset, CropGenerator
from ..Types import PathLike


def highlighted_image(
    t: torch.Tensor, color_intensity: Mapping[int,float] = [.06, .08, .08],
) -> torch.Tensor:
    t = t.detach().cpu()
    t = t.permute(0, 2, 3, 1)  # NCHW -> NHWC
    if highlighted_image.RGB_override and t.shape[-1] == 3:  # temorarily override for RGB images
        return t
    if t.shape[-1] == 1:  # only one color
        return t

    cshape = list(t.shape)
    cshape[3] = 3  # color dimension from grey-scale to RGB
    img = torch.zeros(cshape)
    img[...] = t[:,:,:,0,None]

    for i,j in zip([0,2,1], range(1, min(t.shape[3], 3))):
        # iterate in RBG order, because green and
        # red look bad together for 2-channel masks
        img[:,:,:,i] += t[:,:,:,j] * color_intensity[j]

    return img / img.max()


highlighted_image.RGB_override = False


def browse(*folders: Tuple[PathLike]):
    images = ImageIterableDataset(
        folders,
        lambda t: t,
        #data_aug.flip_scale_pipeline(.5),
        #lambda t: iter([t]),
        CropGenerator((256,256), validate_crop = lambda t: .2 < t[1].mean() < .9),
        n_streams=1,
    )

    for i, img in enumerate(images):
        fig, ax = plt.subplots(1, 3, figsize=(16,6))

        ax[0].imshow(img[0].cpu())
        ax[1].imshow(img[1].cpu())

        img = img[None,...]
        img = highlighted_image(img)
        ax[2].imshow(img[0])

        plt.show()


def view_generated_images(
    G: torch.nn.Module,
    D: Optional[torch.nn.Module],
    *subplot_shape: Tuple[int, ...],
    cmap: str = "virdis",
) -> None:
    if len(subplot_shape) == 1:
        subplot_shape = 1, subplot_shape[0]
    elif len(subplot_shape) != 2:
        raise ValueError(f"Subplot shape must be 1 or 2 dims, got {subplot_shape}")

    n_images = torch.prod(torch.as_tensor(subplot_shape))

    out = G(n_images)
    G.zero_grad()

    out = out.reshape(*subplot_shape, *out.shape[1:])

    if D:
        dout = D(out)
        D.zero_grad()
        dout = dout.reshape(*subplot_shape, *dout.shape[1:])

    # double rows
    subplot_shape = 2*subplot_shape[0], subplot_shape[1]
    fig, ax = plt.subplots(*subplot_shape, figsize=(20, 10))

    for i in range(0, subplot_shape[0], 2):
        for j in range(subplot_shape[1]):
            ax[i  ,j].imshow(highlighted_image(out[i//2,j, None])[0], cmap=cmap)
            ax[i+1,j].imshow(out[i//2,j,0].detach().cpu(), cmap=cmap)
            if D: ax[i,j].set_title(f"D(G(z)) = {dout[i//2,j].item():.4f}")

    plt.show()


def browse_all_cell_masks(indices = None, downsample = True, auto_save = None, show=True):
    if indices is None:
        indices = range(0, 2000, 200)

    if downsample and not callable(downsample):
        downsample = lambda img: img[::4, ::4]
    elif not downsample:
        downsample = lambda img: img

    for i in indices:
        i = str(i).zfill(4)
        fname = i + ".pt"

        plt.figure(figsize=(12,10))

        raw = torch.load(dirs.images.images / fname).cpu()
        if (dirs.images.cell / fname).exists():
            cell = torch.load(dirs.images.cell / fname).cpu()
        else:
            print(fname, "Cell Mask DNE")
            cell = torch.zeros_like(raw)

        h = torch.cat((raw, cell))
        h = highlighted_image(h[None])[0]
        h = downsample(h)
        plt.imshow(h)
        plt.title("Image + Cell Mask #" + fname[:4])

        plt.tight_layout()

        if auto_save:
            print("Saving to ", auto_save / ("img"+fname[:4]+".png"))
            plt.savefig(auto_save / ("img"+fname[:4]+".png"))

        if show: plt.show(block=True)
        plt.close()



def browse_all_images(indices = None, downsample = True):
    if indices is None:
        indices = range(0, 2000, 200)

    if downsample and not callable(downsample):
        downsample = lambda img: img[::4, ::4]
    elif not downsample:
        downsample = lambda img: img

    for i in indices:
        i = str(i).zfill(4)
        fname = i + ".pt"
        fig, ax = plt.subplots(2, 2, figsize=(22,12))

        raw = torch.load(dirs.images.images / fname).cpu()
        mcell = torch.load(dirs.images.modified_cell / fname).cpu()

        h = torch.cat((raw, mcell))
        h = highlighted_image(h[None])[0]
        h = downsample(h)
        ax[1,0].imshow(h)
        ax[1,0].set_title("Cell + Nuclei + Image")


        raw = raw[0].float().numpy()
        raw = downsample(raw)
        ax[0,0].imshow(raw)
        ax[0,0].set_title("Plain Image")

        if (dirs.images.cell / fname).exists():
            cell = torch.load(dirs.images.cell / fname).cpu()
        else:
            cell = torch.zeros_like(mcell)

        cell = cell[0].float().numpy()
        cell = downsample(cell)
        ax[0,1].imshow(cell)
        ax[0,1].set_title("Cell Mask")

        mcell = mcell[0].float().numpy()
        mcell = downsample(mcell)
        ax[1,1].imshow(mcell)
        ax[1,1].set_title("Cell + Nuclei Mask")

        plt.suptitle("Image " + fname[:4])
        plt.tight_layout()

        plt.show(block=True)
        plt.close()


