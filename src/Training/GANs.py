import torch
import random

from torch import autograd

from matplotlib import pyplot as plt; plt.style.use("dark_background");
from matplotlib import lines as mlines
from tqdm import tqdm

from typing import Optional, Union, Iterable, Tuple

from ..Device import device
from ..Data import dirs
from ..Data.highlighted_image import highlighted_image


def print_summary(epoch, epochs, loss_item):
    D_primary, D_secondary, G_loss, real_score, fake_score = loss_item
    print(
        f"[{epoch} / {epochs}]"
        + f" : D Loss {D_primary.mean():.4f}(primary) {D_secondary.mean():.4f}(secondary)"
        + f" : G Loss {G_loss.mean():.4f}"
        + f" : D Score {real_score.mean():.4f}(real) {fake_score.mean():.4f}(fake)"
    )


def train(D, G, data, lossfn, optimD, optimG, epochs, save_dir, n_batches=1024, init_batches=256,):
    print("Training...")
    print("using", lossfn, "loss")
    loss = []

    for epoch in range(epochs):
        loss.append(train_epoch(
            D,
            G,
            data,
            lossfn,
            optimD,
            optimG,
            n_batches if epoch else init_batches,
        ))

        print_summary(epoch, epochs, loss[-1])

        if epoch % 1 == 0:
            plot_progress(D[0], G, data, loss, save_dir, epoch)

    plot_progress(D[0], G, data, loss, save_dir, "Final", show=True, block=True)

    return loss


def train_epoch(D, G, dataloader, lossfn, optimD, optimG, n_batches=256):
    res = torch.zeros(5, len(D))
    for i, data in tqdm(enumerate(dataloader, 1), total=n_batches):
        res += train_multi_discriminator_batch(D, G, data, optimD, optimG, lossfn)
        if i > n_batches: break

    return res / i


def train_multi_discriminator_batch(
    Dlist: Iterable[torch.nn.Module],
    G: torch.nn.Module,
    data: torch.Tensor,
    optimDlist,
    optimG,
    lossfn: Union[str, torch.nn.Module],
):
    res = torch.zeros(5, len(Dlist))

    if isinstance(lossfn, str) and lossfn.lower().startswith("wgan"):
        trainfn = train_batch_wasserstein
    else:
        trainfn = train_batch

    for i, (D, optimD) in enumerate(zip(Dlist, optimDlist)):
        res[:,i] = trainfn(D, G, data, optimD, optimG, lossfn)

    return res


def train_batch_wasserstein(
    D: Iterable[torch.nn.Module],
    G: torch.nn.Module,
    data: torch.Tensor,
    optimD,
    optimG,
    lossfn: Optional[torch.nn.Module] = None,
    clip: float = .1,
    gradient_penalty_lambda: Union[float, torch.Tensor] = 12.0,
):
    G.zero_grad()
    D.zero_grad()

    ## Discriminator Step ##
    G_z = G(len(data))

    D_x = D(data)
    D_G_z = D(G_z.detach())

    Dloss = - D_x + D_G_z  # Critic loss / wasserstein loss
    Dloss = Dloss.mean()
    Dloss.backward()

    #Dloss = Dloss + .1*( (1-D_x)**2 + D_G_z**2 )

    if gradient_penalty_lambda:
        grad_pen = gradient_penalty(D, data, G_z).mean()
        grad_pen = grad_pen * gradient_penalty_lambda
        grad_pen.backward()
    else:
        grad_pen = torch.as_tensor(0.0).to(device)

    if clip:
        torch.nn.utils.clip_grad_norm_(D.parameters(), clip)
    optimD.step()
    optimD.zero_grad()

    ## Generator Step ##
    G.zero_grad()
    D.zero_grad()

    G_z = G(len(data))
    Gloss = -D(G_z)  # Generator loss
    Gloss = Gloss.mean()

    if Gloss > 0:
        Gloss.backward()  # Generator train
        torch.nn.utils.clip_grad_norm_(G.parameters(), .1)
        optimG.step()
    optimG.zero_grad()

    real_score = D_x.detach().mean(); del D_x;
    fake_score = D_G_z.detach().mean(); del D_G_z;

    G.zero_grad()
    D.zero_grad()

    return torch.stack((
        Dloss.detach(),  # primary loss
        grad_pen.detach(),  # secondary loss
        Gloss.detach(),
        real_score.detach(),
        fake_score.detach(),
    ))


def train_batch(D, G, data, optimD, optimG, lossfn):
    zeros = torch.zeros(len(data), device=device)
    ones = torch.ones(len(data), device=device)

    ## Discriminator Step ##
    G.zero_grad()
    D.zero_grad()

    # real score
    D.zero_grad();
    out = D(data).view(-1)
    real_score = out.detach().mean()
    Drloss = lossfn(out, ones)

    # fake score
    fake = G(len(data))
    out = D(fake.detach()).view(-1)
    fake_score = out.detach().mean()
    Dfloss = lossfn(out, zeros); del out, zeros

    grad_pen = 6 * gradient_penalty(D, data, fake,).mean()

    Drloss.backward()
    Dfloss.backward()
    grad_pen.backward()
    optimD.step()
    optimD.zero_grad()

    ## Generator Step ##
    G.zero_grad()
    D.zero_grad()

    fake = G(len(data))
    out = D(fake).view(-1); del fake
    Gloss = lossfn(out, ones); del out, ones

    Gloss.backward()
    optimG.step()
    optimG.zero_grad()

    return torch.stack((
        Drloss.detach(),  # primary loss
        #Dfloss.detach(),  # secondary loss
        grad_pen.detach(),  # secondary loss
        Gloss.detach(),
        real_score.detach(),
        fake_score.detach(),
    ))


def plot_progress(D, G, data, loss, save_dir: str, epoch=None, show=True, block=False):
    plt.cla()
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(3, 4, figsize=(16,12))

    batch = next(iter(data))

    ##  Get Generated Images  ##
    out = G(len(batch)).detach()
    G.zero_grad()

    dout = D(out).detach().cpu()
    D.zero_grad()

    out = highlighted_image(out)

    for i in range(4):
        mn = out[i].min()
        mx = out[i].max()
        ax[0,i].imshow(out[i,:,:], vmin=min(0, mn), vmax=max(1, mx))
        ax[0,i].set_title(f"D(G(z))={dout[i].item():.3f}, ({mn:.2f},{mx:.2f})")


    ##  Get Real Images  ##
    dout = D(batch).detach().cpu()
    dout = dout.view(-1)
    D.zero_grad()

    batch = highlighted_image(batch)

    for i in range(4):
        ax[1,i].imshow(batch[i,:,:], vmin=0, vmax=1)
        ax[1,i].set_title(f"D(x)={dout[i].item():.3f}, ({batch[i].mean():.2f})")

    # epoch x metric x model -> epoch x model x metric
    loss = [zip(*metric_by_model) for metric_by_model in loss]

    for i, loss_by_metric in enumerate(zip(*loss)):  # model x epoch x metric
        primary, secondary, gloss, real, fake = zip(*loss_by_metric)  # metric x epoch

        ax[2,0].plot(primary,   label = f"$D_{i}$")
        ax[2,1].plot(secondary, label = f"$D_{i}$")
        ax[2,2].plot(gloss,     label = f"$D_{i}$")
        ax[2,3].plot(real, f"C{i}",   label = "D score on real data")
        ax[2,3].plot(fake, f"--C{i}", label = "D score on fake data")

    ax[2,0].set_title(f"Discriminator primary loss ({primary[-1]:.2f})")
    ax[2,1].set_title(f"Discriminator secondary loss ({secondary[-1]:.2f})")
    ax[2,2].set_title(f"Generator loss ({gloss[-1]:.2f})")
    ax[2,3].set_title(f"Score by epoch ({real[-1]:.2f}, {fake[-1]:.2f})")

    # same legend for (2,0), (2,1), & (2,2)
    ax[2,0].legend()

    # custom legend for scores subplot
    solid = mlines.Line2D([], [], label="D score on real data")
    dotted = mlines.Line2D([], [], linestyle='--', label="D score on fake data")
    ax[2,3].legend(handles=[solid, dotted])

    plt.suptitle("Progress Report" + (f" epoch {epoch}" if epoch else ''))
    plt.tight_layout()

    savepath = dirs.progress_report / save_dir
    savepath = savepath / f"e{epoch}_progress_report.png"
    print(f"Saving progress report to {savepath.as_posix()}")
    plt.savefig(savepath)

    if show: plt.show(block=block); plt.pause(1.)


def lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    #     t*a + (1-t)*b
    #  => t*a + b - t*b
    #  => t*(a - b) + b
    return t * (a - b) + b


def gradient_penalty(
        D: torch.nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> torch.Tensor:
    n = real.shape[0]  # real and fake must have same n

    fake = fake.detach()
    fake.requires_grad = True

    t = torch.rand(n, 1, 1, 1, dtype=real.dtype, device=real.device)
    xlerp = lerp(real, fake, t)
    out = D(xlerp)

    gradient = autograd.grad(outputs=out.mean(), inputs=xlerp, create_graph=True)
    gradient = gradient[0]

    norm = tensor_norm(gradient, (1,2,3))

    return (norm - 1)**2


def tensor_norm(t: torch.Tensor, dim: Union[int, Tuple[int,...]],) -> torch.Tensor:
    return torch.sqrt((t**2).sum(dim) + 1e-8)

