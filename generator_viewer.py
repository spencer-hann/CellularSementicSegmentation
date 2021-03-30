import torch

from torch import optim

from src.Data import dirs
from src.Data.highlighted_image import view_generated_images
from src.Models.GANs.Style import Generator as Style_Generator
from src.Models.GANs.Vanilla import Generator as Vanilla_Generator


def init_generator(
    channels, lr=1e-3, equalize_lr = False, gen_type = "Vanilla", path = None,
):
    if path: ans = input(f"Load {path}? ([y]/n) ")
    if path and not ans.lower().startswith('n'):
        print("Loading", path)
        G = torch.load(path)
    elif gen_type.lower() == "style":
        G = Style_Generator(channels)
        #G.weight_initialization()
    else:
        G = Vanilla_Generator(channels)

    # use equalized learning rate
    if equalize_lr and hasattr(G, "equalized_lr_dicts"):
        print("Base lr", lr)
        params = G.equalized_lr_dicts(lr=lr)
        for param in params:
            print('\n'.join(str(p) for p in param.items()))
            print()
    else:
        params = G.parameters()

    optimG = optim.AdamW(params, lr=lr, betas=(.5, 0.999), weight_decay=0.0001)

    return G, optimG


def generator_view_loop(generator, discriminator=None, *a, **kw):
    ans = lambda: input("View Generated images? ([y]/n) ")
    while not ans().lower().startswith('n'):
        view_generated_images(generator, discriminator, *a, **kw)


if __name__ == "__main__":
    gen, _ = init_generator(None, path=dirs.models / "CellGANGenerator.pt")
    generator_view_loop(gen, None)

