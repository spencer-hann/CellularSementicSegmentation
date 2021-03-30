import numpy as np
import matplotlib.pyplot as plt

from torchvision.transforms import functional as F

from PIL import Image

from generator_viewer import init_generator
from src.Data import Dataset, dirs, data_aug


def raw_data(
    scale=0.125,
    crop=(128,128),
    n_streams=12,
    batch_size=16,
    cell_prominence_min=0.3,
    cell_prominence_max=0.9,
):
    def validate_crop(tensor):
        mean = tensor[1:].sum(dim=0).mean()
        if cell_prominence_min < mean < cell_prominence_max:
            return tensor[:1]
        return None

    images = Dataset.ImageIterableDataset(
        [dirs.images.images, dirs.images.modified_cell],
        data_aug.flip_scale_pipeline(scale),
        Dataset.CropGenerator(crop, validate_crop=validate_crop),
        n_streams=n_streams,  # large memory impact
        indices = [*range(0, 204), *range(306, 2526)],
    )

    return images


def generator_plot(n = 10, bar_width = 2):
    data = raw_data()
    G,_ = init_generator(None, path=dirs.models / "CellGANGenerator.pt")

    for i, img in enumerate(data):
        img = img[0].detach().cpu().numpy()

        gen = G(1); G.zero_grad();
        gen = gen[0,0].detach().cpu().numpy()

        h = img.shape[0]
        assert h == gen.shape[0]
        w = img.shape[1]
        assert w == gen.shape[1]
        b = bar_width

        arr = np.empty((h,w*2 + bar_width))

        arr[:,:w] = gen
        arr[:,w:w+b] = 0.0
        arr[:,w+b:] = img

        arr *= 255

        plt.imshow(arr)
        plt.show()
        #Image.fromarray(arr).convert("RGB").save(f"./report/img/gan/{i}.png")

        if i >= n: return


if __name__ == "__main__":
    generator_plot()

