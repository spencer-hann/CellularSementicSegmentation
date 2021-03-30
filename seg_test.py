import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from tqdm import tqdm
from itertools import chain
from collections import defaultdict

from seg_main import create_validation_dataloader, create_ground_truth_dataset
from src.Data import Dataset, dirs, data_aug
from src.Data.highlighted_image import highlighted_image
from src.Metrics import Segmentation as metrics
from src.Training.Segmentation import full_size_segmentation


def create_test_dataloader(crop=True, modified_cell=True):
    if crop:
        cropper = Dataset.CropGenerator((256, 256))
    else:
        cropper = lambda img: iter((img,))

    folders = [dirs.images.images / "test"]
    if modified_cell:
        folders.append(dirs.images.modified_cell / "test")
    else:
        folders.append(dirs.images.cell / "test")

    images = Dataset.ImageIterableDataset(
        folders,
        data_aug.scale_pipeline(0.25),
        cropper,
        n_streams=1,  # large memory impact
        indices = [*range(204, 306)],
    )
    return DataLoader(images, batch_size=1)

def create_generic_dataloader(crop=True, modified_cell=True):
    if crop:
        cropper = Dataset.CropGenerator((256, 256))
    else:
        cropper = lambda img: iter((img,))

    folders = [dirs.images.images]
    if modified_cell:
        folders.append(dirs.images.modified_cell)
    else:
        folders.append(dirs.images.cell)

    images = Dataset.ImageIterableDataset(
        folders,
        data_aug.scale_pipeline(0.25),
        cropper,
        n_streams=1,  # large memory impact
        indices = [*range(350, 2526)],
    )
    return DataLoader(images, batch_size=1)


def load_models():
    files = chain(
        dirs.self_training_parents.iterdir(),
        (dirs.models / "old_parents").iterdir()
    )
    files = list(files)
    files = sorted(files, key=lambda path: path.name)

    for f in files:
        model = torch.load(f)
        model.eval()
        yield f.name[:-3], model


def test(dataloader, n = float('inf')):
    metric_fns = {
        "Dice":         metrics.DiceCoefficient(),
        "Jaccard":      metrics.JaccardIndex(),
        "Precision":    metrics.Precision(),
        "Recall":       metrics.Recall(),
    }
    results = defaultdict(lambda: defaultdict(lambda: 0.0))

    for idx, model in load_models():
        model.eval()
        m1_total = 0.0
        m2_total = 0.0
        count = 0

        for image in tqdm(dataloader):
            image, mask = image[:,:1], image[:,1:]

            out = model(image).detach()
            model.zero_grad()
            del image

            #out = (out > .5).float()

            count += 1
            for name, metric in metric_fns.items():
                results[name][idx] += metric(out, mask).item()

            if count >= n:
                break

        print("Model", idx)
        for met_name, model_dict in results.items():
            model_dict[idx] /= count
            print('\t', met_name, model_dict[idx])

    return dict(results)


def browse():
    images = create_test_dataloader(crop=False)
    for name, model in load_models():
        print("Showing", name)
        full_size_segmentation(model, images, block=True)

        ans = input("Continue? [Y/n]")
        if ans.lower().startswith('n'):
            break

def browse2():
    mpl.rcParams.update(mpl.rcParamsDefault)

    images = create_generic_dataloader(crop=False, modified_cell=False)
    #images = create_test_dataloader(crop=False, modified_cell=False)
    images = iter(images)

    for idx, model in load_models():
        if idx == 9:
            break

    img = next(images)
    *_, h, w = img.shape
    h_buff = (h%256) // 2
    w_buff = (w%256) // 2
    img = img[:, :, h_buff:h-h_buff, w_buff:w-w_buff]
    img, mask = img[:, :1], img[:, 1:]

    out = model.segment_full_image(img)
    out = out > .6

    highlighted = highlighted_image(torch.cat((img, out), dim=1))

    img = img[0,0].detach().cpu()
    out = out[0,0].detach().cpu()
    mask = mask[0,0].detach().cpu()
    highlighted = highlighted[0].detach().cpu()

    fig, ax = plt.subplots(2, 2, figsize=(20,14))

    ax[0,0].imshow(img, cmap="gray")
    ax[0,1].imshow(mask, cmap="gray")
    ax[1,0].imshow(highlighted)
    ax[1,1].imshow(out, cmap="gray")

    plt.tight_layout()
    plt.show()


def plot_results(results, title, ymin=.5):
    print("Using title", title)
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use("ggplot")

    width = .4
    for metric, model_dict in results.items():
        values = [(idx, val) for idx, val in sorted(model_dict.items())]
        print("values", metric)
        print('\n'.join(map(str, values)))
        values = [v for i, v in values]

        x = range(len(values))
        plt.plot(x, values, linewidth=2, label=metric)

    plt.ylim(bottom=ymin)

    legend = plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.setp(legend.get_texts(), color='k')
    plt.title(f"{title} Results Across Generations")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if 1:
        ans = input("Browse? [y/N] ")
        if ans.lower().startswith("y"):
            browse()

    if 1:
        print("\nRunning Training set")
        train = DataLoader(create_ground_truth_dataset(lower_bound=-1), batch_size=1)
        results = test(train, n=40_000)
        del train
        plot_results(results, "Train", .6)
    if 1:
        print("\nRunning Validation set")
        val = create_validation_dataloader(batch_size=1, lower_bound=-1, upper_bound=2)
        results = test(val)
        del val
        plot_results(results, "Validation", .7)
    if 1:
        print("\nRunning Test set")
        _test = create_test_dataloader()
        results = test(_test)
        del _test
        plot_results(results, "Test", .7)

    for name, metric in results.items():
        print(name,'\t',metric)


