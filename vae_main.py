import torch

from src import Training, Models, Data


def main():
    #Data.highlighted_image.browse(Data.dirs.images.images, Data.dirs.images.cell)
    #return


    images = Data.ImageIterableDataset(
        (Data.dirs.images.images, Data.dirs.images.cell),
        Data.data_aug.flip_scale_pipeline,
        Data.CropGenerator(
            (256,256),
            validate_crop = lambda t: .1 < t[1].mean(),  # cell prominence
        ),
        n_streams=8,
    )
    images = torch.utils.data.DataLoader(images, batch_size=64, drop_last=True)

    ae = Models.Autoencoders.VAE(2048, 2)

    Models.LayerInfo.on()
    print(ae)
    out = ae(next(iter(images)));
    Models.LayerInfo.off()

    Training.VAE.train(ae, images, epochs=256)


if __name__ == "__main__":
    main()

