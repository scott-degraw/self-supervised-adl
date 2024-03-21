import os
import torch
import numpy as np
from PIL import Image

from unet.model import UNet
from unet.datasets import OxfordPetsDataset


if __name__ == "__main__":
    runs_dir = os.path.join(os.getcwd(), "runs")
    data_dir = os.path.join(os.getcwd(), "data")

    trainset = OxfordPetsDataset(data_dir, split="train", image_size=(240, 240))
    testset = OxfordPetsDataset(data_dir, split="test", image_size=(240, 240))

    model = UNet()
    model_path = os.path.join(runs_dir, "unet_20240312T162901.pt")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Successfully loaded model.")

    index = 10
    testset.show_image(index)
    testset.show_trimap(index)

    input = testset.__getitem__(index)[0].unsqueeze(0)
    label = testset.__getitem__(index)[1]

    pred = model(input)

    image = (np.array(pred.argmax(dim=1).squeeze(0) + 1, dtype=np.uint8)) * 80
    image = Image.fromarray(image)
    image.convert("L")
    image.show()
