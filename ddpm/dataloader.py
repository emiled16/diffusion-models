import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


class DataManipulator:
    def __init__(self, img_size: int, batch_size: int) -> None:
        self.img_size = img_size
        self.batch_size = batch_size

        self.transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ])

        self.reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    def _load_data(self) -> torchvision.datasets:

        train = torchvision.datasets.StanfordCars(root=".", download=True, 
                                         transform=self.transform)

        test = torchvision.datasets.StanfordCars(root=".", download=True, 
                                         transform=self.transform, split='test')
        return torch.utils.data.ConcatDataset([train, test])

    def get_data_loader(self) -> DataLoader:
        data = self._load_data()
        return DataLoader(data, batch_size=self.batch_size, shuffle=True, drop_last=True)

def plot_tensor_image(self, image: torch.tensor) -> None:
        # Take first image of batch
        if len(image.shape) == 4:
            image = image[0, :, :, :] 
        plt.imshow(self.reverse_transforms(image))
