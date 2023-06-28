import torch.utils.data as torchio
import numpy as np
from PIL import Image


class MyDataset(torchio.Dataset):
    def __init__(self, dataset, transforms):
        self.transforms = transforms
        self.dataset = dataset

    def __getitem__(self, idx):
        image, label = self.dataset[0][idx], self.dataset[1][idx]
        image, label = np.array(image).astype('float32'), int(label)
        image = np.reshape(image, [28, 28])
        image = Image.fromarray(image.astype('uint8'), mode='L')
        image = self.transforms(image)
        image = np.array(image, dtype=np.float32)
        image = np.array([image], dtype=np.float32)
        return image, label

    def __len__(self):
        return len(self.dataset[0])
