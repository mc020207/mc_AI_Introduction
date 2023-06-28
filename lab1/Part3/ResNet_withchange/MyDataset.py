import random

import matplotlib.pyplot as plt
import torch.utils.data as torchio
import numpy as np
from PIL import Image


class MyDataset(torchio.Dataset):
    def __init__(self, dataset):

        self.dataset = dataset

    def __getitem__(self, idx):
        # 获取图像和标签
        image, label = self.dataset[idx][0], self.dataset[idx][1]
        image, label = np.array(image).astype('float32'), int(label)
        image = np.array([image], dtype=np.float32)/255
        return image, label

    def __len__(self):
        return len(self.dataset)
