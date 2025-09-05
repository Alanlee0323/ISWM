import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), torch.as_tensor(target, dtype=torch.int64)

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        return F.normalize(image, mean=self.mean, std=self.std), target
