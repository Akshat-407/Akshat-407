import random
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F, transforms


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class PILToTensor:
    def __call__(self, image, target):
        #image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


import torch



class SegmentationPresetTrain:
    def __init__(self, *, base_size, crop_size, hflip_prob=0.5, mean=(0.208, 0.208 ,0.208), std=(0.175, 0.175, 0.175)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = []
        if hflip_prob > 0:
            trans.append(RandomHorizontalFlip(hflip_prob))
        trans.extend(
            [
                #RandomCrop(crop_size),
                #transforms.Grayscale(num_output_channels=1),
                PILToTensor(),
                ConvertImageDtype(torch.float32),
                #Normalize(mean=mean, std=std),
            ]
        )
        self.transforms = Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, *, base_size, mean=(0.208, 0.208 ,0.208), std=(0.175, 0.175, 0.175)):
        self.transforms = Compose(
            [
                #RandomResize(base_size, base_size),
                #transforms.Grayscale(num_output_channels=1),
                PILToTensor(),
                ConvertImageDtype(torch.float32),
                #Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img, target):
        return self.transforms(img, target)