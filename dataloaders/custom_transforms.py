import torch
from torchvision.transforms import functional as F
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.

    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}

class RandomVerticalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': img,
                'label': mask}

class ColorJitter(object):
    def __init__(self, brightness, saturation, contrast, hue):
        self.brightness = brightness
        self.saturation = saturation
        self.contrast = contrast
        self.hue = hue

    def check_params(self):
        if self.brightness is not None:
            assert self.brightness[0] < self.brightness[1]
            assert all(i >= 0 for i in self.brightness) is True

        if self.contrast is not None:
            assert self.contrast[0] < self.contrast[1]
            assert all(i >= 0 for i in self.contrast) is True

        if self.saturation is not None:
            assert self.saturation[0] < self.saturation[1]
            assert all(i >= 0 for i in self.saturation) is True

        if self.hue is not None:
            assert self.hue[0] < self.hue[1]
            assert all(i >= -0.5 for i in self.hue) is True
            assert all(i <= 0.5 for i in self.hue) is True


    def __call__(self, sample):
        self.check_params()

        img = sample['image']
        mask = sample['label']

        if self.brightness is not None:
            """
            Brightness Factor

            n=0 gives a black image,
            n=1 gives the original image
            n>1 increases the brightness by a factor of n.
            """
            brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
            img = F.adjust_brightness(img, brightness_factor)

        if self.contrast is not None:
            """
            Contrast Factor

            n=0 gives a grey image,
            n=1 gives the original image
            n>1 increases the contrast by a factor of n.
            """
            contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
            img = F.adjust_contrast(img, contrast_factor)

        if self.saturation is not None:
            """
            Saturation Factor

            n=0 gives a black & white image,
            n=1 gives the original image
            n>1 increases the saturation by a factor of n.
            """
            saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
            img = F.adjust_saturation(img, saturation_factor)

        if self.hue is not None:
            """
            Hue Factor

            Refer https://pytorch.org/docs/0.4.1/torchvision/transforms.html?highlight=color%20jitter#torchvision.transforms.functional.adjust_hue

            -0.5 < 0 < 0.5
            """
            hue_factor = random.uniform(self.hue[0], self.hue[1])
            img = F.adjust_hue(img, hue_factor)

        return {'image': img,
                'label': mask}

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=255):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}