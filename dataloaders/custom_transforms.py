import torch
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
        preimg = sample['pre_image']
        postimg = sample['post_image']
        mask = sample['label']

        preimg = np.array(preimg).astype(np.float32)
        postimg = np.array(postimg).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        
        preimg /= 255.0
        preimg -= self.mean
        preimg /= self.std
        postimg /= 255.0
        postimg -= self.mean
        postimg /= self.std

        return {'pre_image': preimg,
                'post_image': postimg,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        preimg = sample['pre_image']
        postimg = sample['post_image']
        mask = sample['label']

        preimg = np.array(preimg).astype(np.float32).transpose((2, 0, 1))
        postimg = np.array(postimg).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        preimg = torch.from_numpy(preimg).float()
        postimg = torch.from_numpy(postimg).float()
        mask = torch.from_numpy(mask).float()

        return {'pre_image': preimg,
                'post_image': postimg,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        preimg = sample['pre_image']
        postimg = sample['post_image']
        mask = sample['label']

        if random.random() < 0.5:
            preimg = preimg.transpose(Image.FLIP_LEFT_RIGHT)
            postimg = postimg.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'pre_image': preimg,
                'post_image': postimg,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        preimg = sample['pre_image']
        postimg = sample['post_image']
        mask = sample['label']
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        preimg = preimg.rotate(rotate_degree, Image.BILINEAR)
        postimg = postimg.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'pre_image': preimg,
                'post_image': postimg,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        preimg = sample['pre_image']
        postimg = sample['post_image']
        mask = sample['label']
        if random.random() < 0.5:
            preimg = preimg.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            postimg = postimg.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'pre_image': preimg,
                'post_image': postimg,
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        preimg = sample['pre_image']
        postimg = sample['post_image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = preimg.size
        assert preimg.size == postimg.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        preimg = preimg.resize((ow, oh), Image.BILINEAR)
        postimg = postimg.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            preimg = ImageOps.expand(preimg, border=(0, 0, padw, padh), fill=0)
            postimg = ImageOps.expand(postimg, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = preimg.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        preimg = preimg.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        postimg = postimg.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'pre_image': preimg,
                'post_image': postimg,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        preimg = sample['pre_image']
        postimg = sample['post_image']
        mask = sample['label']
        w, h = preimg.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        preimg = preimg.resize((ow, oh), Image.BILINEAR)
        postimg = postimg.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = preimg.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        preimg = preimg.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        postimg = postimg.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'pre_image': preimg,
                'post_image': postimg,
                'label': mask}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        preimg = sample['pre_image']
        postimg = sample['post_image']
        mask = sample['label']

        assert preimg.size == postimg.size == mask.size

        preimg = preimg.resize(self.size, Image.BILINEAR)
        postimg = postimg.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'pre_image': preimg,
                'post_image': postimg,
                'label': mask}
