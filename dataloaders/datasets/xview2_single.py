from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
import random


class XView2SingleSegmentation(Dataset):
    """
    xView2 single dataset
    """

    NUM_CLASSES = 2

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('xview2_single'),
                 split='train',
                 ):
        """
        Initialise the dataset.

        :param base_dir: path to xview2 dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'images')
        self._mask_dir = os.path.join(self._base_dir, 'masks')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        self.im_ids = []
        self.images = []
        self.masks = []

        for split in self.split:
            self.im_ids += os.listdir(os.path.join(self._image_dir, split))
            self.images += [os.path.join(self._image_dir, split, im) for im in sorted(self.im_ids)]
            self.masks += [os.path.join(self._mask_dir, split, mask) for mask in sorted(self.im_ids)]

        assert len(self.images) == len(self.masks)

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.masks[index])

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomVerticalFlip(),
            tr.RandomRotate(30),
            #tr.ColorJitter(brightness=[0.6, 1.5], contrast=[0.6, 1.5], saturation=[0.6, 1.5], hue=[0, 0.5], siamese=False),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'xView2_single(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 1024
    args.crop_size = 513

    xtrain = XView2SingleSegmentation(args, split='val')

    dataloader = DataLoader(xtrain, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["pre_image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='xview2_single')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)

            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax1.title.set_text('Image')
            ax1.axis('off')
            ax1.imshow(img_tmp)
            ax2 = fig.add_subplot(122)
            ax2.title.set_text('Segmentation Map')
            ax2.axis('off')
            ax2.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
