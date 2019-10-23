from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr


class XView2Segmentation(Dataset):
    """
    xView2 dataset
    """

    NUM_CLASSES = 5

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('xview2'),
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
        self._preimage_dir = os.path.join(self._base_dir, 'pre_images')
        self._postimage_dir = os.path.join(self._base_dir, 'post_images')
        self._mask_dir = os.path.join(self._base_dir, 'masks')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        self.pre_im_ids = []
        self.post_im_ids = []
        self.preimages = []
        self.postimages = []
        self.masks = []

        for split in self.split:
            self.pre_im_ids += os.listdir(os.path.join(self._preimage_dir, split))
            self.post_im_ids += os.listdir(os.path.join(self._postimage_dir, split))
            self.preimages += [os.path.join(self._preimage_dir, split, im) for im in sorted(self.pre_im_ids)]
            self.postimages += [os.path.join(self._postimage_dir, split, im) for im in sorted(self.post_im_ids)]
            self.masks += [os.path.join(self._mask_dir, split, mask) for mask in sorted(self.post_im_ids)]

        assert (len(self.preimages) == len(self.postimages) == len(self.masks))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.preimages)))

    def __len__(self):
        return len(self.preimages)

    def __getitem__(self, index):
        _preimg, _postimg, _target = self._make_img_gt_point_pair(index)
        sample = {'pre_image': _preimg, 'post_image': _postimg, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)


    def _make_img_gt_point_pair(self, index):
        _img1 = Image.open(self.preimages[index]).convert('RGB')
        _img2 = Image.open(self.postimages[index]).convert('RGB')
        _target = Image.open(self.masks[index])

        return _img1, _img2, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'xView2(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 1024
    args.crop_size = 513

    xtrain = XView2Segmentation(args, split='val')

    dataloader = DataLoader(xtrain, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["pre_image"].size()[0]):
            preimg = sample['pre_image'].numpy()
            postimg = sample['post_image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='xview2')
            preimg_tmp = np.transpose(preimg[jj], axes=[1, 2, 0])
            preimg_tmp *= (0.229, 0.224, 0.225)
            preimg_tmp += (0.485, 0.456, 0.406)
            preimg_tmp *= 255.0
            preimg_tmp = preimg_tmp.astype(np.uint8)
            postimg_tmp = np.transpose(postimg[jj], axes=[1, 2, 0])
            postimg_tmp *= (0.229, 0.224, 0.225)
            postimg_tmp += (0.485, 0.456, 0.406)
            postimg_tmp *= 255.0
            postimg_tmp = postimg_tmp.astype(np.uint8)

            fig = plt.figure()
            ax1 = fig.add_subplot(131)
            ax1.title.set_text('Pre Disaster')
            ax1.axis('off')
            ax1.imshow(preimg_tmp)
            ax2 = fig.add_subplot(132)
            ax2.title.set_text('Post Disaster')
            ax2.axis('off')
            ax2.imshow(postimg_tmp)
            ax3 = fig.add_subplot(133)
            ax3.title.set_text('Segmentation Map')
            ax3.axis('off')
            ax3.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
