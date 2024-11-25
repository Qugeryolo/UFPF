# -*- coding: utf-8 -*-
# @Time    : 2024/3/26 
# @Author  : Geng Qin
# @File    : OSCC_dataset.py
# @Software: Vscode
import os
from torch.utils.data import Dataset
import scipy.io as sio
from PIL import Image
import numpy as np
import imageio
import spectral
import albumentations as A
import random
import torch
from torch.utils import data


def open_file(datapath):
    _, ext = os.path.splitext(datapath)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return sio.loadmat(datapath)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return imageio.imread(datapath)
    elif ext == '.hdr':
        img = spectral.open_image(datapath)
        return img.load()
    elif ext == '.png':
        img = Image.open(datapath)
        img = np.asarray(img)
        return img
    elif ext == '.jpg':
        img = Image.open(datapath)
        img = np.asarray(img).squeeze()
        return img
    elif ext == '.jpeg':
        img = Image.open(datapath)
        img = np.asarray(img).squeeze()
        return img
    else:
        raise ValueError("Unknown file format: {}".format(ext))


def light_aug(image, mask, seed=42):
    """
    Light non-destructive augmentations.
    """
    aug = A.Compose([
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5)]
    )
    random.seed(seed)
    augmented = aug(image=image, mask=mask)
    image_aug = augmented['image']
    mask_aug = augmented['mask']

    return image_aug, mask_aug


def medium_aug(image, mask, seed=42):
    """
    Medium non-destructive augmentations.
    Let's add non_rigid transformations and RandomSizedCrop
    """
    aug = A.Compose([
        A.OneOf([
        A.RandomSizedCrop(min_max_height=(50, 101), height=256, width=256, p=0.5),
        A.PadIfNeeded(min_height=256, min_width=256, p=0.5)
    ],p=1),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.OneOf([
        A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
    ], p=0.8)]
    )
    random.seed(seed)
    augmented = aug(image=image, mask=mask)
    image_aug = augmented['image']
    mask_aug = augmented['mask']

    return image_aug, mask_aug


def high_aug(image, mask, seed=42):
    """
    Args:
        add non-spatial stransformations
    Many non-spatial transformations like CLAHE, RandomBrightness,
    RandomContrast, RandomGamma can be also added.
    They will be applied only to the image and not the mask
    """
    aug = A.Compose([
        A.OneOf([
            A.RandomSizedCrop(min_max_height=(50, 101), height=256, width=256, p=0.5),
            A.PadIfNeeded(min_height=256, min_width=256, p=0.5)
        ], p=1),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
            ], p=0.8),
        A.CLAHE(p=0.8),
        A.RandomBrightnessContrast(p=0.8),
        A.RandomGamma(p=0.8)])
    random.seed(seed)
    augmented = aug(image=image, mask=mask)
    image_aug = augmented['image']
    mask_aug = augmented['mask']

    return image_aug, mask_aug


class HSISegmentation(Dataset):
    """
    Args:
        My dataset : ndarray
        Supervised training
        Image size : (256,256,60)

    """
    def __init__(self, root, txt_name: str = "train.txt", training=True, transforms=None):
        super(HSISegmentation, self).__init__()

        self.image_dir = os.path.join(root,  'roi_tif')
        self.mask_dir = os.path.join(root, 'roi_label')
        # train.txt // test.txt path
        abs_path = r"/home/Qugeryolo/PycharmProjects/pythonProject/project/1_ACCL_CINet/data_split_OSCC"
        txt_path = os.path.join(abs_path, txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        file_list = []

        for name in file_names:
            for i in range(99):
                new_name = name + '_' + str(i)
                file_list.append(new_name)

        self.images = [os.path.join(self.image_dir, x + ".tif") for x in file_list]
        self.masks = [os.path.join(self.mask_dir, x + ".tif") for x in file_list]
        assert (len(self.images) == len(self.masks))
        self.is_training = training
        self.transforms = transforms

        if self.transforms == 'light_aug':
            self.transforms = light_aug
        elif self.transforms == 'medium_aug':
            self.transforms = medium_aug
        elif self.transforms == 'high_aug':
            self.transforms = high_aug

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        hsi_img = open_file(self.images[index])
        target = open_file(self.masks[index])

        hsi_img = np.transpose(hsi_img, (1, 2, 0))
        img_hsi_aug, target_aug = self.transforms(hsi_img, target)

        img_hsi_aug = img_hsi_aug.astype(np.float32)

        data = torch.from_numpy(img_hsi_aug).permute(2, 0, 1)
        label = torch.from_numpy(target_aug)

        return data, label

    def __len__(self):
        return len(self.images)


def get_train_loader(dataset, batch_size, num_workers, collate_fn=None):

    is_shuffle = True
    train_loader = data.DataLoader(dataset,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=False,
                                   collate_fn=collate_fn)

    return train_loader


if __name__ == "__main__":

    train_dataset = HSISegmentation(root=r"/media/datasets1/Quger/datasets/OSCC", txt_name='fold1_train.txt', transforms='light_aug')

    test_dataset = HSISegmentation(root=r"/media/datasets1/Quger/datasets/OSCC", txt_name='fold1_test.txt', transforms='light_aug')

    train_loader = get_train_loader(train_dataset, batch_size=4, num_workers=0)
    test_loader = get_train_loader(test_dataset, batch_size=4, num_workers=0)
