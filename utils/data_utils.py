#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from os import listdir
from os.path import join, basename

import numbers
import numpy as np
import cv2
from libtiff import TIFF
import torch

from PIL import Image
from torch.utils.data.dataset import Dataset


class RandomCrop_and_concat(object):
    """Crop the given Image pair at a random location.

    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = np.random.randint(0, h - th)
        j = np.random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img1, img2):
        """
        Args:
            img1: shape with [w,h]
            img2: shape with [w,h,3]

        Returns:
            Image: Cropped image.
        """
        i, j, h, w = self.get_params(img1, self.size)
        img1 = np.expand_dims(img1[i:i+h, j:j+w], 2)
        img2 = img2[i:i+h, j:j+w, :]

        img = np.concatenate([img1, img2], axis=2).astype(np.float32)

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}'.format(self.size)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', 'tif', 'TIF'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


class TrainDatasetFromFolder(Dataset):
    def __init__(self, hr_dataset_dir, lr_dataset_dir, crop_size, data_range=1023.):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [[join(hr_dataset_dir, x), join(lr_dataset_dir, x)] for x in listdir(lr_dataset_dir) if is_image_file(x)]
        self.transform = RandomCrop_and_concat(crop_size)
        self.data_range = data_range

    def __getitem__(self, index):
        image = self.transform(cv2.imread(self.image_filenames[index][0], -1),
                               cv2.imread(self.image_filenames[index][1], -1))
        hr_image = np.expand_dims(image[:, :, 0]/self.data_range, 0)
        lr_image = np.expand_dims(image[:, :, 1]/self.data_range, 0)
        run_length = np.expand_dims(image[:, :, 2]/self.data_range, 0)
        near = np.expand_dims(image[:, :, 3]/self.data_range, 0)
        return torch.from_numpy(lr_image), torch.from_numpy(hr_image), torch.from_numpy(run_length), torch.from_numpy(near)

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, hr_dataset_dir, lr_dataset_dir, data_range=1023.):
        super(ValDatasetFromFolder, self).__init__()
        self.image_filenames = [[join(hr_dataset_dir, x), join(lr_dataset_dir, x)] for x in listdir(lr_dataset_dir) if
                                is_image_file(x)]
        self.data_range = data_range

    def __getitem__(self, index):
        image_name = basename(self.image_filenames[index][0])
        hr_image = cv2.imread(self.image_filenames[index][0], -1)
        lr_image_ = cv2.imread(self.image_filenames[index][1], -1)
        hr_image = np.expand_dims(hr_image.astype(np.float32) / self.data_range, axis=0)
        lr_image = np.expand_dims(np.squeeze(lr_image_[:, :, 0]).astype(np.float32) / self.data_range, 0)
        run_length = np.expand_dims(np.squeeze(lr_image_[:, :, 1]).astype(np.float32) / self.data_range, 0)
        near = np.expand_dims(np.squeeze(lr_image_[:, :, 2]).astype(np.float32) / self.data_range, 0)

        return image_name, torch.from_numpy(lr_image), torch.from_numpy(hr_image), \
               torch.from_numpy(run_length), torch.from_numpy(near)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, hr_dataset_dir, lr_dataset_dir, data_range=1023.):
        super(TestDatasetFromFolder, self).__init__()
        self.image_filenames = [[join(hr_dataset_dir, x), join(lr_dataset_dir, x)] for x in listdir(lr_dataset_dir) if
                                is_image_file(x)]
        self.data_range = data_range

    def __getitem__(self, index):
        image_name = self.image_filenames[index][0].split('/')[-1]
        hr_image = TIFF.open(self.image_filenames[index][0]).read_image()
        lr_image_ = TIFF.open(self.image_filenames[index][1]).read_image()
        print(lr_image_.shape)
        hr_image = np.expand_dims(hr_image.astype(np.float32) / self.data_range, axis=0)
        lr_image = np.expand_dims(np.squeeze(lr_image_[0]).astype(np.float32) / self.data_range, 0)
        run_length = np.expand_dims(np.squeeze(lr_image_[1]).astype(np.float32) / self.data_range, 0)
        near = np.expand_dims(np.squeeze(lr_image_[2]).astype(np.float32) / self.data_range, 0)
        # hr_image = cv2.imread(self.image_filenames[index][0], -1)
        # lr_image_ = cv2.imread(self.image_filenames[index][1], -1)
        # hr_image = np.expand_dims(hr_image.astype(np.float32) / self.data_range, axis=0)
        # lr_image = np.expand_dims(np.squeeze(lr_image_[:, :, 0]).astype(np.float32) / self.data_range, 0)
        # run_length = np.expand_dims(np.squeeze(lr_image_[:, :, 1]).astype(np.float32) / self.data_range, 0)
        # near = np.expand_dims(np.squeeze(lr_image_[:, :, 2]).astype(np.float32) / self.data_range, 0)
        return image_name, torch.from_numpy(lr_image), torch.from_numpy(hr_image), \
               torch.from_numpy(run_length), torch.from_numpy(near)

    def __len__(self):
        return len(self.image_filenames)
