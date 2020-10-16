# Read images and augments - Pytorch Dataset Loader
# Example Image augmentation and loader
#
# 02/2020
# 08/2020 : include loading from RAM

import os
import sys
import random
from random import randint
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import cv2

from img_process import flip, affine

PATCH_SIZE = 224   # final size of patch. centered
N_CHANNELS = 3     # Color images


def make_dataset(dir, class_to_idx):
    """ returns a list with complete file name and label """
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                images.append(item)

    return images


class MyDataset(Dataset):
    """ Implements a custom PyTorch dataloader """
    def __init__(self, image_path, train=True):
        self.image_path = image_path

        classes, class_to_idx = self._find_classes(self.image_path)
        print(classes, class_to_idx)

        self.samples = make_dataset(self.image_path, class_to_idx)

        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " +
                                self.image_path + "\n"))

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.train = train

        self.limit = 0.2
        self.max_brightness = 255
        self.angle = 25  # 15
        self.shear = 12
        self.translate = 0
        self.zoom_in = 0.20             # increase zoom
        self.zoom_out = 0.20            # decrease zoom

        self.debug = False

        num_images = len(self.samples)

        # full array for all images
        self.samples_array = np.empty((num_images,
                                       PATCH_SIZE, PATCH_SIZE, N_CHANNELS),
                                      dtype=np.uint8)

        # imagenet mean, std to consider
        self.mean = [0.406, 0.456, 0.485] # BGR from (red, green, blue) [0.485, 0.456, 0.406]
        self.std = [0.225, 0.224, 0.229]  # Same (rgb: [0.229, 0.224, 0.225])

        for i, sample in enumerate(self.samples):
            path, target = sample
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (PATCH_SIZE, PATCH_SIZE))
            self.samples_array[i] = image

        print(f'[Dataloader] Loaded {num_images}, size: {self.samples_array.nbytes} bytes from {self.image_path}')


    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir),
                   and class_to_idx is a dictionary.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    # Perform data augmentation in training data
    def transform(self, image):

        if self.debug:
            cv2.imshow('Img antes Aug', image)

        # intensity shift
        beta = self.limit * random.uniform(-1, 1)
        image[:, :, 0] = cv2.add(image[:, :, 0], beta*self.max_brightness)
        image[:, :, 1] = cv2.add(image[:, :, 1], beta*self.max_brightness)
        image[:, :, 2] = cv2.add(image[:, :, 2], beta*self.max_brightness)
        # image = cv2.add(image, beta*self.max_brightness)

        # rotate, translation, scale and shift augs
        angle = randint(-self.angle, self.angle)
        trans_x = randint(-self.translate, self.translate)
        trans_y = randint(-self.translate, self.translate)

        if randint(0, 1) == 0:
            scale = 1 + random.uniform(0, self.zoom_out)
        else:
            scale = 1 - random.uniform(0, self.zoom_in)

        shear = randint(-self.shear, self.shear)

        # AFFINE - all at once
        image = affine(image, angle, (trans_x, trans_y), scale, shear, mode=cv2.BORDER_REFLECT)

        # flip {0: vertical, 1: horizontal, 2: both, 3: none}
        flip_num = randint(0, 3)
        image = flip(image, flip_num)

        if self.debug:
            print(image.shape, image.dtype, np.mean(image), scale)
            cv2.imshow('Img Pos Aug', image)
            cv2.waitKey(0)

        image = self.standard_normalize(image)
        image = torch.from_numpy(image.transpose(2, 0, 1))

        return image

    def passthrough(self, image):
        image = self.standard_normalize(image)
        image = torch.from_numpy(image.transpose(2, 0, 1))
        return image

    def __len__(self):
        return len(self.samples)

    # normalize accordingly for model
    def standard_normalize(self, image):
        image = np.float32(image)
        image /= 255
        for channel in range(N_CHANNELS):
            image[channel] = (image[channel] - self.mean[channel]) / self.std[channel]

        return image

    def __getitem__(self, idx):

        _, target = self.samples[idx]
        image = self.samples_array[idx]

        if self.train is True:
            image = self.transform(image)
        else:
            image = self.passthrough(image)

        return image, target
