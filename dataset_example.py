# Read images and augments - Pytorch Dataset Loader
# Patch loader para Sinai
#
# 02/2020

import numpy as np
import os
from random import randint
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF
import random
import sys

from img_process import flip, affine

import cv2

size_8 = True  # True convert all to uint8 - False: uint16, slightly better

#TRAIN_DS_MEAN = 78 if size_8 else 20143  # mean from full train images - COM CROP
TRAIN_DS_MEAN = 53 if size_8 else 13655  # mean from full train images
TRAIN_DS_STD = 65.90    # only considered for 8 bits

PATCH_SIZE = 224   # final size of patch. centered. Doesn't input real size.


# returns a list with complete file name and label
def make_dataset(dir, class_to_idx, view):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                # load only view (MLO or CC) according to parameter
                # if view.upper() in fname or view.lower() in fname:
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                images.append(item)

    return images


class MyDataset(Dataset):
    def __init__(self, image_path, view='CC', train=True):
        self.image_path = image_path

        classes, class_to_idx = self._find_classes(self.image_path)
        print(classes, class_to_idx)

        samples = make_dataset(self.image_path, class_to_idx, view)
        # print(samples)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " +
                   self.image_path + "\n"))

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.train = train

        self.limit = 0.2
        self.max_brightness = 255 if size_8 else 65535
        self.elastic = 100              # gain for elastic transform
        self.angle = 25  # 15
        self.shear = 12
        self.ps = PATCH_SIZE            # patch or image size
        self.translate = 0 #self.ps // 10  # 1/10 of image or patch size - Paper sinai nao tem
        self.gaussian_blur = False      # perform gaussian filter
        self.zoom_in = 0.20             # aumentar zoom # 0.35
        self.zoom_out = 0.20             # diminuir zoom # 0.2

        self.debug = False

        print('[DataLoader] Loaded images for view: ' + view + ', samples size: ', len(samples), self.image_path)

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

    def extract_dataset(self):
        data = []
        labels = []
        for content in self.samples:
            data.append(content[0])
            labels.append(content[1])

        return data, labels

    # Perform data augmentation in training data
    def transform(self, image, target):

        if self.gaussian_blur:
            image = cv2.GaussianBlur(image, (3, 3), 0.5)

        if self.debug:
            x_begin = (image.shape[1] - self.ps) // 2
            y_begin = (image.shape[0] - self.ps) // 2
            image_ = image[y_begin:y_begin+self.ps, x_begin:x_begin+self.ps]
            cv2.imshow('Img antes Aug', image_)

        # intensity shift
        beta = self.limit * random.uniform(-1, 1)   # <0 estraga com os pretos da imagem (full)
        image[:, :, 0] = cv2.add(image[:, :, 0], int(beta*self.max_brightness))
        image[:, :, 1] = cv2.add(image[:, :, 1], int(beta*self.max_brightness))
        image[:, :, 2] = cv2.add(image[:, :, 2], int(beta*self.max_brightness))

        # rotate, translation, scale and shift augs
        angle = randint(-self.angle, self.angle)
        trans_x = randint(-self.translate, self.translate)
        trans_y = randint(-self.translate, self.translate)

        if randint(0, 1) == 0:
            scale = 1 + random.uniform(0, self.zoom_out)  # diminuir zoom # 0.1
        else:
            scale = 1 - random.uniform(0, self.zoom_in)   # aumentar zoom # 0.35

        shear = randint(-self.shear, self.shear)
        # AFFINE - all at once
        image = affine(image, angle, (trans_x, trans_y), scale, shear,
                       mode=cv2.BORDER_REFLECT)

        # flip {0: vertical, 1: horizontal, 2: both, 3: none}
        flip_num = randint(0, 3)
        image = flip(image, flip_num)

        # convert to final patch size
        x_begin = (image.shape[1] - self.ps) // 2
        y_begin = (image.shape[0] - self.ps) // 2
        image = image[y_begin:y_begin+self.ps, x_begin:x_begin+self.ps]

        if self.debug:
            print(image.shape, image.dtype, np.mean(image), self.get_category(target), scale)
            cv2.imshow('Img Pos Aug', image)
            cv2.waitKey(0)

        image = self.standard_normalize(image)

        image = TF.to_tensor(image)

        # label = self.get_label(target)

        return image, target  # label

    def passthrough(self, image, target):

        if self.gaussian_blur:
            image = cv2.GaussianBlur(image, (3, 3), 0.5)

        # UNZOOM NAS IMAGENS, Para pegar mais area (diminuir efeito crop)
        # image = affine(image, 0, (0, 0), 0.8, 0)

        # convert to final patch size
        x_begin = (image.shape[1] - self.ps) // 2
        y_begin = (image.shape[0] - self.ps) // 2
        image = image[y_begin:y_begin+self.ps, x_begin:x_begin+self.ps]

        image = self.standard_normalize(image)
        image = TF.to_tensor(image)

        # label = self.get_label(target)

        return image, target  #  label

    # set labels according to category (folder)
    def get_label(self, target):
        if target == 0:
            label = np.array([1, 0, 0, 0, 0])    # background
        elif target == 1:
            label = np.array([0, 1, 0, 0, 0])    # calc_benign
        elif target == 2:
            label = np.array([0, 0, 1, 0, 0])    # calc_malign
        elif target == 3:
            label = np.array([0, 0, 0, 1, 0])    # mass_benign
        elif target == 4:
            label = np.array([0, 0, 0, 0, 1])    # mass_malign
        return label

    # get category
    def get_category(self, target):
        if target == 0:
            category = 'background'
        elif target == 1:
            category = 'calc_benign'
        elif target == 2:
            category = 'calc_malign'
        elif target == 3:
            category = 'mass_benign'
        elif target == 4:
            category = 'mass_malign'
        return category

    def __len__(self):
        return len(self.samples)

    def show_image(self, image):
        #temp = np.uint16(image)
        temp = np.uint8(image)
        # temp =  cv2.normalize(temp, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
        temp = cv2.resize(temp, (temp.shape[1]//6, temp.shape[0]//6))
        cv2.imshow('name', temp)
        cv2.waitKey(0)
        return

    # normalize accordingly for model
    def standard_normalize(self, image):
        image = np.float32(image)
        # image -= np.mean(image)           # ---> NYU Style
        # image /= np.maximum(np.std(image), 10**(-5))    # UNDER TEST

        image -= TRAIN_DS_MEAN   # Sinai
        image /= 255 if size_8 else 65535          # float [-1,1] 65535

        # transforms.Normalize([0.485, 0.456, 0.406],
        #     	             [0.229, 0.224, 0.225])

        # print (image.max(), image.min())

        return image

    def __getitem__(self, idx):

        path, target = self.samples[idx]

        # Read image and replicate channels for Resnet (16 bit)
        image_gray = cv2.imread(path, cv2.IMREAD_COLOR if size_8 else cv2.IMREAD_UNCHANGED)
        d_type = np.uint8 if size_8 else np.uint16

        image = image_gray

        # image = np.zeros((*image_gray.shape[0:2], 3), dtype=d_type)
        # image[:, :, 0] = image_gray
        # image[:, :, 1] = image_gray
        # image[:, :, 2] = image_gray

        image = cv2.resize(image, (PATCH_SIZE, PATCH_SIZE))
        
        #print(target, idx)


        if self.train is True:
            x, y = self.transform(image, target)
        else:
            x, y = self.passthrough(image, target)

        return x, y
