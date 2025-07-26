# -*- coding: utf-8 -*-
"""
Utility functions, used in api.py
(which allow to keep api.py simpler)
"""

import torch
from torch.utils.data import Dataset

import torchvision.transforms.v2 as tr
import torchvision.transforms.v2.functional as trf
from torchvision.io import read_image


# Prepare a zooscan image by removing the scale bar and centering on the object
def prepare_zooscan_img(img, bottom_crop):
    # convert to tensor if it isn't one already
    img = trf.to_image(img)

    # crop bottom
    d, h, w = img.size()
    img = trf.crop(img, 0, 0, h-bottom_crop, w)

    # center on object
    img = trf.invert(img)
    sum_col = torch.sum(img[0,], 0)
    sum_row = torch.sum(img[0,], 1)

    obj_col = torch.where(sum_col > 2)[0]
    min_col = torch.min(obj_col)
    max_col = torch.max(obj_col)

    obj_row = torch.where(sum_row > 2)[0]
    min_row = torch.min(obj_row)
    max_row = torch.max(obj_row)

    w = max_col - min_col
    h = max_row - min_row
    img = trf.crop(img, min_row, min_col, h, w)

    # pad with black
    pad = torch.abs((h-w)/2)
    pad_1 = int(pad+0.5)
    pad_2 = int(pad)
    if w < h:
        img = trf.pad(img, padding=(pad_1, 0, pad_2, 0), fill=0)
    else:
        img = trf.pad(img, padding=(0, pad_1, 0, pad_2), fill=0)

    return img


# Prepare the image and augment it
def transform_train(img, bottom_crop):
    img = prepare_zooscan_img(img, bottom_crop)
    # augment
    augment = tr.Compose([
        tr.RandomResizedCrop(224, scale=(1, 1.4), ratio=(1, 1)),
        tr.RandomRotation(90, fill=0),
        tr.RandomVerticalFlip(),
        tr.ColorJitter(brightness=0, contrast=0.2, saturation=0, hue=0),
        tr.ToDtype(torch.float32, scale=True)
    ])
    img = augment(img)

    return img


# Prepare the image and only resize it
def transform_valid(img, bottom_crop):
    img = prepare_zooscan_img(img, bottom_crop)
    # resize
    convert = tr.Compose([
        tr.Resize(224),
        tr.ToDtype(torch.float32, scale=True)
    ])
    img = convert(img)

    return img

# NB: for the training and validation datasets, we'll use pytorch.datasets.ImageFolder()

# Custom dataset class for the evaluation
class ZooScanEvalDataset(Dataset):
    def __init__(self, paths, names, transform=None, bottom_crop=0):
      self.paths = paths
      self.names = names
      self.transform = transform
      self.bottom_crop = bottom_crop

    def __len__(self):
      return len(self.paths)

    def __getitem__(self, idx):
      img = read_image(self.paths[idx])
      if self.transform:
        img = self.transform(img, bottom_crop=self.bottom_crop)
      name = self.names[idx]
      return img,name
