# -*- coding: utf-8 -*-
"""
Utility functions, used in api.py
(which allow to keep api.py simple)
"""

import torch
import torchvision.transforms.v2 as tr
import torchvision.transforms.v2.functional as trf
import numpy as np


# Prepare a zooscan image by removing the scale bar and centering on the object
def prepare_zooscan_img(img):
    # crop bottom
    w, h = img.size
    img = trf.to_image(img)  # faster on tensors
    img = trf.crop(img, 0, 0, h-31, w)

    # center on object
    img = trf.invert(img)
    sum_col = torch.sum(img[0,], 0)
    sum_row = torch.sum(img[0,], 1)

    obj_col = np.where(sum_col > 2)
    min_col = np.min(obj_col)
    max_col = np.max(obj_col)

    obj_row = np.where(sum_row > 2)
    min_row = np.min(obj_row)
    max_row = np.max(obj_row)

    w = max_col - min_col
    h = max_row - min_row
    img = trf.crop(img, min_row, min_col, h, w)

    # pad with black
    pad = np.abs((h-w)/2)
    pad_1 = int(pad+0.5)
    pad_2 = int(pad)
    if w < h:
        img = trf.pad(img, padding=(pad_1, 0, pad_2, 0), fill=0)
    else:
        img = trf.pad(img, padding=(0, pad_1, 0, pad_2), fill=0)

    return img


# Prepare the image and augment it
def transform_train(img):
    img = prepare_zooscan_img(img)
    # augment
    augment = tr.Compose([
        tr.RandomResizedCrop(224, scale=(1,1.4), ratio=(1,1)),
        tr.RandomRotation(90, fill=0),
        tr.RandomVerticalFlip(),
        tr.ColorJitter(brightness=0, contrast=0.2, saturation=0, hue=0),
        tr.ToDtype(torch.float32, scale=True)
    ])
    img = augment(img)

    return img


# Prepare the image and only resize it
def transform_valid(img):
    img = prepare_zooscan_img(img)
    # convert
    convert = tr.Compose([
        tr.Resize(224),
        tr.ToDtype(torch.float32, scale=True)
    ])
    img = convert(img)

    return img
