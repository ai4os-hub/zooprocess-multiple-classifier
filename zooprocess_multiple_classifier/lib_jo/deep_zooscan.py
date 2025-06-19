import numpy as np

import torchvision.transforms.v2 as tr
import torchvision.transforms.v2.functional as trf
import torch


# Prepare a zooscan image by removing the scale bar and centering on the object
def prepare_zooscan_img(img):
    # convert to tensor (faster
    img = trf.to_image(img)
    
    # crop bottom
    d,h,w = img.size()
    img = trf.crop(img, 0, 0, h-31, w)

    # center on object
    img = trf.invert(img)
    sum_col = torch.sum(img[0,], 0)
    sum_row = torch.sum(img[0,], 1)

    obj_col = np.where(sum_col > 0)
    min_col = np.min(obj_col)
    max_col = np.max(obj_col)

    obj_row = np.where(sum_row > 0)
    min_row = np.min(obj_row)
    max_row = np.max(obj_row)

    w = max_col - min_col + 1
    h = max_row - min_row + 1
    img = trf.crop(img, min_row, min_col, h, w)

    # pad with black
    pad = np.abs((h-w)/2)
    pad_1 = int(pad+0.5)
    pad_2 = int(pad)
    if w < h:
        img = trf.pad(img, padding=(pad_1, 0, pad_2, 0), fill=0)
    else:
        img = trf.pad(img, padding=(0, pad_1, 0, pad_2), fill=0)

    return(img)

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

    return(img)

# Prepare the image and only resize it
def transform_valid(img):
    img = prepare_zooscan_img(img)
    # convert
    convert = tr.Compose([
        tr.Resize(224),
        tr.ToDtype(torch.float32, scale=True)
    ])
    img = convert(img)

    return(img)

# Prepare the image and only resize it
def transform_eval(img):
    img = prepare_zooscan_img(img)
    # some data augmentation
    convert = tr.Compose([
        tr.RandomResizedCrop(224, scale=(0.9,1.1), ratio=(1,1)),
        tr.RandomRotation(90, fill=0),
        tr.RandomVerticalFlip(),
        tr.ColorJitter(brightness=0, contrast=0.1, saturation=0, hue=0),
        tr.ToDtype(torch.float32, scale=True)
    ])
    img = convert(img)

    return(img)


# # test
# from matplotlib import pyplot as plt
# path = '/remote/ecotaxa/vault/26791/2167.jpg'
# 
# 
# img = torchvision.io.read_image(path)
# plt.imshow(img.permute(1, 2, 0));plt.show()
# 
# img_prep = deep_zooscan.prepare_zooscan_img(img)
# plt.imshow(img_prep.permute(1, 2, 0));plt.show()
# 
# img_aug = transform_train(img)
# plt.imshow(img_aug.permute(1, 2, 0));plt.show()

from torch.utils.data import Dataset
from torchvision.io import read_image

class EcoTaxaDataset(Dataset):
    def __init__(self, paths, labels=None, class_to_idx=None, idx_to_class=None, transform=None):
      self.paths = paths
      self.labels = labels
      self.transform = transform
      self.class_to_idx = class_to_idx
      self.idx_to_class = idx_to_class

    def __len__(self):
      return len(self.paths)

    def __getitem__(self, idx):
      img = read_image(self.paths[idx])
      if self.transform:
        img = self.transform(img)
      if self.labels is None:
        label = 0
      else:
        label = self.class_to_idx[self.labels[idx]]
      return img,label

class EcoTaxaEvalDataset(Dataset):
    def __init__(self, paths, names, transform=None):
      self.paths = paths
      self.names = names
      self.transform = transform

    def __len__(self):
      return len(self.paths)

    def __getitem__(self, idx):
      img = read_image(self.paths[idx])
      if self.transform:
        img = self.transform(img)
      name = self.names[idx]
      return img,name
