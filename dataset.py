import errno
import os
import os.path as osp
from collections import OrderedDict
import collections
import re

import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import Tensor
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torch._six import string_classes, int_classes

import torchvision
from torchvision import datasets, models

import numpy as np
from lxml import etree

import utils
from utils import load_image, letterbox_image, \
				  letterbox_label, letterbox_label_reverse, \
				  bbox_x1y1x2y2_to_xywh, bbox_x1y1x2y2_to_cxcywh, bbox_cxcywh_to_x1y1x2y2, bbox_cxcywh_to_xywh
import transforms			

class COCODataset(Dataset):
    def __init__(self, targ_txt, dim=None, bbs_idx=np.arange(0,4), transform=None):
        with open(targ_txt, 'r') as f:
            self.img_list = [lines.strip() for lines in f.readlines()]
        self.label_list = [img_path.replace('jpg', 'txt').replace('images', 'labels') for img_path in self.img_list]
        self.dim = dim
        self.bbs_idx = bbs_idx
        self.transform = transform
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        label = None
        img_path = self.img_list[idx]
        if osp.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), img_path)
        
        label_path = self.label_list[idx]
        if osp.exists(label_path):
            label = np.loadtxt(label_path).reshape(-1,5)
        
        sample = { 'img': img, 'org_img': img.copy(), 'label': label, 'transform': None, 'img_path': img_path }
        sample = self.transform(sample)
        
        return sample

class CVATDataset(Dataset):
    def __init__(self, img_dir, label_xml_path, dim=None, transform=None):
        self.img_dir = img_dir
        self.label_xml_path = label_xml_path
        self.xml_dict = list(get_xml_labels(self.label_xml_path).items())
        self.dim = dim
        self.transform = transform
        self.class2id = { 'x_wing': 0, 'tie': 1}
        self.id2class = {v:k for k,v in self.class2id.items()}
        
    def __len__(self):
        return len(self.xml_dict)
    
    def isTrain(self, is_train):
        self.is_train = is_train
        return self
    
    def __getitem__(self, idx):
        label = None
        img_path, label = self.xml_dict[idx]
        
        img_path = osp.join(self.img_dir, img_path)
        if osp.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), img_path)

        label = np.array( [ [self.class2id[l['cls']],
                                             l['x1'],
                                             l['y1'],
                                             l['x2'],
                                             l['y2'] ] for l in label] ).astype(np.float)
        bbs_idx = np.arange(1,5)
        label[..., bbs_idx] = bbox_x1y1x2y2_to_cxcywh(label[..., bbs_idx])
        
        sample = { 'img': img, 'org_img': img.copy(), 'label': label, 'transform': None, 'img_path': img_path }
        sample = self.transform(sample)
        return sample

def get_xml_labels(xml_path):
    labels = OrderedDict()
    
    tree = etree.parse(xml_path)
    root = tree.getroot() 
    
    img_tags = root.xpath("image")

    for image in img_tags:
        img = image.get('name', None)
        labels[img] = []
        for box in image:
            cls = box.get('label', None)
            x1 = box.get('xtl', None)
            y1 = box.get('ytl', None)
            x2 = box.get('xbr', None)
            y2 = box.get('ybr', None)
            labels[img] += [{'cls' : cls, 
                             'x1'  : x1 ,
                             'y1'  : y1 ,
                             'x2'  : x2 ,
                             'y2'  : y2  }]
    return labels

def fill_label_np_tensor(label, row, col):
    label_tmp = np.full((row, col), 0.0)
    if label is not None:
        length = label.shape[0] if label.shape[0] < row else row
        label_tmp[:length] = label[:length]
    return label_tmp

# Modify 'default_collate' from dataloader.py in pytorch library
# Read 'default_collate' from https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py
# Only small portion is modified

def variable_shape_collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    _use_shared_memory = True

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        # Check if the tensors have same shapes.
        # If True, stack the tensors. If false, return a list of tensors
        is_same_shape = all([b.shape == batch[0].shape for b in batch])
        if not is_same_shape:
            return batch
        else:
            out = None
            if _use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: variable_shape_collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [variable_shape_collate_fn(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))