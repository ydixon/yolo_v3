import errno
import os
import os.path as osp
from collections import OrderedDict
import collections
import re
import math

import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import Tensor
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split, Subset
from torch._six import string_classes, int_classes

import torchvision
from torchvision import datasets, models

import numpy as np
from lxml import etree

import utils
from boundingbox import bbox_x1y1x2y2_to_xywh, bbox_x1y1x2y2_to_cxcywh, bbox_cxcywh_to_x1y1x2y2, bbox_cxcywh_to_xywh, \
                        bbox_xywh_to_cxcywh, bbox_xywh_to_x1y1x2y2, CoordinateType, FormatType, BoundingBoxConverter
import transforms
import imgaug as ia         

class RandomCyclicDataset(Dataset):
    def __init__(self, batch_size, shuffle=True, cyclic=True, dim=None, rand_dim_interval=None):
        self.cyclic = cyclic
        
        self.base_indices = self.get_base_indices()
        self.base_length = len(self.base_indices)
        
        self.batch_size = batch_size
        if self.cyclic:
            self.indices_batch = self.base_length // self.batch_size
            self.indices_size = self.indices_batch * self.batch_size
        else:
            self.indices_batch = math.ceil(self.base_length / self.batch_size)
            self.indices_size = len(self.base_indices)
        
        self.shuffle = shuffle
        self.dim = dim
        self.rand_dim_interval = rand_dim_interval
        self.rng_state = None
        self.cyclic = cyclic
        
        self.indices_queue = torch.LongTensor([])
        self.dims_queue = torch.LongTensor([])
        self.rands_queue = torch.LongTensor([])
        self.randomize()
        
    def get_base_indices(self):
        raise NotImplementedError
        

    def _generate_indices_list(self):
        if self.shuffle:
            new_indices = torch.randperm(self.base_length)
        else:
            new_indices = torch.arange(0,self.base_length, dtype=torch.int64)
        
        if self.cyclic:
            if len(self.indices_queue) < self.indices_size:
                self.indices_queue = torch.cat((self.indices_queue, new_indices))
            indices, self.indices_queue = self.indices_queue[:self.indices_size], self.indices_queue[self.indices_size:]
        else:
            indices = new_indices
            self.indices_queue = torch.LongTensor([])
        return indices.tolist()
    
    def _generate_dims_list(self, indices, dim=None, rand_dim_interval=8):
        if dim is not None:
            return [dim for i in range(0, self.indices_size)]
        
        if self.base_length <= rand_dim_interval:
            nDim = 1
        else:
            nDim = math.ceil(self.base_length / rand_dim_interval)
        
        if len(self.dims_queue) < self.indices_size:
            new_dims = torch.randint(10, 20, (nDim,1), dtype=torch.int64) * 32
            new_dims = new_dims.repeat((1,rand_dim_interval)).reshape(1, -1).squeeze()
            self.dims_queue = torch.cat((self.dims_queue, new_dims))
        dims, self.dims_queue = self.dims_queue[:self.indices_size], self.dims_queue[self.indices_size:]
        return [(sz,sz) for sz in dims.tolist()]
    
    def _generate_rands_list(self):
        if len(self.rands_queue) < self.indices_size:
            new_rands = torch.randint(2**32, (self.base_length,), dtype=torch.int64)
            self.rands_queue = torch.cat((self.rands_queue, new_rands))
        rands, self.rands_queue = self.rands_queue[:self.indices_size], self.rands_queue[self.indices_size:]
        return rands.tolist()
    
    def randomize(self, rng_state=None):
        if rng_state is not None:
            torch.set_rng_state(rng_state)
        elif self.rng_state is not None:
            torch.set_rng_state(self.rng_state)

        self.indices = self._generate_indices_list()
        self.dims = self._generate_dims_list(self.indices, dim=self.dim, rand_dim_interval=self.rand_dim_interval)
        self.rands = self._generate_rands_list()
        
        self.rng_state = torch.get_rng_state()
    
    def get_state_dict(self):
        state_dict = { 'dataset_indices' : self.indices,
                       'dataset_dims'  : self.dims,
                       'dataset_rands' : self.rands,
                       'dataset_indices_queue' : self.indices_queue,
                       'dataset_dims_queue'  : self.dims_queue,
                       'dataset_rands_queue' : self.rands_queue,
                       'dataset_rng_state' : self.rng_state,   
                     }
        return state_dict
        
    def load_state_dict(self, state_dict):
        self.indices = state_dict['dataset_indices']
        self.dims = state_dict['dataset_dims']
        self.rands = state_dict['dataset_rands']      
        self.indices_queue = state_dict['dataset_indices_queue']
        self.dims_queue = state_dict['dataset_dims_queue']
        self.rands_queue = state_dict['dataset_rands_queue']
        self.rng_state = state_dict['dataset_rng_state']
    
    # Remove indices based on given batches or idx
    def trimm(self, idx=None, batch_idx=None):
        if batch_idx is not None and idx is None:
            offset_batch_idx = batch_idx % self.indices_batch
            if offset_batch_idx == 0:
                self.indices = []
                self.dims = []
                self.rands = []
                #print("Trim idx:{}, batch_idx:{}".format(idx, offset_batch_idx))
            else:
                offset = self.indices_size - len(self.indices)
                idx = (offset_batch_idx * self.batch_size) - offset 
                
                self.indices = self.indices[idx:]
                self.dims = self.dims[idx:]
                self.rands = self.rands[idx:]
                #print("Trim idx:{}, offset:{}, batch_idx:{}, len(self.indices):{} ".format(idx, offset, offset_batch_idx, len(self.indices)))

    def __len__(self):
        return len(self.indices)
       
    def __getitem__(self, idx):
        sel_idx = self.base_indices[self.indices[idx]]
        return str(sel_idx) + '-' + str(self.dims[idx][0]) + '-' + str(self.rands[idx])
        
class COCODataset(RandomCyclicDataset):
    
    def __init__(self, targ_txt_path, batch_size, shuffle=True, cyclic=True, dim=None, rand_dim_interval=None,
                 trans_fn=None, subset_idx=None):
        self.trans_fn = trans_fn
        self.subset_idx = subset_idx
        self.img_list, self.label_list = self._get_images_and_labels(targ_txt_path)
        super().__init__(batch_size, shuffle, cyclic, dim, rand_dim_interval)
        
    def get_base_indices(self):
        length = len(self.img_list)
        base_indices = torch.arange(0, length, dtype=torch.int64)
        if self.subset_idx is not None:
            base_indices = base_indices[self.subset_idx]
        return base_indices.tolist()
    
    def _get_images_and_labels(self, targ_txt_path):
        with open(targ_txt_path, 'r') as f:
            img_list = [lines.strip() for lines in f.readlines()]
        label_list = [img_path.replace('jpg', 'txt').replace('images', 'labels') for img_path in img_list]
        return img_list, label_list
    
    def __getitem__(self, idx):
        label = None
        
        seed = self.rands[idx]
        ia.seed(seed)
        np.random.seed(seed)
        
        dim = self.dims[idx]
        transform = self.trans_fn(dim)
        
        sel_idx = self.base_indices[self.indices[idx]]
        img_path = self.img_list[sel_idx]
        if osp.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), img_path)
        
        label_path = self.label_list[sel_idx]
        if osp.exists(label_path):
            label = np.loadtxt(label_path).reshape(-1,5)
        
        sample = { 'img': img, 'org_img': img.copy(), 'label': label, 'transform': None, 'img_path': img_path }
        sample = transform(sample)
        return sample

class CVATDataset(RandomCyclicDataset):
    def __init__(self, img_dir, label_xml_path, batch_size, subset_idx=None, trans_fn=None,
                 shuffle=True, cyclic=True, dim=None, rand_dim_interval=None):
        self.img_dir = img_dir
        self.label_xml_path = label_xml_path
        
        self.trans_fn = trans_fn
        self.subset_idx = subset_idx
        self.trans_fn = trans_fn

        self.class2id = { 'x_wing': 0, 'tie': 1}
        self.id2class = {v:k for k,v in self.class2id.items()}
        self.xml_dict = list(get_xml_labels(self.label_xml_path).items())
        
        super().__init__(batch_size, shuffle, cyclic, dim, rand_dim_interval)
        
    def get_base_indices(self):
        length = len(self.xml_dict)
        base_indices = torch.arange(0, length, dtype=torch.int64)
        if self.subset_idx is not None:
            base_indices = base_indices[self.subset_idx]
        return base_indices.tolist()
    
    def __getitem__(self, idx):
        label = None

        seed = self.rands[idx]
        ia.seed(seed)
        np.random.seed(seed)
        
        dim = self.dims[idx]
        transform = self.trans_fn(dim)
        
        sel_idx = self.base_indices[self.indices[idx]]
        
        img_path, label = self.xml_dict[sel_idx]
        
        img_path = osp.join(self.img_dir, img_path)
        if osp.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), img_path)
        img_dim = img.shape[1], img.shape[0]
        
        bbs_idx = np.arange(1,5)
        label = np.array( [ [self.class2id[l['cls']],
                                             l['x1'],
                                             l['y1'],
                                             l['x2'],
                                             l['y2'] ] for l in label] ).astype(np.float)
        label = BoundingBoxConverter.convert(label,
                                             CoordinateType.Absolute, FormatType.x1y1x2y2,
                                             CoordinateType.Relative, FormatType.cxcywh,
                                             bbs_idx, img_dim)
         
        sample = { 'img': img, 'org_img': img.copy(), 'label': label, 'img_path': img_path }
        sample = transform(sample)
        return sample

class ImageFolderDataset(Dataset):
    def __init__(self, img_dir, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        
        self.img_list = os.listdir(img_dir)
        
    def __getitem__(self, idx):
        img_path = osp.join(self.img_dir, self.img_list[idx])
        if osp.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), img_path)

        sample = { 'img': img, 'org_img': img, 'label':None }
        if self.transforms is not None:
            sample = self.transforms(sample)
            
        return sample
    
    def __len__(self):
        return len(self.img_list)




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

# DataHelper - Manages dataset and dataloader, provides the iterator for getting batches
class DataHelper():
    def __init__(self, dataset, dataloader, current_batch=0, current_epoch=0,
                 max_net_batches=None, max_batches=None, net_subdivisions=1):
        self.dataset = dataset
        self.dataloader = dataloader
        
        self.batch_size = self.dataset.batch_size
        self.current_batch = current_batch
        self.current_epoch = current_epoch
        self.net_subdivisions = net_subdivisions
        
        if max_net_batches is not None:
            self.max_net_batches = max_net_batches
            self.max_batches = self.max_net_batches * self.net_subdivisions
        elif max_batches is not None:
            self.max_batches = max_batches
        else:
            self.max_batches = self.dataset.indices_batch
        
        self.iterator = None

    def __iter__(self):
        if self.iterator is None:
            self.iterator = iter(self.gen())
        return self.iterator

    def gen(self):
        while self.current_batch < self.max_batches:
            for i in self.dataloader:
                yield(i)
                self.current_batch += 1
                if self.current_batch >= self.max_batches:
                    break
            self.dataset.randomize()

    def get_state_dict(self):
        state_dict = { 'current_batch' : self.current_batch,
                       'dataset': self.dataset.get_state_dict(),
                     }
        return state_dict

    def load_state_dict(self, state_dict):
        self.iterator = None
        self.current_batch = state_dict['current_batch'] + 1
        self.dataset.load_state_dict(state_dict['dataset'])
        # Remove processed indices from the dataset
        self.dataset.trimm(batch_idx=self.current_batch)

    def reset(self):
        self.iterator = None
        self.current_batch = 0
        return self

    # Helper functions to get batch information
    def get_batch(self):
        return self.current_batch

    def get_net_batch(self):
        return self.current_batch // self.net_subdivisions
    
    def get_epoch(self):
        return self.current_batch // self.get_epoch_num_batches()

    def get_epoch_batch(self):
        return self.current_batch % self.get_epoch_num_batches()
    
    def get_epoch_num_batches(self):
        return self.dataset.indices_batch

    def isStartOfEpoch(self):
        return ((self.current_batch) % self.get_epoch_num_batches()) == 0

    def isEndOfEpoch(self):
        return ((self.current_batch + 1) % self.get_epoch_num_batches()) == 0
    
    


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
    # If all items are 'None' in the list, return None. 
    elif batch[0] is None and (batch.count(batch[0]) == len(batch)):
        return None
    print(batch)
    print(len(batch))

    raise TypeError((error_msg.format(type(batch[0]))))

# Use worker_init_fn to reinitialize seed across multiprocessing workers (ensure deterministic behaviour)
def worker_init_fn(worker_id):
    base_seed = int(torch.randint(2**32, (1,)).item())
    lib_seed = (base_seed + worker_id) % (2**32)
    ia.seed(lib_seed)
    np.random.seed(lib_seed)