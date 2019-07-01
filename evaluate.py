import os
import os.path as osp
import sys
import glob
import time
import timeit
import math
import random
import json
from contextlib import contextmanager

import itertools
from collections import OrderedDict
from tqdm import tqdm, tnrange, tqdm_notebook

import numpy as np
import pandas as pd
import cv2
import imgaug as ia
from imgaug import augmenters as iaa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split, Subset

from utils import *
from darknet import YoloNet
from draw import show_img, show_img_grid
from boundingbox import CoordinateType, FormatType, BoundingBoxConverter, correct_yolo_boxes
from transforms import IaaAugmentations, IaaLetterbox, ToTensor, Compose, \
                       iaa_hsv_aug, iaa_random_crop, iaa_letterbox
from dataset import worker_init_fn, variable_shape_collate_fn                       

# Pycocotools Format
#
# Create Ground Truth json

def create_annotations(cat_list, img_list, ann_list):
    return OrderedDict({'categories': cat_list,
                        'images': img_list,
                        'annotations': ann_list})

def create_images_entry(image_id, width=None, height=None):
    if width is None or height is None:
        return OrderedDict({'id':image_id })
    else:
        return OrderedDict({'id':image_id, 'width':width, 'height':height })

def create_categories(class_names):
    return [{'id':i, 'name':cls} for i, cls in enumerate(class_names)]

def create_annotations_entry(image_id, bbox, category_id, ann_id, iscrowd=0, area=None, segmentation=None):
    if area is None:
        if segmentation is None:
            #Calulate area with bbox
            area = bbox[2] * bbox[3]
        else:
            raise NotImplementedError()
            
    return OrderedDict({
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "iscrowd": iscrowd,
            "area": area,
            "bbox": bbox
           })

def generate_annotations_file(target_txt, class_names, out):
    ann_dict = create_annotations_dict(target_txt, class_names)
    with open(out, 'w') as f:
        json.dump(ann_dict, f, indent=4, separators=(',', ':'))

def create_annotations_dict(target_txt, class_names):
    with open(target_txt, 'r') as f:
        img_path_list = [lines.strip() for lines in f.readlines()]
    label_path_list = [img_path.replace('jpg', 'txt').replace('images', 'labels') for img_path in img_path_list]
    
    img_list, ann_list = get_img_ann_list(img_path_list, label_path_list)
    cat_list = create_categories(class_names)
    
    ann_dict = create_annotations(cat_list, img_list, ann_list)  
    return ann_dict

def get_img_ann_list(img_path_list, label_path_list):
    img_list, ann_list = [],[]
    for img_path, label_path in tqdm(zip(img_path_list, label_path_list), file=sys.stdout, leave=True, total=len(img_path_list)):
        image_id = get_image_id_from_path(img_path)
        # Read Image
        if osp.exists(img_path):
            img = cv2.imread(img_path)
        
        height, width = img.shape[0], img.shape[1]
        img_list.append(create_images_entry(image_id, width, height))
        # Read Labels
        if osp.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1,5)
        labels[..., 1:5] = BoundingBoxConverter.convert(labels[..., 1:5],
                                                     CoordinateType.Relative, FormatType.cxcywh,
                                                     CoordinateType.Absolute, FormatType.xywh,
                                                     img_dim=(width, height))
        #print(labels)
        for label in labels:
            category_id = int(label[0])
            bbox = list(label[1:5])
            ann_id = len(ann_list)
            ann_list.append(create_annotations_entry(image_id, bbox, category_id, ann_id))
            
    return img_list, ann_list

# Create Detection json

def create_results_entry(image_id, category_id, bbox, score):
    return OrderedDict({"image_id":image_id,
                        "category_id":category_id,
                        "bbox":bbox,
                        "score":score})

class COCOEvalDataset(Dataset):
    def __init__(self, targ_txt, dim=None, transform=None):
        with open(targ_txt, 'r') as f:
            self.img_list = [lines.strip() for lines in f.readlines()]
        self.label_list = [img_path.replace('jpg', 'txt').replace('images', 'labels') for img_path in self.img_list]
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

@contextmanager
def open_json_pred_writer(out_path, classes_names, is_letterbox=False):
    pred_writer = JsonPredictionWriter(out_path, classes_names, is_letterbox)
    try:
        pred_writer.write_start()
        yield pred_writer
    finally:
        pred_writer.write_end()

class BatchHandler:
    def process_batch(self, sample, predictions):
        raise NotImplementedError
        
class JsonPredictionWriter(BatchHandler):
    def __init__(self, out_path, classes_names, is_letterbox=False):
        self.out_path = out_path
        self.file = open(out_path, 'w')
        self.classes_names = classes_names
        self.is_letterbox = is_letterbox
        
    def write_start(self):
        self.file.write('[')
        
    def write_end(self):
        self.file.seek(self.file.tell() - 1, os.SEEK_SET)
        self.file.truncate()
        self.file.write(']')
        self.file.close()
            
    def process_batch(self, sample, predictions):
        imgs, org_imgs, img_paths = sample['img'], sample['org_img'], sample['img_path']
        for img, org_img, img_path, prediction in zip(imgs, org_imgs, img_paths, predictions):
            img_w, img_h, org_w, org_h = img.shape[2], img.shape[1], org_img.shape[2], org_img.shape[1]
            image_id = get_image_id_from_path(img_path)
            
            if prediction is not None and len(prediction) != 0:
                bboxes = correct_yolo_boxes(prediction[..., 0:4], org_w, org_h, img_w, img_h, self.is_letterbox)
                category_ids = prediction[..., 6]
                scores = prediction[..., 5]
                               
                for category_id, bbox, score in zip(category_ids, bboxes, scores):
                    category_id, bbox, score = int(category_id.item()), bbox.tolist(), score.item()
                    res = create_results_entry(image_id, category_id, bbox, score)
                    json.dump(res, self.file, indent=4, separators=(',', ':'))
                    self.file.write(',')

def predict_and_process(data, net, num_classes, batch_handler=None):
    with torch.no_grad(): 
        for sample in tqdm(data, file=sys.stdout, leave=True):           
            # Pass images to the network
            det1, det2, det3 = net(sample['img'].cuda(), None)
            predictions = postprocessing(torch.cat((det1,det2,det3), 1),
                                         num_classes, obj_conf_thr=0.005, nms_thr=0.45,
                                         is_eval=True, use_nms=True)
            # Batch Handler - write file
            batch_handler.process_batch(sample, predictions)

def generate_results_file(net, target_txt, classes_names, out, bs, dim, is_letterbox=False):
    numclass = len(classes_names)
    if is_letterbox:
        transform = Compose([IaaAugmentations([IaaLetterbox(dim)]), ToTensor()])
    else:
        transform = Compose([IaaAugmentations([iaa.Scale(dim)]), ToTensor()])

    ds = COCOEvalDataset(target_txt, dim, transform)
    dl = DataLoader(ds, batch_size=bs, num_workers=4, collate_fn=variable_shape_collate_fn)

    with open_json_pred_writer(out, classes_names, is_letterbox) as pred_writer:
        predict_and_process(dl, net, num_classes=numclass, batch_handler=pred_writer)