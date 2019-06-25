import re
import os.path as osp
import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import cv2
from enum import Enum
from boundingbox import *

def letterbox_label(label, transform, dim):
    label_x_offset = transform[..., 2] / dim[0]
    label_y_offset = transform[..., 3] / dim[1]
    box_w_ratio = transform[..., 0] / dim[0]
    box_h_ratio = transform[..., 1] / dim[1]
    label[..., [0,2]] = label[..., [0,2]] * box_w_ratio 
    label[..., [1,3]] = label[..., [1,3]] * box_h_ratio
    label[..., 0] = label[..., 0] + label_x_offset
    label[..., 1] = label[..., 1] + label_y_offset
    return label

def letterbox_label_reverse(label, transform, dim):
    label_x_offset = transform[..., 2] / dim[0]
    label_y_offset = transform[..., 3] / dim[1]
    box_w_ratio = transform[..., 0] / dim[0]
    box_h_ratio = transform[..., 1] / dim[1]
    label[..., 0] = label[..., 0] - label_x_offset
    label[..., 1] = label[..., 1] - label_y_offset
    label[..., [0,2]] = torch.clamp(label[..., [0,2]] / box_w_ratio, 0, 1) 
    label[..., [1,3]] = torch.clamp(label[..., [1,3]] / box_h_ratio, 0, 1)
    return label

def letterbox_transforms(inner_dim, outer_dim):
    outer_w, outer_h = outer_dim
    inner_w, inner_h = inner_dim
    ratio = min(outer_w / inner_w, outer_h / inner_h)
    box_w = int(inner_w * ratio)
    box_h = int(inner_h * ratio)
    box_x_offset = (outer_w // 2) - (box_w // 2)
    box_y_offset = (outer_h // 2) - (box_h // 2)
    return box_w, box_h, box_x_offset, box_y_offset, ratio

def letterbox_image(img, dim):
    #Create the background
    image = np.full(dim +(3,), 128)
        
    img_dim = (img.shape[1], img.shape[0])
    box_w, box_h, box_x, box_y, ratio = letterbox_transforms(img_dim, dim)
    box_image = cv2.resize(img, (box_w,box_h), interpolation = cv2.INTER_CUBIC)
        
    #Put the box image on top of the blank image
    image[box_y:box_y+box_h, box_x:box_x+box_w] = box_image
    
    transform = Tensor([box_w, box_h, box_x, box_y, ratio])
    return image, transform

# Mode - letterbox, resize
def load_image(img_path, mode=None, dim=None):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
  
    trans = None
    if mode is not None and dim is not None:
        if mode == 'letterbox':
            img, trans = letterbox_image(img, dim)
        elif mode == 'resize':
            img = cv2.resize(img, dim)
    
    img = torch.from_numpy(img).float().permute(2,0,1) / 255
    return img, trans


def torch_unique(inp, CUDA=True):
    if CUDA:
        inp_cpu = inp.detach().cpu()
    
    res_cpu = torch.unique(inp_cpu)
    res = inp.new(res_cpu.shape)
    res.copy_(res_cpu)
    
    return res


def unqiue_with_order(inp, CUDA=True):
    if CUDA:
        inp_np = inp.detach().cpu().numpy()
    
    _, idx = np.unique(inp, return_index=True)
    result = inp_np[np.sort(idx)]
    result_tensor = torch.from_numpy(result)
    res = inp.new(result_tensor.shape)
    res.copy_(result_tensor)
    return res


def iou_vectorized(bbox):
    num_box = bbox.shape[0]
    
    bbox_leftTop_x =  bbox[:,0]
    bbox_leftTop_y =  bbox[:,1]
    bbox_rightBottom_x = bbox[:,2]
    bbox_rightBottom_y = bbox[:,3]
    
    #print(bbox_leftTop_x.shape)
    #print(bbox_leftTop_x.unsqueeze(1).repeat(1,num_box).shape)
    
    inter_leftTop_x     =  torch.max(bbox_leftTop_x.unsqueeze(1).repeat(1,num_box), bbox_leftTop_x)
    inter_leftTop_y     =  torch.max(bbox_leftTop_y.unsqueeze(1).repeat(1,num_box), bbox_leftTop_y)
    inter_rightBottom_x =  torch.min(bbox_rightBottom_x.unsqueeze(1).repeat(1,num_box), bbox_rightBottom_x)
    inter_rightBottom_y =  torch.min(bbox_rightBottom_y.unsqueeze(1).repeat(1,num_box), bbox_rightBottom_y)
    
    inter_area = torch.clamp(inter_rightBottom_x - inter_leftTop_x, min=0) * torch.clamp(inter_rightBottom_y - inter_leftTop_y, min=0)
    bbox_area = (bbox_rightBottom_x - bbox_leftTop_x) * (bbox_rightBottom_y - bbox_leftTop_y)
    union_area = bbox_area.expand(num_box,-1) + bbox_area.expand(num_box,-1).transpose(0, 1) - inter_area
    
    iou = inter_area / union_area
    return iou

# mode - x1y1x2y2, cxcywh
def bbox_iou(b1, b2, mode="x1y1x2y2"):
    if mode == "x1y1x2y2":
        b1_x1, b1_y1, b1_x2, b1_y2 = b1[...,0], b1[...,1], b1[...,2], b1[...,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = b2[...,0], b2[...,1], b2[...,2], b2[...,3]  
    elif mode == "cxcywh":
        b1_x1, b1_x2 = b1[..., 0] - b1[..., 2] / 2, b1[..., 0] + b1[..., 2] / 2
        b1_y1, b1_y2 = b1[..., 1] - b1[..., 3] / 2, b1[..., 1] + b1[..., 3] / 2
        b2_x1, b2_x2 = b2[..., 0] - b2[..., 2] / 2, b2[..., 0] + b2[..., 2] / 2
        b2_y1, b2_y2 = b2[..., 1] - b2[..., 3] / 2, b2[..., 1] + b2[..., 3] / 2
    
    num_b1 = b1.shape[0]
    num_b2 = b2.shape[0]
    
    inter_x1 = torch.max(b1_x1.unsqueeze(1).repeat(1, num_b2), b2_x1)
    inter_y1 = torch.max(b1_y1.unsqueeze(1).repeat(1, num_b2), b2_y1)
    inter_x2 = torch.min(b1_x2.unsqueeze(1).repeat(1, num_b2), b2_x2)
    inter_y2 = torch.min(b1_y2.unsqueeze(1).repeat(1, num_b2), b2_y2)
            
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area.unsqueeze(1).repeat(1, num_b2) + b2_area.unsqueeze(0).repeat(num_b1, 1) - inter_area
    
    iou = inter_area / union_area
    return iou

def get_nms_detections(detections, detection_idx, num_classes, obj_conf_thr, nms_thr):
    nB = detections.shape[0]
    results = list()

    for batch_idx in range(nB):
        batch_results = torch.Tensor()
        # Select detections for this image
        det_idx_mask = detection_idx[:, 0] == batch_idx
        if not det_idx_mask.any():
            results.append(batch_results)
            continue

        # Find the detected classes with unique()
        img_classes = detection_idx[det_idx_mask][:, 2].unique()
        for c in img_classes:
            # Select detections with "c" class
            cls_index = detection_idx[det_idx_mask & (detection_idx[:, 2] == c)]
            if len(cls_index) == 0:
                continue

            det_img_class = detections[cls_index[:,0], cls_index[:,1]]

            # Sort by detection prob
            _, sort_idx = det_img_class[:, 5+c].sort(descending=True)
            det_img_class = det_img_class[sort_idx]

            # Get iou
            iou = iou_vectorized(det_img_class)
            # Find iou > nms threshold
            iou = iou > nms_thr

            # Iterate each detection by rows
            for idx in (range(len(iou))):
                # Ignore if diagonal element is 0
                if iou[idx, idx].item() == 0:
                    continue

                # Find detection with (iou > nms_thr)
                cols = idx + 1
                # Only need to check upper diagonal half of the matrix
                ignore_idx = iou[idx, cols:].nonzero().squeeze() + cols
                # Set rows and cols to 0 for detections in ignore_idx
                iou[ignore_idx, :], iou[:, ignore_idx] = 0, 0
            
            # Valid detections are marked as 1 along the diagonal vector 
            selected = iou.diagonal().nonzero().squeeze()
            det_img_class = det_img_class[selected].view(-1, 5+num_classes)
            det_img_class = torch.cat((det_img_class[:, :5], # box and objectness
                                       det_img_class[:, 5+c].view(-1, 1), # detection prob
                                       Tensor([c]).repeat(len(det_img_class), 1) ), -1) # class
            # Add class detections to batch_results
            batch_results = torch.cat((batch_results, det_img_class), 0)

        results.append(batch_results)
    return results

def get_raw_detections(detections, index):
    nB = detections.shape[0]
    results = list()

    for batch_idx in range(nB):
        batch_results = torch.Tensor()
        # Select detections for this image
        det_idx_mask = index[:, 0] == batch_idx
        if not det_idx_mask.any():
            results.append(batch_results)
            continue

        selected = index[det_idx_mask]
        bbox_obj= detections[selected[:, 0], selected[:, 1], :5]
        prob = detections[selected[:, 0], selected[:, 1], selected[:, 2]+5]
        cls = selected[:, 2].float()
        batch_results = torch.cat((bbox_obj[:, :5],
                                   prob.unsqueeze(-1),
                                   cls.unsqueeze(-1)), -1)
        results.append(batch_results)
    return results

def postprocessing(detections, num_classes, obj_conf_thr=0.5, nms_thr=0.4, is_eval=False, use_nms=True):
    detections = detections.cpu()
          
    # Transform bounding box coordinates from cxcywh to x1y1x2y2
    detections[..., :4] = bbox_cxcywh_to_x1y1x2y2(detections[..., :4])
    
    # detection prob = class prob * objectness
    detections[..., 5: 5+num_classes] = detections[..., 5:5+num_classes] * detections[..., 4].unsqueeze(-1)

    # Allow multiple classes assigned to a single detection box. Used for mAP evaluation
    if is_eval:
        # Get the detections with detection prob > obj_conf_thr
        index = (detections[..., 5: 5+num_classes] > obj_conf_thr).nonzero()
    # Allow only one class assigned to a single detection box. Used for image output
    else:
        # Find max detection prob and filter by obj_conf_thr
        max_class_score, max_class_idx= torch.max(detections[..., 5:5+num_classes], -1)
        index_mask = max_class_score > obj_conf_thr
        if index_mask.any():
            index = torch.cat((index_mask.nonzero(),
                           max_class_idx[index_mask].unsqueeze(-1)), -1)
        else:
            return []

    if len(index) == 0:
        return []
    
    if use_nms:
        results = get_nms_detections(detections, index, num_classes, obj_conf_thr, nms_thr)
    else:
        results = get_raw_detections(detections, index)

    return results

def get_image_shape(img):
    if isinstance(img, tuple):
        return img
    else:
        return img.shape[1], img.shape[0]


def fill_label_np_tensor(label, row, col):
    label_tmp = np.full((row, col), 0.0)
    if label is not None and len(label) != 0 :
        length = label.shape[0] if label.shape[0] < row else row
        label_tmp[:length] = label[:length]
    return label_tmp


# Mask rows and columns given the source tensor
def build_2D_mask(src, rows_idx, cols_idx):
    nH, nW = src.shape[0], src.shape[1]
    rows_mask = torch.zeros_like(src).byte()
    rows_mask[rows_idx] = 1
    cols_mask = torch.zeros_like(src).byte()
    cols_mask[..., cols_idx] = 1
    mask = rows_mask * cols_mask
    return mask.byte()


# Exponential weighted moving average

def ewma_online(new_value, previous_average, window):
    alpha = 2 /(window + 1.0)
    new_average = alpha * new_value + (1 - alpha) * previous_average
    return new_average


def get_image_id_from_path(image_path):
    image_path = osp.splitext(image_path)[0]
    m = re.search(r'\d+$', image_path)
    return int(m.group())