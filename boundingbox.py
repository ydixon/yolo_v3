import copy
import cv2
import numpy as np
import torch

# import imgaug as ia
# from imgaug import augmenters as iaa

# Bounding box transforms

def bbox_x1y1x2y2_to_xywh(box):
    bx, by = box[..., 0], box[..., 1]
    bw = box[..., 2] - box[..., 0]
    bh = box[..., 3] - box[..., 1]
    box[..., 0], box[..., 1], box[..., 2], box[..., 3] = bx, by, bw, bh
    return box

def bbox_x1y1x2y2_to_cxcywh(box):
    bw = box[..., 2] - box[..., 0]
    bh = box[..., 3] - box[..., 1]
    cx, cy = box[..., 0] + bw / 2, box[..., 1] + bh / 2
    box[..., 0], box[..., 1], box[..., 2], box[..., 3] = cx, cy, bw, bh
    return box

def bbox_cxcywh_to_x1y1x2y2(box):
    x1, x2 = box[..., 0] - box[..., 2] / 2, box[..., 0] + box[..., 2] / 2
    y1, y2 = box[..., 1] - box[..., 3] / 2, box[..., 1] + box[..., 3] / 2
    box[..., 0], box[..., 1], box[..., 2], box[..., 3] = x1, y1, x2, y2
    return box

def bbox_cxcywh_to_xywh(box):
    x, y = box[..., 0] - box[..., 2] / 2, box[..., 1] - box[..., 3] / 2
    box[..., 0], box[..., 1] = x, y
    return box

def bbox_xywh_to_cxcywh(box):
    raise NotImplementedError

def bbox_xywh_to_x1y1x2y2(box):
    raise NotImplementedError

def bbox_absolute_to_relative(box, img_dim, x_idx=[0,2], y_idx=[1,3]):
    box[..., x_idx] /= img_dim[0]
    box[..., y_idx] /= img_dim[1]
    return box

def bbox_relative_to_absolute(box, img_dim, x_idx=[0,2], y_idx=[1,3]):
    box[..., x_idx] *= img_dim[0]
    box[..., y_idx] *= img_dim[1]
    return box

class BoundingBoxConverter():
    bbox_format_converter = [[lambda x: x, bbox_x1y1x2y2_to_cxcywh, bbox_x1y1x2y2_to_xywh, ],
                             [bbox_cxcywh_to_x1y1x2y2, lambda x: x, bbox_cxcywh_to_xywh],
                             [bbox_xywh_to_x1y1x2y2, bbox_xywh_to_cxcywh, lambda x: x]]          #xywh not implementeted
    bbox_coord_converter = [[lambda x,y: x, bbox_absolute_to_relative],
                            [bbox_relative_to_absolute, lambda x,y: x]]
    
    def convert(labels,
                src_coord_type, src_format_type,
                dest_coord_type, dest_format_type,
                bbox_idx=[0,1,2,3],
                img_dim=None,
                inplace=False):
        if not inplace:
            if isinstance(labels, torch.Tensor): 
                labels = labels.clone()
            elif isinstance(labels, np.ndarray):
                labels = labels.copy()
            else:
                raise TypeError("Labels must be a numpy array or pytorch tensor")

        if len(labels) == 0:
            return labels

        box = labels[..., bbox_idx]
        box = BoundingBoxConverter.bbox_format_converter[src_format_type][dest_format_type](box)
        box = BoundingBoxConverter.bbox_coord_converter[src_coord_type][dest_coord_type](box, img_dim)
        labels[..., bbox_idx] = box
        return labels


# Types

class CoordinateType():
	Absolute = 0
	Relative = 1
	

class FormatType():
	x1y1x2y2 = 0
	cxcywh = 1
	xywh = 2
