import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import torch

from utils import bbox_x1y1x2y2_to_cxcywh, bbox_cxcywh_to_x1y1x2y2


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

class Rotate():
    def __init__(self, rotate, mode="constant", cval=128, bbs_idx=np.arange(0,4)):
        self.rotate = rotate
        self.mode = mode
        self.cval = cval
        self.bbs_idx = bbs_idx

        seq = iaa.Sequential([iaa.Affine(rotate=self.rotate, mode=self.mode, cval=self.cval)])
        self.seq_det = seq.to_deterministic()

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        img_w, img_h = img.shape[1], img.shape[0]

        img = self.seq_det.augment_images([img])[0]

        if label is not None:
            label[..., self.bbs_idx] = bbox_cxcywh_to_x1y1x2y2(label[..., self.bbs_idx])

            bbs = label_np_to_bbs(label, shape=(img_h, img_w), bbs_idx=self.bbs_idx)
            bbs = self.seq_det.augment_bounding_boxes([bbs])[0]
            bbs = bbs.remove_out_of_image().cut_out_of_image()

            if len(bbs.bounding_boxes) != 0:
                label = label_bbs_to_np(bbs, bbs_idx=self.bbs_idx)
                label[..., self.bbs_idx] = bbox_x1y1x2y2_to_cxcywh(label[..., self.bbs_idx])
            else:
                label = None

        sample.update({'img': img, 'label': label})
        return sample

class ToTensor():
    def __init__(self, max_labels=50, max_label_cols=5):
        self.max_labels = max_labels
        self.max_label_cols = max_label_cols

    def __call__(self, sample):
        img, org_img, label = sample.get('img', None), sample.get('org_img', None), sample.get('label', None)

        img = torch.from_numpy(img).float().permute(2,0,1) / 255.0 if img is not None else None
        org_img = torch.from_numpy(org_img).float().permute(2,0,1) / 255.0 if org_img is not None else None
        label = torch.from_numpy(fill_label_np_tensor(label, self.max_labels, self.max_label_cols))

        sample.update({'img': img, 'org_img': org_img, 'label': label})
        return sample

class BasicAugmentations():
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.seq = iaa.Sequential(
            [
                # Blur each image with varying strength using
                # gaussian blur (sigma between 0 and 3.0),
                # average/uniform blur (kernel size between 2x2 and 7x7)
                # median blur (kernel size between 3x3 and 11x11).
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),
                # Sharpen each image, overlay the result with the original
                # image using an alpha between 0 (no sharpening) and 1
                # (full sharpening effect).
                sometimes(iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5))),
                # Add gaussian noise to some images.
                sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
                # Add a value of -5 to 5 to each pixel.
                sometimes(iaa.Add((-5, 5), per_channel=0.5)),
                # Change brightness of images (80-120% of original value).
                sometimes(iaa.Multiply((0.8, 1.2), per_channel=0.5)),
                # Improve or worsen the contrast of images.
                sometimes(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5))
            ],
            # do all of the above augmentations in random order
            random_order=True
        )

    def __call__(self, sample):
        seq_det = self.seq.to_deterministic()
        img, label = sample['img'], sample['label']
        img = seq_det.augment_images([img])[0]
        sample.update({'img': img, 'label': label})
        return sample

class Letterbox:
    """Letterbox image and labels
    Args:
        new_dim: target dimension (weight, height)
        box_coord_scale: Given (bx, bx, bw, bh) with image of (w, h)
                         'pixel' -- labels value are interpreted as pixels (bx, bx, bw, bh)
                         'ratio' -- labels value are interpreted as ratio (bx/w, bx/h, bw/w, bh/h)
    """
    def __init__(self, new_dim, box_coord_scale='ratio', bbs_idx=np.arange(0,4)):
        self.new_dim = new_dim
        self.box_coord_scale = box_coord_scale
        self.bbs_idx = bbs_idx
        
    def __call__(self, sample):
        org_img = sample['img']
        org_w, org_h = org_img.shape[1], org_img.shape[0]
        new_w, new_h = self.new_dim[1], self.new_dim[0]
        resize_w, resize_h, x_off, y_off, ratio = letterbox_transforms(org_w, org_h, new_w, new_h)
        resize_img = cv2.resize(org_img, (resize_w, resize_h), interpolation = cv2.INTER_CUBIC)
        
        #Put the box image on top of the blank image
        img = np.full(self.new_dim +(3,), 128)
        img[y_off:y_off+resize_h, x_off:x_off+resize_w] = resize_img
        
        labels = sample['label']
        if sample['label'] is not None:
            if self.box_coord_scale == 'pixel':
                x_idx, y_idx = self.bbs_idx[[0,2]], self.bbs_idx[[1,3]] 
                labels[..., x_idx] = labels[..., x_idx] / org_w
                labels[..., y_idx] = labels[..., y_idx] / org_h

        x_off, y_off = x_off / new_w, y_off / new_h

        if labels is not None:
            labels = letterbox_labels(labels, x_off, y_off, (resize_w, resize_h), (new_w, new_h), self.bbs_idx)

        transform = np.array([x_off, y_off, resize_w, resize_h,new_w, new_h])
        sample['img'], sample['label'], sample['transform'] = img, labels, transform
        return sample
   

def letterbox_labels(labels, x_off, y_off, resize_dim, new_dim, bbs_idx=np.arange(0,4)):
    if isinstance(labels, torch.Tensor): 
        labels = labels.clone()
    elif isinstance(labels, np.ndarray):
        labels = labels.copy()
    else:
        raise TypeError("Labels must be a numpy array or pytorch tensor")
    ratio_x = resize_dim[0] / new_dim[0]
    ratio_y = resize_dim[1] / new_dim[1]
    x_idx, y_idx, cxw_idx, cyh_idx = bbs_idx[0], bbs_idx[1], bbs_idx[[0,2]], bbs_idx[[1,3]]    
    labels[..., cxw_idx] *= ratio_x 
    labels[..., cyh_idx] *= ratio_y 
    labels[..., x_idx] += x_off
    labels[..., y_idx] += y_off

    return labels

def letterbox_label_reverse(labels, x_off, y_off, resize_dim, new_dim, bbs_idx=np.arange(0,4)):
    if isinstance(labels, torch.Tensor):
        labels = labels.clone()
    elif isinstance(labels, np.ndarray):
        labels = labels.copy()
    else:
        raise TypeError("Labels must be a numpy array or pytorch tensor")
    ratio_x = resize_dim[0] / new_dim[0]
    ratio_y = resize_dim[1] / new_dim[1]
    x_idx, y_idx, cxw_idx, cyh_idx = bbs_idx[0], bbs_idx[1], bbs_idx[[0,2]], bbs_idx[[1,3]]  
    labels[..., x_idx] -= x_off
    labels[..., y_idx] -= y_off
    labels[..., cxw_idx] = np.clip((labels[..., cxw_idx] ) / ratio_x, 0, 1) 
    labels[..., cyh_idx] = np.clip((labels[..., cyh_idx] ) / ratio_y, 0, 1)
    return labels

def letterbox_transforms(org_w, org_h, new_w, new_h):
    ratio = min(new_w / org_w, new_h / org_h)
    resize_w, resize_h = int(org_w * ratio), int(org_h * ratio)
    x_off, y_off = (new_w - resize_w) // 2, (new_h - resize_h) // 2
    return resize_w, resize_h, x_off, y_off, ratio

def fill_label_np_tensor(label, row, col):
    label_tmp = np.full((row, col), 0.0)
    if label is not None:
        length = label.shape[0] if label.shape[0] < row else row
        label_tmp[:length] = label[:length]
    return label_tmp


def label_np_to_bbs(label, shape, bbs_idx=np.arange(0,4)):
    bbs = [ia.BoundingBox(x1=b[bbs_idx[0]], y1=b[bbs_idx[1]], x2=b[bbs_idx[2]], y2=b[bbs_idx[3]], label=b) for b in label]
    bbs = ia.BoundingBoxesOnImage(bbs, shape=shape)
    return bbs

def label_bbs_to_np(bbs, bbs_idx=np.arange(0,4)):
    if len(bbs.bounding_boxes) == 0:
        return None

    row = len(bbs.bounding_boxes)
    col = bbs.bounding_boxes[0].label.shape[0]
    label = np.zeros((row, col))
    for i, b in enumerate(bbs.bounding_boxes):
        label_row = b.label
        label_row[..., bbs_idx] = [b.x1, b.y1, b.x2, b.y2]
        label[i] = label_row
    return label