import copy
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import torch

from boundingbox import bbox_x1y1x2y2_to_cxcywh, bbox_cxcywh_to_x1y1x2y2, BoundingBoxConverter, CoordinateType, FormatType
from utils import get_image_shape


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor():
    def __init__(self, max_labels=50, max_label_cols=5):
        self.max_labels = max_labels
        self.max_label_cols = max_label_cols

    def __call__(self, sample):
        img, org_img, label = sample.get('img', None), sample.get('org_img', None), sample.get('label', None)
        lb_reverter = sample.get('lb_reverter', None)


        img = torch.from_numpy(img).float().permute(2,0,1) / 255.0 if img is not None else None
        org_img = torch.from_numpy(org_img).float().permute(2,0,1) / 255.0 if org_img is not None else None
        label = torch.from_numpy(fill_label_np_tensor(label, self.max_labels, self.max_label_cols)).float()
        lb_reverter = torch.from_numpy(lb_reverter).float() if lb_reverter is not None else None
        
        update = {'img': img, 'org_img': org_img, 'label': label, 'lb_reverter': lb_reverter}
        sample.update({k:v for k,v in update.items() if v is not None})
        sample.pop('transform', None)
        sample.pop('bbs', None)
        return sample

def fill_label_np_tensor(label, row, col):
    label_tmp = np.full((row, col), 0.0)
    if label is not None:
        length = label.shape[0] if label.shape[0] < row else row
        label_tmp[:length] = label[:length]
    return label_tmp

class ToNp():
    def __init__(self, bbs_idx=[1,2,3,4]):
        self.bbs_idx = bbs_idx

    def __call__(self, sample):
        img, label, bbs = sample['img'], sample['label'], sample['bbs']
        if label is not None:
            label = label_bbs_to_np(bbs, bbs_idx=self.bbs_idx)
        sample.update({'img': img.copy(), 'label': label, 'bbs':bbs})
        return sample

class ToIaa():
    def __init__(self, bbs_idx=[1,2,3,4], seed=None):
        if seed is None:
            seed = np.random.randint(0, 2**16)

        self.bbs_idx = bbs_idx
        self.seed = seed
        
        ia.seed(self.seed)

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        bbs = None
        if label is not None:
            bbs = label_np_to_bbs(label, img.shape, bbs_idx=self.bbs_idx)
        sample.update({'img': img, 'label': label, 'bbs':bbs})
        return sample

class IaaAugmentations():
    def __init__(self, aug_list):
        self.seq = iaa.Sequential(aug_list)


    def __call__(self, sample):
        img, label, bbs = sample['img'], sample['label'], sample['bbs']
        seq = self.seq.to_deterministic()
        img, bbs = iaa_run_seq(seq, img, bbs)
        sample.update({'img': img, 'label': label, 'bbs':bbs})
        return sample

def label_np_to_bbs(label, shape, bbs_idx=np.arange(0,4)):
    # Filter bounding boxes that doesn't satisfy x2 > x1 and y2 > y1 
    bbs = ((b[bbs_idx[0]], b[bbs_idx[1]], b[bbs_idx[2]], b[bbs_idx[3]], b) for b in label if b[bbs_idx[2]] > b[bbs_idx[0]] and b[bbs_idx[3]] > b[bbs_idx[1]])
    bbs = [ia.BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3], label=b[4]) for b in bbs]
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


"""
    Follow darknet format:darknet/src/http_stream.cpp
    
         hsv[1] *= dsat; 
         hsv[2] *= dexp; 
         hsv[0] += 179 * dhue; 

"""
def iaa_hsv_aug(hue=0, saturation=1, exposure=1):
    h_l, h_t = 179 * -hue, 179 * hue
    s_l, s_t = 1 / saturation if saturation else 0, saturation
    v_l, v_t = 1 / exposure if exposure else 0, exposure
    
    return iaa.Sequential([iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                           iaa.WithChannels([0], iaa.Add((h_l, h_t))),
                           iaa.WithChannels([1], iaa.Multiply((s_l, s_t))),
                           iaa.WithChannels([2], iaa.Multiply((v_l, v_t))),
                           iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")])

"""
    From darknet/src/data.c

        int dw = (ow*jitter); 
        int dh = (oh*jitter); 

        int pleft  = rand_uniform_strong(-dw, dw); 
        int pright = rand_uniform_strong(-dw, dw); 
        int ptop   = rand_uniform_strong(-dh, dh); 
        int pbot   = rand_uniform_strong(-dh, dh); 

        int swidth =  ow - pleft - pright; 
        int sheight = oh - ptop - pbot; 
"""
def iaa_random_crop(jitter):
    return iaa.Sequential([iaa.Crop(None, ((-jitter,jitter),(-jitter,jitter), (-jitter,jitter), (-jitter,jitter)), keep_size=False)])

def iaa_letterbox(img, new_dim):
    if isinstance(img, tuple):
        org_dim = img
    else:
        org_dim = img.shape[1], img.shape[0]
    
    padded_w, padded_h, x_pad, y_pad, ratio = letterbox_transforms(*org_dim, *new_dim)
    l_pad, r_pad = x_pad, new_dim[0] - padded_w - x_pad
    t_pad, b_pad = y_pad, new_dim[1] - padded_h - y_pad
    lb_reverter = np.array([org_dim[0], org_dim[1], padded_w, padded_h, x_pad, y_pad])

    return iaa.Sequential([iaa.Scale({ "width": padded_w, "height": padded_h }),
                           iaa.Pad(px=(t_pad, r_pad, b_pad, l_pad), keep_size=False, pad_cval=128),
                          ]), \
           lb_reverter
           #[x_pad, y_pad, org_dim[0] / padded_w, org_dim[1] / padded_h]

def letterbox_transforms(org_w, org_h, new_w, new_h):
    ratio = min(new_w / org_w, new_h / org_h)
    resize_w, resize_h = int(org_w * ratio), int(org_h * ratio)
    x_off, y_off = (new_w - resize_w) // 2, (new_h - resize_h) // 2
    return resize_w, resize_h, x_off, y_off, ratio

class IaaLetterbox():
    def __init__(self, new_dim):
        super().__init__()
        self.new_dim = new_dim

    def __call__(self, sample):
        org_img, label, bbs = sample['img'], sample['label'], sample['bbs']
        seq, lb_reverter = iaa_letterbox(org_img, self.new_dim)
        seq = seq.to_deterministic()
        img, bbs = iaa_run_seq(seq, org_img, bbs)
        sample.update({'img': img, 'org_img': org_img.copy(), 'label': label, 'bbs':bbs, 'lb_reverter':lb_reverter})
        return sample




def letterbox_reverter(labels, org_img, padded_dim, x_pad, y_pad, bbs_idx=np.array([1,2,3,4])):
    if isinstance(labels, torch.Tensor):
        labels = labels.clone()
    elif isinstance(labels, np.ndarray):
        labels = labels.copy()
    else:
        raise TypeError("Labels must be a numpy array or pytorch tensor")

    mask = labels.sum(-1) != 0
    labels = labels[mask]

    org_dim = get_image_shape(org_img)
    ratio_x, ratio_y = org_dim[1] / padded_dim[1], org_dim[0] / padded_dim[0]
    x_idx, y_idx = bbs_idx[[0,2]], bbs_idx[[1,3]] 
    labels[..., x_idx] = (labels[..., x_idx] - x_pad) * ratio_x
    labels[..., y_idx] = (labels[..., y_idx] - y_pad) * ratio_y

    return labels



def iaa_run_seq(seq, img, bbs=None):
    aug_bbs = None
    aug_img = seq.augment_images([img])[0]
    if bbs:
        aug_bbs = seq.augment_bounding_boxes([bbs])[0]
        aug_bbs = bbs_remove_cut_out(aug_bbs, 0.2)
    return aug_img, aug_bbs

"""
    Remove bounding boxes that are partially/fully out of the image
    Determine if the bounding box is kept by:
        bbs_clip.area = area of the clipped bounding box
        bbs_org.area = area of the original bounding box

        keep if (bbs_clip.area / bbs_org.area) > area_thr   
"""
def bbs_remove_cut_out(bbs, area_thr=0):
    bbs_clean = [bbs_clip(b, bbs.shape, area_thr) for b in bbs.bounding_boxes]
    bbs_clean = list(filter(None.__ne__, bbs_clean))
    return ia.BoundingBoxesOnImage(bbs_clean, shape=bbs.shape)

"""
    Clip bounding boxes that are outside of the image
"""
def bbs_clip(bbs, shape, area_thr):
    height, width = shape[0:2]

    eps = np.finfo(np.float32).eps
    x1 = np.clip(bbs.x1, 0, width - eps)
    x2 = np.clip(bbs.x2, 0, width - eps)
    y1 = np.clip(bbs.y1, 0, height - eps)
    y2 = np.clip(bbs.y2, 0, height - eps)

    area = (x2 - x1) * (y2 - y1)
    keep = area / bbs.area

    if keep > area_thr:
        return bbs.copy(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            label=bbs.label
        )
    else:
        return None

"""
    Wrap BoundingBoxConverter function into a transform object
"""
class BoundingBoxFormatConvert(object):
    def __init__(self, src_coord_type, src_format_type,
                       dest_coord_type, dest_format_type,
                       bbox_idx):
        self.src_coord_type = src_coord_type
        self.src_format_type = src_format_type
        self.dest_coord_type = dest_coord_type
        self.dest_format_type = dest_format_type
        self.bbox_idx = bbox_idx

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        img_dim = img.shape[1], img.shape[0]

        if label is not None:
            label = BoundingBoxConverter.convert(label,
                                                 self.src_coord_type, self.src_format_type,
                                                 self.dest_coord_type, self.dest_format_type,
                                                 self.bbox_idx,
                                                 img_dim
                                                )

        sample.update({'img': img, 'label': label})
        return sample

    def reverse(self):
        rev = copy.copy(self)
        rev.src_coord_type = self.dest_coord_type
        rev.src_format_type = self.dest_format_type
        rev.dest_coord_type = self.src_coord_type
        rev.dest_format_type = self.src_format_type
        return rev

class ToX1y1x2y2Abs(BoundingBoxFormatConvert):
    def __init__(self, src_coord_type, src_format_type, bbox_idx):
        super().__init__(src_coord_type, src_format_type,
                         CoordinateType.Absolute,
                         FormatType.x1y1x2y2,
                         bbox_idx)

class ToCxcywhRel(BoundingBoxFormatConvert):
    def __init__(self, src_coord_type, src_format_type, bbox_idx):
        super().__init__(src_coord_type, src_format_type,
                         CoordinateType.Relative,
                         FormatType.cxcywh,
                         bbox_idx)



"""
    Additional augmentations
"""
class ExtraAugmentations():
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
        seq_det = self.seq
        img, label = sample['img'], sample['label']
        img = seq_det.augment_images([img])[0]
        sample.update({'img': img, 'label': label})
        return sample




"""
    Legacy versions. To be removed
"""
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

