import copy
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmenters import Augmenter
import six.moves as sm

import numpy as np
import torch

from boundingbox import bbox_x1y1x2y2_to_cxcywh, bbox_cxcywh_to_x1y1x2y2, BoundingBoxConverter, CoordinateType, FormatType
from utils import get_image_shape, build_2D_mask, fill_label_np_tensor


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor():
    def __init__(self, max_labels=90, max_label_cols=5):
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

class IaaAugmentations():
    def __init__(self, aug_list, bbs_idx=[1,2,3,4]):
        self.seq = iaa.Sequential(aug_list)
        self.bbs_idx = bbs_idx

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        img_dim = img.shape[1], img.shape[0]
        bbs = None

        if label is not None:
            label = BoundingBoxConverter.convert(label,
                                             CoordinateType.Relative, FormatType.cxcywh,
                                             CoordinateType.Absolute, FormatType.x1y1x2y2,
                                             bbox_idx=self.bbs_idx, img_dim=img_dim)

            bbs = label_np_to_bbs(label, img.shape, bbs_idx=self.bbs_idx)

        seq = self.seq.to_deterministic()
        img, bbs = iaa_run_seq(seq, img, bbs)

        if label is not None:
            label = label_bbs_to_np(bbs, bbs_idx=self.bbs_idx)
            img_dim = img.shape[1], img.shape[0]
            label = BoundingBoxConverter.convert(label,
                                                 CoordinateType.Absolute, FormatType.x1y1x2y2,
                                                 CoordinateType.Relative, FormatType.cxcywh,
                                                 bbox_idx=self.bbs_idx, img_dim=img_dim)

        sample.update({'img': img, 'label': label, 'bbs':bbs})
        return sample

def rand_uniform(val1, val2):
    return np.random.uniform(val1, val2)

def rand_scale(val):
    val = np.random.uniform(1, val)
    if np.random.random() < 0.5:
        val = 1 / val
    return val


"""
    Follow darknet format:darknet/src/http_stream.cpp
    
         hsv[1] *= dsat; 
         hsv[2] *= dexp; 
         hsv[0] += 179 * dhue; 

"""
def iaa_hsv_aug(hue=0, saturation=1, exposure=1):
    dhue = rand_uniform(-hue, hue) * 179
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)

    # h_l, h_t = 179 * -hue, 179 * hue
    # s_l, s_t = 1 / saturation if saturation else 0, saturation
    # v_l, v_t = 1 / exposure if exposure else 0, exposure
    
    return iaa.Sequential([iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                           iaa.WithChannels([0], iaa.Add((dhue, dhue))),
                           iaa.WithChannels([1], iaa.Multiply((dsat, dsat))),
                           iaa.WithChannels([2], iaa.Multiply((dexp, dexp))),
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
    return iaa.Sequential([iaa.CropAndPad(None, ((-jitter,jitter),(-jitter,jitter), (-jitter,jitter), (-jitter,jitter)), keep_size=False, pad_cval=128)])

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

class IaaLetterbox(Augmenter):
    def __init__(self, dim, pad_val=128, interpolation="cubic", name=None, deterministic=False, random_state=None):
        super().__init__(name=name, deterministic=deterministic, random_state=random_state)
        
        # dim = (width, height)
        self.dim = dim
        self.pad_cval = pad_val  
        self.interpolation = interpolation

    def _augment_images(self, images, random_state, parents, hooks):
        result = []
        nb_images = len(images)
        height, width = self.dim[1], self.dim[0]
        pad_cval = self.pad_cval
        for i in sm.xrange(nb_images):
            image = images[i]
            # Calculate letterbox parameters
            resize_w, resize_h, x_pad, y_pad = self._compute_height_width_pad(image.shape, height, width)
            # Resize
            image_rs = ia.imresize_single_image(image, (resize_h, resize_w), interpolation=self.interpolation)
            # Add paddings
            pad_left, pad_right = x_pad, width - resize_w - x_pad
            pad_top, pad_bottom = y_pad, height - resize_h - y_pad
            if image_rs.ndim == 2:
                pad_vals = ((pad_top, pad_bottom), (pad_left, pad_right))
            else:
                pad_vals = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
            image_cr_pa = np.pad(image_rs, pad_vals, mode="constant", constant_values=pad_cval)   
            result.append(image_cr_pa)

        if not isinstance(images, list):
            all_same_size = (len(set([image.shape for image in result])) == 1)
            if all_same_size:
                result = np.array(result, dtype=np.uint8)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        nb_images = len(keypoints_on_images)
        height, width = self.dim[1], self.dim[0]
        for i in sm.xrange(nb_images):
            keypoints_on_image = keypoints_on_images[i]
            # Calculate letterbox parameters
            resize_w, resize_h, x_pad, y_pad = self._compute_height_width_pad(keypoints_on_image.shape, height, width)
            # Resize
            new_shape = (resize_h, resize_w) + keypoints_on_image.shape[2:]
            keypoints_on_image_rs = keypoints_on_image.on(new_shape)
            # Add paddings
            pad_left, pad_right = x_pad, width - resize_w - x_pad
            pad_top, pad_bottom = y_pad, height - resize_h - y_pad
            shifted = keypoints_on_image_rs.shift(x=x_pad, y=y_pad)
            shifted.shape = (height + pad_left + pad_right,
                             width + pad_top + pad_bottom) + shifted.shape[2:]
            result.append(shifted)
        return result

    @classmethod
    def _compute_height_width_pad(self, image_shape, new_h, new_w):
        img_h, img_w = image_shape[0:2]
        h, w = new_h, new_w
        
        ratio = min(new_w / img_w, new_h / img_h)
        resize_w, resize_h = int(img_w * ratio), int(img_h * ratio)
        x_pad, y_pad = (new_w - resize_w) // 2, (new_h - resize_h) // 2
        
        return resize_w, resize_h, x_pad, y_pad
    
    def get_parameters(self):
        return [self.width, self.height, self.interpolation]

def iaa_run_seq(seq, img, bbs=None):
    aug_bbs = None
    aug_img = seq.augment_images([img])[0]
    if bbs:
        aug_bbs = seq.augment_bounding_boxes([bbs])[0]
        aug_bbs = bbs_remove_cut_out(aug_bbs, 0.1)
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

def letterbox_transforms(org_w, org_h, new_w, new_h):
    ratio = min(new_w / org_w, new_h / org_h)
    resize_w, resize_h = int(org_w * ratio), int(org_h * ratio)
    x_off, y_off = (new_w - resize_w) // 2, (new_h - resize_h) // 2
    return resize_w, resize_h, x_off, y_off, ratio

def label_np_to_bbs(label, shape, bbs_idx=np.arange(0,4)):
    # Filter bounding boxes that doesn't satisfy x2 > x1 and y2 > y1 
    bbs = ((b[bbs_idx[0]], b[bbs_idx[1]], b[bbs_idx[2]], b[bbs_idx[3]], b) for b in label if b[bbs_idx[2]] > b[bbs_idx[0]] and b[bbs_idx[3]] > b[bbs_idx[1]])
    bbs = [ia.BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3], label=b[4]) for b in bbs]
    bbs = ia.BoundingBoxesOnImage(bbs, shape=shape)
    return bbs

def label_bbs_to_np(bbs, bbs_idx=np.arange(0,4)):
    if len(bbs.bounding_boxes) == 0:
        return np.array([])

    row = len(bbs.bounding_boxes)
    col = bbs.bounding_boxes[0].label.shape[0]
    label = np.zeros((row, col))
    for i, b in enumerate(bbs.bounding_boxes):
        label_row = b.label
        label_row[..., bbs_idx] = [b.x1, b.y1, b.x2, b.y2]
        label[i] = label_row
    return label



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





