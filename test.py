
import cv2
import PIL
from PIL import ImageDraw, ImageFont
from matplotlib import patches, patheffects
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.ticker as plticker
import torch
import numpy as np
import itertools
import collections

from boundingbox import *
from utils import *
from draw import *
import transforms

def prep_img_for_plt(img_list):
    if isinstance(img_list, collections.abc.Sequence):
        img_list = [img.permute(1,2,0).numpy() for img in img_list] 
    else:
        img_list = img_list.permute(0,2,3,1).cpu().numpy()
        img_list = [img for img in img_list]
    return img_list

def predict(data, net, num_classes=80, is_letterbox=True):
    img_list, preds_list = [], []
    with torch.no_grad(): 
        for sample in data:
            imgs, org_imgs, labels = sample['img'].cuda(), sample['org_img'], sample['label']

            # Pass images to the network
            det1, det2, det3 = net(imgs, None)
            predictions = postprocessing(torch.cat((det1,det2,det3), 1), num_classes, obj_conf_thr=0.5, nms_thr=0.4, is_eval=False, use_nms=True)

            for img, org_img, prediction in zip(imgs, org_imgs, predictions):
                img_w, img_h, org_w, org_h = img.shape[2], img.shape[1], org_img.shape[2], org_img.shape[1]
                if prediction is not None and len(prediction) != 0:
                    bboxes = correct_yolo_boxes(prediction[..., 0:4], org_w, org_h, img_w, img_h, is_letterbox)
                    prediction = torch.cat((prediction[..., 6:7], bboxes), -1)

                preds_list += [prediction.cpu().numpy()]
            img_list += prep_img_for_plt(org_imgs)
    return img_list, preds_list

def show_detections(data, net, classes_names, cols=2, is_letterbox=True):
    img_list, preds_list = predict(data, net, num_classes=len(classes_names), is_letterbox=is_letterbox)
    show_img_grid(img_list, cols=cols, classes=classes_names, labels_list=preds_list)

# Commented code reason: Since we are not display the original image(org_img), there is no need to correct yolo boxes.
#                        Add Option to choose later.
def predict_multiple(data, nets, num_classes=80, is_letterbox=True):
    img_list = []
    preds_list = [[] for i in range(len(nets))]
    labels_list = []
    with torch.no_grad(): 
        for sample in data:
            imgs, org_imgs, labels = sample['img'].cuda(), sample['org_img'], sample['label']

            for i, net in enumerate(nets):
                det1, det2, det3 = net(imgs, None)
                predictions = postprocessing(torch.cat((det1,det2,det3), 1), num_classes, obj_conf_thr=0.5, nms_thr=0.4, is_eval=False, use_nms=True)

                for img, org_img, prediction in zip(imgs, org_imgs, predictions):
                    img_w, img_h, org_w, org_h = img.shape[2], img.shape[1], org_img.shape[2], org_img.shape[1]

                    if prediction is not None and len(prediction) != 0:
                        # bboxes = correct_yolo_boxes(prediction[..., 0:4], org_w, org_h, img_w, img_h, is_letterbox)
                        prediction = torch.cat((prediction[..., 6:7], prediction[..., 0:4]), -1)
                        prediction = BoundingBoxConverter.convert(prediction, CoordinateType.Absolute, FormatType.x1y1x2y2,
                                                      CoordinateType.Absolute, FormatType.xywh,
                                                      np.array([1,2,3,4]), (416,416))
                    preds_list[i] += [prediction.cpu().numpy()]


            img_list += prep_img_for_plt(imgs)
            for img, org_img, label in zip(imgs, org_imgs, labels):
                img_w, img_h, org_w, org_h = img.shape[2], img.shape[1], org_img.shape[2], org_img.shape[1]

                bboxes = label[..., 1:5]
                # bboxes = correct_yolo_boxes(bboxes, org_w, org_h, img_w, img_h, is_letterbox)
                # if is_letterbox:
                #     bboxes = letterbox_reverse(bboxes, org_w, org_h, img_w, img_h)
                # else:
                #     bboxes = rescale_bbox(bboxes, org_w, org_h, img_w, img_h)

                label[..., 1:5] = BoundingBoxConverter.convert(bboxes, 
                                                      CoordinateType.Relative, FormatType.cxcywh,
                                                      CoordinateType.Absolute, FormatType.xywh,
                                                      img_dim=(img_w, img_h))
                labels_list += [label.numpy()]
    return img_list, preds_list, labels_list

def show_detections_comparisons(model_id, nets, data, classes_names, is_letterbox=False, cols=0):
    img_list, preds_list, labels_list = predict_multiple(data, nets, num_classes=len(classes_names), is_letterbox=is_letterbox)
    
    show_img_list = [x for x in itertools.chain.from_iterable(itertools.zip_longest(img_list, img_list, img_list))]
    show_preds_list = [x for x in itertools.chain.from_iterable(itertools.zip_longest(labels_list, *preds_list))]
    
    cols = len(nets) + 1 if not cols else cols
    show_img_grid(show_img_list, cols=cols, classes=classes_names, labels_list=show_preds_list,
                  col_title_dict= { 'title': ['Labels', 'Darknet', 'Test Model'],
                                    'pad': 20,
                                    'fontsize': 40,
                                    'fontweight': 20,
                                    })
