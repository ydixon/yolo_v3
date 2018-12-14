
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

# Move to utils / transfroms / boundingbox
def letterbox_reverse_exp(labels, params, bbs_idx=np.array([0,1,2,3])):
    #print('labels:{}'.format(labels.shape))
    for i, (l, p) in enumerate(zip(labels, params)):
        org_img, padded_dim = tuple(p[0:2]), tuple(p[2:4])
        x_pad, y_pad = p[4], p[5]
        new_l = transforms.letterbox_reverter(l, org_img, padded_dim, x_pad, y_pad, bbs_idx)
        #print(new_l.shape)
        labels[i] =  new_l
    return labels

def prep_labels_for_plt(labels_list, dim, lb_params_list, reverse_letterbox):
    if reverse_letterbox:
        labels_list = BoundingBoxConverter.convert(labels_list,
                                       CoordinateType.Relative, FormatType.cxcywh,
                                       CoordinateType.Absolute, FormatType.x1y1x2y2,
                                       np.array([1,2,3,4]), dim)
        labels_list = letterbox_reverse_exp(labels_list, lb_params_list, np.array([1,2,3,4]))

        labels_list = BoundingBoxConverter.convert(labels_list.numpy(),
                                                   CoordinateType.Absolute, FormatType.x1y1x2y2,
                                                   CoordinateType.Absolute, FormatType.xywh,
                                                   np.array([1,2,3,4]), dim)
    else:
        labels_list = BoundingBoxConverter.convert(labels_list.numpy(),
                                               CoordinateType.Relative, FormatType.cxcywh,
                                               CoordinateType.Absolute, FormatType.xywh,
                                               np.array([1,2,3,4]), dim)

    return labels_list

def prep_img_for_plt(img_list):
    if isinstance(img_list, collections.abc.Sequence):
        img_list = [img.permute(1,2,0).numpy() for img in img_list] 
    else:
        img_list = img_list.permute(0,2,3,1).numpy()
    return img_list
      
def prep_predictions_for_plt(preds_list, dim, lb_params, reverse_letterbox):
    preds_list = Tensor([fill_label_np_tensor(p.numpy(), 50, 7) for p in preds_list])
    if reverse_letterbox:
        preds_list = letterbox_reverse_exp(preds_list, lb_params)
    preds_list = preds_list[..., [6, 0, 1, 2, 3,]]
    preds_list = BoundingBoxConverter.convert(preds_list, CoordinateType.Absolute, FormatType.x1y1x2y2,
                                                      CoordinateType.Absolute, FormatType.xywh,
                                                      np.array([1,2,3,4]), dim)
    preds_list = preds_list.numpy()
    return preds_list

def prep_img_for_opencv(img_list):
    img_list = img_list.permute(0,2,3,1) * 255
    img_list = img_list.numpy().astype(np.uint8)
    return img_list


def predict(data, net, num_classes=80, reverse_letterbox=True):
    img_list = []
    preds_list = []
    lb_params_list = torch.FloatTensor()
    with torch.no_grad(): 
        for sample in data:
            imgs, org_imgs = sample['img'].cuda(), sample['org_img']
            labels, lb_params = sample['label'], sample['lb_reverter']
            dim = (imgs.shape[3], imgs.shape[2])

            lb_params_list = torch.cat((lb_params_list, lb_params), 0)
            # Copy original image instead of the letterboxed image
            if reverse_letterbox:                      
                img_list += [img.cpu() for img in org_imgs]
            else:
                img_list += [img for img in imgs.cpu()]
            
            # Pass images to the network
            det1, det2, det3 = net(imgs, None)
            predictions = postprocessing(torch.cat((det1,det2,det3), 1), num_classes, obj_conf_thr=0.5, nms_thr=0.4)
            preds_list += predictions

            #print('Batch:{} {}'.format(data.current_batch, str(sample['img'].shape)))
        img_list = prep_img_for_plt(img_list)
        preds_list = prep_predictions_for_plt(preds_list, dim, lb_params_list, reverse_letterbox)
    return img_list, preds_list


def show_detections(data, net, classes_names, cols=2, reverse_letterbox=True):
    img_list, preds_list = predict(data, net, num_classes=len(classes_names), reverse_letterbox=reverse_letterbox)
    show_img_grid(img_list, cols=cols, classes=classes_names, labels_list=preds_list)


def predict_multiple(data, nets, num_classes=80, reverse_letterbox=True):
    img_list = []
    preds_list = [[] for i in range(len(nets))]
    labels_list = torch.FloatTensor()
    lb_params_list = torch.FloatTensor()
    with torch.no_grad(): 
        for sample in data:
            imgs, org_imgs = sample['img'].cuda(), sample['org_img']
            labels, lb_params = sample['label'], sample['lb_reverter']
            dim = (imgs.shape[3], imgs.shape[2])

            labels_list = torch.cat((labels_list, labels), 0)
            lb_params_list = torch.cat((lb_params_list, lb_params), 0)
            if reverse_letterbox:                      
                img_list += [img.cpu() for img in org_imgs]
            else:
                img_list += [img for img in imgs.cpu()]

            # Pass images to the  multiple networks
            for i, net in enumerate(nets):
                det1, det2, det3 = net(imgs, None)
                predictions = postprocessing(torch.cat((det1,det2,det3), 1), num_classes, obj_conf_thr=0.5, nms_thr=0.4)
                preds_list[i] += predictions
            
        img_list = prep_img_for_plt(img_list)
        labels_list = prep_labels_for_plt(labels_list, dim, lb_params_list, reverse_letterbox)
        preds_list = [prep_predictions_for_plt(sub_list, dim, lb_params_list, reverse_letterbox)
                      for sub_list in preds_list]

    return img_list, preds_list, labels_list

def show_detections_comparisons(model_id, nets, data, classes_names, reverse_letterbox=False, cols=0):
    img_list, preds_list, labels_list = predict_multiple(data, nets, num_classes=len(classes_names), reverse_letterbox=reverse_letterbox)
    
    show_img_list = [x for x in itertools.chain.from_iterable(itertools.zip_longest(img_list, img_list, img_list))]
    show_preds_list = [x for x in itertools.chain.from_iterable(itertools.zip_longest(labels_list, *preds_list))]
    
    cols = len(nets) + 1 if not cols else cols
    show_img_grid(show_img_list, cols=cols, classes=classes_names, labels_list=show_preds_list,
                  col_title_dict= { 'title': ['Labels', 'Darknet', 'Test Model'],
                                    'pad': 20,
                                    'fontsize': 40,
                                    'fontweight': 20,
                                    })
