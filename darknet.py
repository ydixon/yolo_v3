import numpy as np
import pandas as pd
import os
import os.path as osp
import sys
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import Tensor
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

import torchvision
from torchvision import transforms, datasets, models

from yololayer import YoloLayer
from utils import postprocessing



class conv_bn_relu(nn.Module):
    def __init__(self, nin, nout, ks, s=1, pad='SAME', padding=0, bn=True, act="leakyRelu"):
        super().__init__()
        
        self.bn = bn
        self.act = act
                
        if pad == 'SAME':
            padding = (ks - 1) // 2
            
        self.conv = nn.Conv2d(nin, nout, ks, s, padding, bias=not bn)
        if bn == True:
            self.bn = nn.BatchNorm2d(nout)
        if act == "leakyRelu":
            self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
class res_layer(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.conv1 = conv_bn_relu(nin, nin//2, ks=1)
        self.conv2 = conv_bn_relu(nin//2, nin, ks=3)
        
    def forward(self, x):
        return x + self.conv2(self.conv1(x))

def map2cfgDict(mlist):
    idx = 0
    mdict = OrderedDict()
    for i,m in enumerate(mlist):
        if isinstance(m, res_layer):
            mdict[idx] = None
            mdict[idx+1] = None
            idx += 2
        mdict[idx] = i
        idx += 1
    return mdict


def make_res_stack(nin, num_blk):
    return nn.ModuleList([conv_bn_relu(nin, nin*2, 3, s=2)] \
           + [res_layer(nin*2) for n in range(num_blk)])

class Darknet(nn.Module):
    def __init__(self, blkList, nout=32):
        super().__init__()
        self.mlist = nn.ModuleList()
        self.mlist += [conv_bn_relu(3, nout, 3)]
        for i,nb in enumerate(blkList):
            self.mlist += make_res_stack(nout*(2**i), nb)
            
        self.map2yolocfg = map2cfgDict(self.mlist)
        self.cachedOutDict = dict()
        
    def forward(self,x):
        for i,m in enumerate(self.mlist):
            x = m(x)
            if i in self.cachedOutDict:
                self.cachedOutDict[i] = x
        return x
    
    #mode - normal  -- direct index to mlist
    #     - yolocfg -- index follow the sequences of the cfg file from https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
    def addCachedOut(self, idx, mode="yolocfg"):
        if mode == "yolocfg":
            idxs = self.map2yolocfg[idx]
        self.cachedOutDict[idxs] = None
        
    def getCachedOut(self, idx, mode="yolocfg"):
        if mode == "yolocfg":
            idxs = self.map2yolocfg[idx]
        return self.cachedOutDict[idxs]
        
    def loadWeight(self, weights_path):
        wm = WeightManager(self)
        wm.loadWeight(weights_path)


class PreDetectionConvGroup(nn.Module):
    def __init__(self, nin, nout, num_conv=3, numClass=80):
        super().__init__()
        self.mlist = nn.ModuleList()
        
        for i in range(num_conv):
            self.mlist += [conv_bn_relu(nin, nout, ks=1)]
            self.mlist += [conv_bn_relu(nout, nout*2, ks=3)]
            if i == 0:
                nin = nout*2
                
        self.mlist += [nn.Conv2d(nin, (numClass+5)*3, 1)]
        self.map2yolocfg = map2cfgDict(self.mlist)
        self.cachedOutDict = dict()
        
    def forward(self,x):
        for i,m in enumerate(self.mlist):
            x = m(x)
            if i in self.cachedOutDict:
                self.cachedOutDict[i] = x
        return x
    
    #mode - normal  -- direct index to mlist 
    #     - yolocfg -- index follow the sequences of the cfg file from https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
    def addCachedOut(self, idx, mode="yolocfg"):
        if mode == "yolocfg":
            idx = self.getIdxFromYoloIdx(idx)
        elif idx < 0:
            idx = len(self.mlist) - idx
        
        self.cachedOutDict[idx] = None
        
    def getCachedOut(self, idx, mode="yolocfg"):
        if mode == "yolocfg":
            idx = self.getIdxFromYoloIdx(idx)
        elif idx < 0:
            idx = len(self.mlist) - idx
        return self.cachedOutDict[idx]
    
    def getIdxFromYoloIdx(self,idx):
        if idx < 0:
            return len(self.map2yolocfg) + idx
        else:
            return self.map2yolocfg[idx]


class UpsampleGroup(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.conv = conv_bn_relu(nin, nin//2, ks=1)
        # self.up = nn.Upsample(scale_factor=2, mode="nearest")
        
    def forward(self, route_head, route_tail):
        out = self.conv(route_head)
        out = nn.functional.interpolate(out, scale_factor=2, mode="nearest")
        return torch.cat((out, route_tail), 1)




class YoloNet(nn.Module):
    def __init__(self, img_dim, anchors = [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326], numClass=80):
        super().__init__()
        nin = 32
        self.numClass = numClass
        self.img_dim = img_dim
        self.stat_keys = ['loss', 'loss_x', 'loss_y', 'loss_w', 'loss_h', 'loss_conf', 'loss_cls',
                          'nCorrect', 'nGT', 'recall']
        
        anchors = [(anchors[i], anchors[i+1]) for i in range(0,len(anchors),2)]
        # anchors = [anchors[i:i+3] for i in range(0, len(anchors), 3)][::-1]
                
        self.feature = Darknet([1,2,8,8,4])
        self.feature.addCachedOut(61)
        self.feature.addCachedOut(36)
        
        self.pre_det1 = PreDetectionConvGroup(1024, 512, numClass=self.numClass)
        self.yolo1 = YoloLayer(anchors, [6, 7, 8], img_dim, self.numClass)
        self.pre_det1.addCachedOut(-3) #Fetch output from 4th layer backward including yolo layer
        
        self.up1 = UpsampleGroup(512)
        self.pre_det2 = PreDetectionConvGroup(768, 256, numClass=self.numClass)
        self.yolo2 = YoloLayer(anchors, [3, 4, 5], img_dim, self.numClass)
        self.pre_det2.addCachedOut(-3)
        
        self.up2 = UpsampleGroup(256)
        self.pre_det3 = PreDetectionConvGroup(384, 128, numClass=self.numClass)
        self.yolo3 = YoloLayer(anchors, [0, 1, 2], img_dim, self.numClass)
        
        
        
    def forward(self, x, target=None):
        img_dim = (x.shape[3], x.shape[2])
        #Extract features
        out = self.feature(x)
                
        #Detection layer 1
        out = self.pre_det1(out)
        det1 = self.yolo1(out, img_dim, target)
        
        #Upsample 1
        r_head1 = self.pre_det1.getCachedOut(-3)
        r_tail1 = self.feature.getCachedOut(61)
        out = self.up1(r_head1,r_tail1)
                
        #Detection layer 2
        out = self.pre_det2(out)
        det2 = self.yolo2(out, img_dim, target)
        
        #Upsample 2
        r_head2 = self.pre_det2.getCachedOut(-3)
        r_tail2 = self.feature.getCachedOut(36)
        out = self.up2(r_head2,r_tail2)
                
        #Detection layer 3
        out = self.pre_det3(out)
        det3 = self.yolo3(out, img_dim, target)
        
        if target is not None:
            loss, *out = [sum(det) for det in zip(det1, det2, det3)]
            self.stats = dict(zip(self.stat_keys, out))
            self.stats['recall'] = self.stats['nCorrect'] / self.stats['nGT'] if self.stats['nGT'] else 0
            return loss
        else:
            return det1, det2, det3
    
    # Format : pytorch / darknet
    def saveWeight(self, weights_path, format='pytorch'):
        if format == 'pytorch':
            torch.save(self.state_dict(), weights_path)
        elif format == 'darknet':
            raise NotImplementedError
    
    def loadWeight(self, weights_path, format='pytorch'):
        if format == 'pytorch':
            weights = torch.load(weights_path, map_location=lambda storage, loc: storage)
            self.load_state_dict(weights)
        elif format == 'darknet':
            wm = WeightManager(self)
            wm.loadWeight(weights_path)


class WeightManager:
    def __init__(self, model):
        super().__init__()
        self.conv_list = self.find_conv_layers(model)

    def loadWeight(self, weight_path):
        ptr = 0
        weights = self.read_file(weight_path)
        #print(len(weights))
        for m in self.conv_list:
            if type(m) == conv_bn_relu:
                ptr = self.load_conv_bn_relu(m, weights, ptr)
            elif type(m) == nn.Conv2d:
                ptr = self.load_conv2D(m, weights, ptr)
        return ptr
                
    def read_file(self, file):
        with open(file, "rb") as fp:
            header = np.fromfile(fp, dtype = np.int32, count = 5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]
            weights = np.fromfile(fp, dtype = np.float32)
        return weights
    
    def copy_weight_to_model_parameters(self, param, weights, ptr):
        num_el = param.numel()
        param.data.copy_(torch.from_numpy(weights[ptr:ptr + num_el])
                             .view_as(param.data))
        return ptr + num_el
    
    def load_conv_bn_relu(self, m, weights, ptr):
        ptr = self.copy_weight_to_model_parameters(m.bn.bias, weights, ptr)
        ptr = self.copy_weight_to_model_parameters(m.bn.weight, weights, ptr)
        ptr = self.copy_weight_to_model_parameters(m.bn.running_mean, weights, ptr)
        ptr = self.copy_weight_to_model_parameters(m.bn.running_var, weights, ptr)
        ptr = self.copy_weight_to_model_parameters(m.conv.weight, weights, ptr)
        return ptr
        
    def load_conv2D(self, m, weights, ptr):
        ptr = self.copy_weight_to_model_parameters(m.bias, weights, ptr)
        ptr = self.copy_weight_to_model_parameters(m.weight, weights, ptr)
        return ptr
        
    def find_conv_layers(self, mod):
        module_list = []
        for m in mod.children():
            if type(m) == conv_bn_relu:
                module_list += [m]
            elif type(m) == nn.Conv2d:
                module_list += [m]
            elif isinstance(m, (nn.ModuleList, nn.Module)):
                module_list += self.find_conv_layers(m)
            elif type(m) == res_layer:
                module_list += self.find_conv_layers(m)
        return module_list