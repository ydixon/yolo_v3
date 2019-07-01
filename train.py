import os
import os.path as osp
import sys
import glob
import re
from tqdm import tqdm
from collections import OrderedDict
from collections import Counter

import torch
import torch.nn as nn

import numpy as np

from darknet import YoloNet
from draw import cv2_drawTextWithBkgd, get_color_pallete
from utils import ewma_online
import imgaug as ia


def train(data, net, optimizer, recorder, 
          model_id='test', weight_dir=None, 
          checkpoint=None, checkpoint_interval=1, use_gpu=True):
    if checkpoint is not None:
        data.load_state_dict(checkpoint['data'])
        net.load_state_dict(checkpoint['net'])
        optimizer = load_optimizer(optimizer, checkpoint['optimizer'])
        recorder.load_state_dict(checkpoint['recorder'])
    
    train_impl(data, net, optimizer, recorder, None,
               model_id, weight_dir, checkpoint_interval,
               use_gpu)

def train_impl(data, net, optimizer, recorder, scheduler,
               model_id='test', weight_dir=None, checkpoint_interval=1,
               use_gpu=True, debug_log=False):
    batch_datasize = 0
    batch_stats = []
    optimizer.zero_grad()
    
    pbar = None
    print_stats_header()

    # data will generate mini-batches of sample
    for sample in data:
        # batch - mini-batch index, net_batch - net batch index, epoch - epoch index
        batch, net_batch, epoch = data.get_batch(), data.get_net_batch(), data.get_epoch()
        dim = sample['img'].shape

        if pbar is None or data.isStartOfEpoch():
            pbar = create_batch_progressbar(data.get_epoch_batch(), data.get_epoch_num_batches())
            update_batch_progressbar(pbar, recorder, epoch, net_batch, batch, dim)
        pbar.update()

        inp, labels = sample['img'], sample['label']
        if use_gpu:
            inp, labels = inp.cuda(), labels
        
        # Accumulate gradients for each mini-batch    
        loss = net(inp, labels)
        # loss = loss / data.net_subdivisions
        loss.backward()
        
        batch_stats.append(net.stats)
        batch_datasize += inp.shape[0] 

        nn.utils.clip_grad_norm_(net.parameters(), 1000)

        # Backpropogate for each net batch
        if ((batch+1) % data.net_subdivisions == 0):
            optimizer.step()
            optimizer.zero_grad()

            stats = {k: sum([d[k] for d in batch_stats]) / data.net_subdivisions for k in net.stat_keys}
            recorder.on_batch_end({k: stats[k] for k in recorder.ewma_keys if k in stats},
                                  batch_datasize)
            update_batch_progressbar(pbar, recorder, epoch, net_batch, batch, dim)

            if debug_log:
                print_stats(net_batch, epoch, recorder)
                print('Net Batch:{} Batch:{} Dim:{}'.format(net_batch, batch, str(dim)))

            batch_datasize = 0
            batch_stats = []

            if ((batch+1) / data.net_subdivisions) % checkpoint_interval == 0:
                print_stats(net_batch, epoch, recorder)
                save_checkpoint(data, net, optimizer, recorder, scheduler, model_id, weight_dir)
                if ((net_batch+1)/checkpoint_interval % 20) == 0:
                    print_stats_header()

        if data.isEndOfEpoch():     
            if pbar is not None:
                pbar.close()
            recorder.on_epoch_end()

    optimizer.zero_grad()

    if pbar is not None:
        pbar.close()

    print("\n[Finish] Net Batch:{}, current_batch:{}".format(data.get_net_batch(), data.get_batch()))

def load_optimizer(optimizer, state_dict):
    if len(optimizer.param_groups) == len(state_dict['param_groups']):
        # Freeze backbone
        if len(optimizer.param_groups) == 1:
            state_dict['param_groups'][0].update({p: optimizer.param_groups[0][p] for p in ['lr', 'weight_decay', 'momentum']})
        # Detection layers and backbone layers
        elif len(optimizer.param_groups) == 2:
            state_dict['param_groups'][0].update({p: optimizer.param_groups[0][p] for p in ['lr', 'weight_decay', 'momentum']})
            state_dict['param_groups'][1].update({p: optimizer.param_groups[1][p] for p in ['lr', 'weight_decay', 'momentum']})
        optimizer.load_state_dict(state_dict)
    else:
        print("Optimizer not loaded")
    return optimizer

def get_optimizer(net, lr, backbone_lr, wd, momentum, freeze_backbone):
    feature_params = map(id, net.feature.parameters())
    detection_params = filter(lambda p : id(p) not in feature_params, net.parameters())
    if freeze_backbone:
        params = [
                    {"params": detection_params, "lr": lr},
                ]

        for p in net.feature.parameters():
            p.requires_grad = False
    else:
        params = [
                {"params": detection_params, "lr": lr},
                {"params": net.feature.parameters(), "lr": backbone_lr}
            ]
    optimizer = torch.optim.SGD(params, lr, weight_decay=wd, momentum=momentum)

    return optimizer

# Display stats and progress bar

def get_stats_string(net_batch, epoch, recorder):
    return '{:>9d} {:>5d} {:0<9.7g} {:0<9.7g} {:0<9.7g} {:0<9.7g} {:0<9.7g} {:0<9.7g} {:0<10.7g} {:0<9.7g}' \
                         .format(net_batch, epoch, *recorder.current_stats.values())

def create_batch_progressbar(start, end):
    return tqdm(file=sys.stdout, leave=False, initial=start,  total=end)

def update_batch_progressbar(progess_bar, recorder, epoch, net_batch, batch, dim):
    progess_bar.set_description_str(get_stats_string(net_batch, epoch, recorder))
    progess_bar.set_postfix_str('Net Batch:{} Batch:{} Dim:{}'.format(net_batch, batch, str(dim)))

def print_stats(net_batch, epoch, recorder, use_tqdm=True):
    out = get_stats_string(net_batch, epoch, recorder)
    if use_tqdm:
        tqdm.write(out)
    else:
        print(out)

def print_stats_header(use_tqdm=True):
    out = "{:>9s} {:>5s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s} {:>10s} {:>9s}" \
          .format('Net_Batch', 'Epoch', 'loss_x', 'loss_y', 'loss_w', 'loss_h', 'loss_conf', 'loss_cls', 'loss_total','recall')
    if use_tqdm:
        tqdm.write(out)
    else:
        print(out)

def print_save_msg(net_batch, batch, use_tqdm=True):
    tqdm.write("Saving at Net Batch:{}, current_batch:{}" \
                .format(net_batch, batch))


# Stats recorder
class Recorder:
    def __init__(self):
        self.loss_keys = ['loss_x', 'loss_y', 'loss_w', 'loss_h', 'loss_conf', 'loss_cls', 'loss']
        self.metrics_keys = ['nCorrect', 'nGT']
        self.acc_keys = self.loss_keys + self.metrics_keys
        
        self.eval_keys = ['recall']
        self.current_keys = self.loss_keys + self.eval_keys
        self.ewma_keys = self.loss_keys + self.eval_keys
        
        self.ewma_stats = OrderedDict([(k, 0.0) for k in self.ewma_keys])
        self.current_stats = OrderedDict([(k, 0.0) for k in self.current_keys])
        # Not used temporarily
        # self.acc_stats = OrderedDict([(k, 0.0) for k in self.acc_keys])
        # self.eval_stats = OrderedDict([(k, 0.0) for k in self.eval_keys])
        # self.acc_datasize = 0

    def state_dict(self):
        state_dict = { 'ewma_stats': self.ewma_stats }
        return state_dict

    def load_state_dict(self, state_dict):
        self.ewma_stats = state_dict['ewma_stats']
        self.current_stats.update({k: self.ewma_stats[k] for k in self.ewma_keys}) 
        
    def on_batch_end(self, batch_stats, batch_datasize):
        # self.ewma_stats = OrderedDict({k: ewma_online(batch_stats[k], self.ewma_stats[k], 10)
        #                               if self.ewma_stats[k] != 0 else batch_stats[k]
        #                               for k in self.ewma_keys})
        self.ewma_stats = OrderedDict({k: batch_stats[k] for k in self.ewma_keys})
        self.current_stats.update({k: self.ewma_stats[k] for k in self.ewma_keys}) 

    def on_epoch_end(self):
        # Supposely used to clear stats, not used at the moment
        pass
        

# Model Checkpoints

def save_checkpoint(data, net, optimizer, recorder, scheduler, model_id, weight_dir):    
    checkpoint = { 'data' : data.get_state_dict(),
                   'net' : net.state_dict(),
                   'optimizer' : optimizer.state_dict(),
                   'recorder' : recorder.state_dict(),
                   'scheduler' : scheduler.state_dict() if scheduler else None,
    }
    model_dir = osp.join(weight_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)
    file_name = 'yolov3_%s_checkpoint_%.6d%s' % (model_id, data.get_net_batch(), '.pth.tar')
    torch.save(checkpoint, osp.join(model_dir, file_name))
    
def load_checkpoint(full_path):
    checkpoint = torch.load(full_path,  map_location=lambda storage, loc: storage)
    return checkpoint

def get_checkpoint_list(model_id, weight_dir):
    files_list = [f for f in glob.glob(osp.join(weight_dir, model_id, '*.*.tar'))]
    return files_list

def remove_checkpoints(model_id, weight_dir, num_remove=20, num_keep=10, remove_all=False, debug=False):
    checkpoint_list = sorted(get_checkpoint_list(model_id, weight_dir))
    if remove_all:
        for f in checkpoint_list:
            print('Deleting {}'.format(f))
            if not debug:
                os.remove(f)
    else:
        remove_items = len(checkpoint_list) - num_keep
        if remove_items >= num_remove:
            for f in checkpoint_list[:remove_items]:
                print('Deleting {}'.format(f))
                if not debug:
                    os.remove(f)
        
def get_latest_checkpoint(model_id, weight_dir):
    files_list = [f for f in glob.glob(osp.join(weight_dir, model_id, '*.*.tar'))]
    if files_list is None:
        return None, 0
    
    latest_iteration = -1
    latest_i = -1
    for i, f in enumerate(files_list):
        pattern = 'yolov3_(.+?)_checkpoint_(.+?)\.'
        m = re.search(pattern, f)
        f_id = m.group(1)
        iteration = int(m.group(2))
        if f_id == model_id and (iteration >= latest_iteration):
            latest_iteration = iteration
            latest_i = i
            
    if latest_i < 0:
        return None, 0
    else:
        return files_list[latest_i], latest_iteration       