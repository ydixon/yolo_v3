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
import imgaug as ia


def train(dataloader, net, num_epoch,
          lr, backbone_lr, wd=0, momentum=0, freeze_backbone=False,
          lr_step_decay=0, lr_step_gamma=0,
          net_bs=64, net_subdivisions=4,
          model_id='test', start_epoch=0, weight_dir=None, checkpoint_interval=1,
          resume_checkpoint=None, use_gpu=True):
    
    optimizer = get_optimizer(net, lr, backbone_lr, wd, momentum, freeze_backbone)
    if not (lr_step_decay == 0 and lr_step_gamma == 0):
        scheduler = lr_scheduler.StepLR(optimizer, lr_step_decay, lr_step_gamma)
    else:
        scheduler = None                     
    
    if resume_checkpoint is not None:
        net.load_state_dict(resume_checkpoint['net'])
        optimizer = load_optimizer(optimizer, resume_checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(resume_checkpoint['scheduler'])
        start_epoch = resume_checkpoint['epoch'] + 1
    
    train_impl(num_epoch, dataloader, net, optimizer, scheduler,
               net_bs, net_subdivisions,
               model_id, start_epoch, weight_dir, checkpoint_interval,
               use_gpu)

def train_impl(num_epoch, dataloader, net, optimizer, scheduler,
               net_bs=64, net_subdivisions=4,
               model_id='test', start_epoch=0, weight_dir=None, checkpoint_interval=1,
               use_gpu=True):
    mini_batch_size = net_bs / net_subdivisions

    batch_datasize = 0
    batch_stats = []
    optimizer.zero_grad()
    recorder = Recorder()
    
    print_stats_header()
    for epoch in range(start_epoch, num_epoch):
        for phase in ['train']: #for phase in ['train', 'valid']:
            if phase == 'train':
                net.train(True)
            else:
                net.train(False)
            pbar = create_batch_progressbar(dataloader[phase])
            for batch_idx, sample in enumerate(pbar):
                inp, labels = sample['img'], sample['label']
                if use_gpu:
                    inp, labels = inp.cuda(), labels.cuda()
                
                              
                loss = net(inp, labels)
                loss = loss / net_subdivisions
                loss.backward()
                
                batch_stats.append(net.stats)
                batch_datasize += inp.shape[0] 

                nn.utils.clip_grad_norm_(net.parameters(), 1000)

                if ((batch_idx+1) % net_subdivisions == 0) or (batch_idx == (len(pbar) - 1)):
                    optimizer.step()
                    optimizer.zero_grad()

                    stats = {k: sum([d[k] for d in batch_stats]) / net_subdivisions for k in net.stat_keys}
                    recorder.on_batch_end({k: stats[k] for k in recorder.acc_keys if k in stats},
                                          batch_datasize)
                    
                    update_batch_progressbar(pbar, epoch, recorder)
                    batch_datasize = 0
                    batch_stats = []
                # Temporary hack, need to call seq.deterministic on each batch if done properly
                ia.seed(np.random.randint(0, 2**16))

            print_stats(epoch, recorder)
            if ((epoch+1) % 20) == 0:
                 print_stats_header()
            
            recorder.on_epoch_end()
            
            if scheduler is not None:
                scheduler.step()
        
            if phase == 'train' and ((epoch+1) % checkpoint_interval == 0):
                save_checkpoint(epoch, net, optimizer, scheduler, model_id, weight_dir)
    
    optimizer.zero_grad()


def load_optimizer(optimizer, state_dict):
    if len(optimizer.state_dict()) != len(state_dict):
        optimizer.load_state_dict(state_dict)
    else:
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

def create_batch_progressbar(dataloader):
    return tqdm(dataloader, file=sys.stdout, leave=False)

def update_batch_progressbar(progess_bar, epoch, recorder):
    progess_bar.set_description_str('{:5d}  {:9.7f} {:9.7f} {:9.7f} {:9.7f} {:9.7f} {:9.7f} {:10.7f} {:9.7f}'
                         .format(epoch, *recorder.current_stats.values()))

def print_stats(epoch, recorder):
    print('{:5d}  {:9.7f} {:9.7f} {:9.7f} {:9.7f} {:9.7f} {:9.7f} {:10.7f} {:9.7f}'
                         .format(epoch, *recorder.current_stats.values()))

def print_stats_header():
    print('{:5s}  {:>9s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s} {:>10s} {:>9s}'
          .format('Epoch', 'loss_x', 'loss_y', 'loss_w', 'loss_h', 'loss_conf', 'loss_cls', 'loss_total','recall'))

# Stats recorder

class Recorder:
    def __init__(self):
        self.loss_keys = ['loss_x', 'loss_y', 'loss_w', 'loss_h', 'loss_conf', 'loss_cls', 'loss']
        self.metrics_keys = ['nCorrect', 'nGT']
        self.acc_keys = self.loss_keys + self.metrics_keys
        
        self.eval_keys = ['recall']
        self.current_keys = self.loss_keys + self.eval_keys

        self.acc_stats = OrderedDict([(k, 0.0) for k in self.acc_keys])
        self.eval_stats = OrderedDict([(k, 0.0) for k in self.eval_keys])
        self.current_stats = OrderedDict([(k, 0.0) for k in self.current_keys])
        self.acc_datasize = 0
        
    def on_batch_end(self, batch_stats, batch_datasize):
        self.acc_stats = {k: self.acc_stats[k] + batch_stats[k] for k in self.acc_keys}
        self.acc_datasize += batch_datasize
        
        self.eval_stats['recall'] = self.acc_stats['nCorrect'] / self.acc_stats['nGT'] if self.acc_stats['nGT'] else 0
        self.current_stats.update({k: self.acc_stats[k] / self.acc_datasize for k in self.loss_keys}) 
        self.current_stats.update(self.eval_stats)
 
    def on_epoch_end(self):
        self.acc_stats = OrderedDict([(k, 0.0) for k in self.acc_keys])
        self.current_stats = OrderedDict([(k, 0.0) for k in self.current_keys])
        self.eval_stats = OrderedDict([(k, 0.0) for k in self.eval_keys])
        self.acc_datasize = 0

# Model Checkpoints

def save_checkpoint(epoch, net, optimizer, scheduler, model_id, weight_dir):    
    checkpoint = { 'epoch' : epoch,
                   'net' : net.state_dict(),
                   'optimizer' : optimizer.state_dict(),
                   'scheduler' : scheduler.state_dict() if scheduler else None,
    }
    model_dir = osp.join(weight_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)
    file_name = 'yolov3_%s_checkpoint_%.4d%s' % (model_id, epoch, '.pth.tar')
    torch.save(checkpoint, osp.join(model_dir, file_name))
    
def load_checkpoint(full_path):
    checkpoint = torch.load(full_path,  map_location=lambda storage, loc: storage)
    return checkpoint

def remove_checkpoints(model_id, weight_dir):
    for f in glob.glob('%s/%s/yolov3_%s_checkpoint_%s' % (weight_dir, model_id, model_id, '*.pth.tar')):
        print(f)
        os.remove(f)
        
def get_latest_checkpoint(model_id, weight_dir):
    files_list = [f for f in glob.glob(osp.join(weight_dir, model_id, '*.*.tar'))]
    if files_list is None:
        return None, 0
    
    latest_epoch = -1
    latest_i = -1
    for i, f in enumerate(files_list):
        pattern = 'yolov3_(.+?)_checkpoint_(.+?)\.'
        m = re.search(pattern, f)
        f_id = m.group(1)
        epoch = int(m.group(2))
        if f_id == model_id and (epoch >= latest_epoch):
            latest_epoch = epoch
            latest_i = i
            
    if latest_i < 0:
        return None, 0
    else:
        return files_list[latest_i], latest_epoch       