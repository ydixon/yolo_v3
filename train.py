import re
import os
import os.path as osp
import glob

import torch
import torch.nn as nn

from darknet import YoloNet

from draw import cv2_drawTextWithBkgd, get_color_pallete





def train(dataloader, net, num_epoch,
          lr, backbone_lr, wd=0, momentum=0,
          lr_step_decay=0, lr_step_gamma=0,
          model_id='test', start_epoch=0, weight_dir=None, checkpoint_interval=1,
          resume_checkpoint=None, use_gpu=True):
    
    optimizer = get_optimizer(net, lr, backbone_lr, wd, momentum)
    if not (lr_step_decay == 0 and lr_step_gamma == 0):
        scheduler = lr_scheduler.StepLR(optimizer, lr_step_decay, lr_step_gamma)
    else:
        scheduler = None
    
    if resume_checkpoint is not None:
        net.load_state_dict(resume_checkpoint['net'])
        optimizer.load_state_dict(resume_checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(resume_checkpoint['scheduler'])
        start_epoch = resume_checkpoint['epoch'] + 1
    
    train_impl(num_epoch, dataloader, net, optimizer, scheduler,
               model_id, start_epoch, weight_dir, checkpoint_interval,
               use_gpu)


def train_impl(num_epoch, dataloader, net, optimizer, scheduler,
               model_id='test', start_epoch=0, weight_dir=None, checkpoint_interval=1,
               use_gpu=True):
    acc_stats = dict.fromkeys(net.stat_keys + ['acc_datasize'], 0)
    
    for epoch in range(start_epoch, num_epoch):
        for phase in ['train']: #for phase in ['train', 'valid']:
            if phase == 'train':
                net.train(True)
            else:
                net.train(False)
            for batch, sample in enumerate(dataloader[phase]):
                inp, labels = sample['letterbox_img'], sample['label']
                if use_gpu:
                    inp, labels = inp.cuda(), labels.cuda()
                
                optimizer.zero_grad()
                
                loss = net(inp, labels)
                loss.backward()
                
                optimizer.step()
                
                datasize = inp.shape[0]
                acc_stats = {k:acc_stats.get(k,0) + net.stats.get(k, 0) for k in set(acc_stats)}
                acc_stats['acc_datasize'] += datasize
                
                print('[Epoch:%d[%d/%d], Batch %d/%d]\n[Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f]' %
                    (epoch, epoch+1, num_epoch, batch, len(dataloader[phase]),
                    acc_stats['loss_x']/acc_stats['acc_datasize'], acc_stats['loss_y']/acc_stats['acc_datasize'],
                    acc_stats['loss_w']/acc_stats['acc_datasize'], acc_stats['loss_h']/acc_stats['acc_datasize'],
                    acc_stats['loss_conf']/acc_stats['acc_datasize'], acc_stats['loss_cls']/acc_stats['acc_datasize'],
                    acc_stats['loss']/acc_stats['acc_datasize'], acc_stats['nCorrect']/acc_stats['nGT']))
            
            if scheduler is not None:
                scheduler.step()
        
            if phase == 'train' and ((epoch+1) % checkpoint_interval == 0):
                save_checkpoint(epoch, net, optimizer, scheduler, model_id, weight_dir)
                
            acc_stats = {k: 0 for k in acc_stats}
            
    optimizer.zero_grad()


def get_optimizer(net, lr, backbone_lr, wd, momentum):
    feature_params = map(id, net.feature.parameters())
    detection_params = filter(lambda p : id(p) not in feature_params, net.parameters())
    params = [
                {"params": detection_params, "lr": lr},
                {"params": net.feature.parameters(), "lr": backbone_lr}
            ]
    
    optimizer = torch.optim.SGD(params, lr, weight_decay=wd, momentum=momentum)
    return optimizer


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