import torch
import torch.nn as nn
from utils import bbox_iou


class YoloLayer(nn.Module):
    def __init__(self, anchors, img_dim, numClass):
        super().__init__()
        self.anchors = anchors
        self.img_dim = img_dim
                
        self.numClass = numClass
        self.bbox_attrib = 5 + numClass
        
        self.lambda_xy = 1
        self.lambda_wh = 1
        self.lambda_conf = 1 #1.0
        self.lambda_cls = 1 #1.0
        
        self.obj_scale = 1 #5
        self.noobj_scale = 1 #1
        
        self.ignore_thres = 0.5
        
        self.mseloss = nn.MSELoss(size_average=False)
        self.bceloss = nn.BCELoss(size_average=False)
        self.bceloss_average = nn.BCELoss(size_average=True)
        
        self.training = False
 
    def forward(self, x, target=None):
        #x : bs x nA*(5 + num_classes) * h * w
        nB = x.shape[0]
        nA = len(self.anchors)
        nH, nW = x.shape[2], x.shape[3]
        stride = self.img_dim[0] / nH
        anchors = torch.FloatTensor(self.anchors) / stride
        
        #Reshape predictions from [B x [A * (5 + numClass)] x H x W] to [B x A x H x W x (5 + numClass)]
        preds = x.view(nB, nA, self.bbox_attrib, nH, nW).permute(0, 1, 3, 4, 2).contiguous()
        
        # tx, ty, tw, wh
        preds_xy = preds[..., :2]
        preds_wh = preds[..., 2:4]
        preds_conf = preds[..., 4].sigmoid()
        preds_cls = preds[..., 5:].sigmoid()
        
        # Calculate cx, cy, anchor mesh
        mesh_x = torch.arange(nW).repeat(nW,1).unsqueeze(2)
        mesh_y = torch.arange(nH).repeat(nH,1).t().unsqueeze(2)
        mesh_xy = torch.cat((mesh_x,mesh_y), 2)
        mesh_anchors = anchors.view(1, nA, 1, 1, 2).repeat(1, 1, nH, nW, 1)
        
        # pred_boxes holds bx,by,bw,bh
        pred_boxes = torch.FloatTensor(preds[..., :4].shape)
        pred_boxes[..., :2] = preds_xy.sigmoid().cpu().detach() + mesh_xy # sig(tx) + cx
        pred_boxes[..., 2:4] = preds_wh.cpu().detach().exp() * mesh_anchors  # exp(tw) * anchor
        
        if target is not None:
            obj_mask, noobj_mask, tconf, tcls, tx, ty, tw, th, nCorrect, nGT = self.build_target_tensor(
                                                                    pred_boxes, target.cpu(), anchors, (nH, nW), self.numClass,
                                                                    self.ignore_thres)
            
            recall = float(nCorrect / nGT) if nGT else 1
            
            # masks for loss calculations
            obj_mask, noobj_mask = obj_mask.cuda(), noobj_mask.cuda()
            cls_mask = (obj_mask == 1)
            tconf, tcls = tconf.cuda(), tcls.cuda()
            tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()

            loss_x = self.lambda_xy * self.mseloss(preds_xy[..., 0] * obj_mask, tx * obj_mask) / nB
            loss_y = self.lambda_xy * self.mseloss(preds_xy[..., 1] * obj_mask, ty * obj_mask) / nB
            loss_w = self.lambda_wh * self.mseloss(preds_wh[..., 0] * obj_mask, tw * obj_mask) / nB
            loss_h = self.lambda_wh * self.mseloss(preds_wh[..., 1] * obj_mask, th * obj_mask) / nB

            loss_conf = self.lambda_conf * \
                        ( self.obj_scale * self.bceloss(preds_conf * obj_mask, obj_mask) + \
                          self.noobj_scale * self.bceloss(preds_conf * noobj_mask, noobj_mask * 0) ) / nB
            loss_cls = self.lambda_cls * self.bceloss(preds_cls[cls_mask], tcls[cls_mask]) / nB
            loss =  loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls 
                
            return loss, loss.item(), loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), \
                   loss_conf.item(), loss_cls.item(), \
                   nCorrect, nGT
           
        # Return predictions if not training 
        out = torch.cat((pred_boxes.cuda() * stride, 
                         preds_conf.cuda().unsqueeze(4),
                         preds_cls.cuda() ), 4)
        
        # Reshape predictions from [B x A x H x W x (5 + numClass)] to [B x [A x H x W] x (5 + numClass)]
        # such that predictions at different strides could be concatenated on the same dimension
        out = out.permute(0, 2, 3, 1, 4).contiguous().view(nB, nA*nH*nW, self.bbox_attrib)
        return out

    def build_target_tensor(self, preds, target, anchors, inp_dim, numClass, ignore_thres):
        nB = target.shape[0]
        nA = len(anchors)
        nH, nW = inp_dim[0], inp_dim[1]
        nCorrect = 0
        nGT = 0
        target = target.float()

        obj_mask = torch.zeros(nB, nA, nH, nW, requires_grad=False)
        noobj_mask = torch.ones(nB, nA, nH, nW, requires_grad=False)
        tconf= torch.zeros(nB, nA, nH, nW, requires_grad=False)
        tcls= torch.zeros(nB, nA, nH, nW, numClass, requires_grad=False)
        tx = torch.zeros(nB, nA, nH, nW, requires_grad=False)
        ty = torch.zeros(nB, nA, nH, nW, requires_grad=False)
        tw = torch.zeros(nB, nA, nH, nW, requires_grad=False)
        th = torch.zeros(nB, nA, nH, nW, requires_grad=False)

        for b in range(nB):
            for t in range(target.shape[1]):
                if target[b, t].sum() == 0:
                    break;
                nGT += 1

                gx = target[b, t, 1] * nW
                gy = target[b, t, 2] * nH
                gw = target[b, t, 3] * nW
                gh = target[b, t, 4] * nH
                gi = int(gx)
                gj = int(gy)

                # preds - [A x W x H x 4]  
                # Do not train for objectness(noobj) if anchor iou > threshold.
                tmp_gt_boxes = torch.FloatTensor([gx, gy, gw, gh]).unsqueeze(0)
                tmp_pred_boxes = preds[b].view(-1, 4)
                tmp_ious, _ = torch.max(bbox_iou(tmp_pred_boxes, tmp_gt_boxes, mode="cxcywh"), 1)
                ignore_idx = (tmp_ious > ignore_thres).view(nA, nH, nW)
                noobj_mask[b][ignore_idx] = 0
                
                #find best fit anchor for each ground truth box
                tmp_gt_boxes = torch.FloatTensor([[0, 0, gw, gh]])
                tmp_anchor_boxes = torch.cat((torch.zeros(nA, 2), anchors), 1)
                tmp_ious = bbox_iou(tmp_anchor_boxes, tmp_gt_boxes, mode="cxcywh")
                best_anchor = torch.argmax(tmp_ious, 0).item()
                
                #find iou for best fit anchor prediction box against the ground truth box
                tmp_gt_box = torch.FloatTensor([gx, gy, gw, gh]).unsqueeze(0)
                tmp_pred_box = preds[b, best_anchor, gj, gi].view(-1, 4)
                tmp_iou = bbox_iou(tmp_gt_box, tmp_pred_box, mode="cxcywh")
                
                if tmp_iou > 0.5:
                    nCorrect += 1

                obj_mask[b, best_anchor, gj, gi] = 1
                tconf[b, best_anchor, gj, gi] = 1
                tcls[b, best_anchor, gj, gi, int(target[b, t, 0])] = 1
                sig_x = gx - gi
                sig_y = gy - gj
                tx[b, best_anchor, gj, gi] = torch.log(sig_x/(1-sig_x) + 1e-16)
                ty[b, best_anchor, gj, gi] = torch.log(sig_y/(1-sig_y) + 1e-16)
                tw[b, best_anchor, gj, gi] = torch.log(gw / anchors[best_anchor, 0] + 1e-16)
                th[b, best_anchor, gj, gi] = torch.log(gh / anchors[best_anchor, 1] + 1e-16)

        return obj_mask, noobj_mask, tconf, tcls, tx, ty, tw, th, nCorrect, nGT


# Legacy version with no back-propagation
class YoloLayer_forward(nn.Module):
    def __init__(self, anchors, numClass):
        super().__init__()
        self.anchors = torch.Tensor(anchors).cuda()
        self.numClass = numClass
        
    def forward(self, x, img_dim):
        grid_size = x.shape[-1]
        numBBoxAttrib = 5 + self.numClass
        numAnchors = len(self.anchors)
        stride = img_dim // grid_size
        
        #Reshape the feature map from [batch x channel x grid_w x grid_h] to [batch x boundingbox x box attributes]
        x = x.permute(0,2,3,1).contiguous().view(-1, grid_size*grid_size*numAnchors, numBBoxAttrib)
        
        #Sigmoid tx,ty,to
        x[:,:,:2] = x[:,:,:2].sigmoid()
        x[:,:,4] = x[:,:,4].sigmoid()
        
        #Add cx,cy to sig(tx),sig(ty) , create cx,cy creating meshgrid of "grid_size"
        x_offset = torch.arange(grid_size).view(1,-1,1,1).repeat(grid_size,1,numAnchors,1).view(grid_size*grid_size*numAnchors, 1)
        y_offset = torch.arange(grid_size).view(-1,1,1,1).repeat(1,grid_size,numAnchors,1).view(grid_size*grid_size*numAnchors, 1)
        mesh = torch.cat((x_offset,y_offset), 1)
        x[:,:,:2] = x[:,:,:2].add(mesh.cuda()   )
        
        #Rescale anchors to fit the stride, multiply it by e^(tw) and e^(th)
        x[:,:,2:4] = torch.exp(x[:,:,2:4]).mul(self.anchors.div(stride).repeat(grid_size**2,1))
        
        #Sigmoid class scores
        x[:,:,5:] = x[:,:,5:].sigmoid()
        
        #Rescale bx,by,bw,bh by stride to orginal image size
        x[:,:,:4] *= stride
                
        return x