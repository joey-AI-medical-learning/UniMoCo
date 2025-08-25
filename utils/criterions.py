import torch.nn.functional as F
import torch
import logging
import torch.nn as nn


__all__ = ['GeneralizedDiceLoss', 'dice_loss', 'LabelSmoothingLoss']

cross_entropy = F.cross_entropy

def dice_loss(output, target, num_cls=5, eps=1e-7):
    target = target.float()
    for i in range(num_cls):
        num = torch.sum(output[:,i,:,:,:] * target[:,i,:,:,:])
        l = torch.sum(output[:,i,:,:,:])
        r = torch.sum(target[:,i,:,:,:])
        if i == 0:
            dice = 2.0 * num / (l+r+eps)
        else:
            dice += 2.0 * num / (l+r+eps)
    return 1.0 - 1.0 * dice / num_cls

def softmax_weighted_loss(output, target, num_cls=5):
    target = target.float()
    B, _, H, W, Z = output.size()
    for i in range(num_cls):
        outputi = output[:, i, :, :, :]
        targeti = target[:, i, :, :, :]
        weighted = 1.0 - (torch.sum(targeti, (1,2,3)) * 1.0 / torch.sum(target, (1,2,3,4)))
        weighted = torch.reshape(weighted, (-1,1,1,1)).repeat(1,H,W,Z)
        if i == 0:
            cross_loss = -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        else:
            cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
    cross_loss = torch.mean(cross_loss)
    return cross_loss



# Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
def GeneralizedDiceLoss(output,target,eps=1e-5,weight_type='square'): # Generalized dice loss

    if target.dim() == 4:
        target[target == 4] = 3 # label [4] -> [3]
        target = expand_target(target, n_class=output.size()[1]) # [N,H,W,D] -> [N,4，H,W,D]

    output = flatten(output)[1:,...] # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:,...] # [class, N*H*W*D]

    target_sum = target.sum(-1) # sub_class_voxels [3,1] -> 3个voxels
    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :',weight_type)

    intersect = (output * target).sum(-1)
    intersect_sum = (intersect * class_weights).sum()
    denominator = (output + target).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    loss1 = 2*intersect[0] / (denominator[0] + eps)
    loss2 = 2*intersect[1] / (denominator[1] + eps)
    loss3 = 2*intersect[2] / (denominator[2] + eps)

    return 1 - 2. * intersect_sum / denominator_sum, [loss1.data, loss2.data, loss3.data]


def expand_target(x, n_class,mode='softmax'):
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == 'softmax':
        xx[:,1,:,:,:] = (x == 1)
        xx[:,2,:,:,:] = (x == 2)
        xx[:,3,:,:,:] = (x == 3)
    if mode.lower() == 'sigmoid':
        xx[:,0,:,:,:] = (x == 1)
        xx[:,1,:,:,:] = (x == 2)
        xx[:,2,:,:,:] = (x == 3)
    return xx.to(x.device)

def flatten(tensor):
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order)

    return transposed.reshape(C, -1)


def temp_kl_loss_bs(logit_s, logit_t, temp=2.0):

    pred_s = F.softmax(logit_s/temp, dim=1)
    pred_t = F.softmax(logit_t/temp, dim=1)

    pred_s = torch.clamp(pred_s, min=0.005, max=1)
    pred_t = torch.clamp(pred_t, min=0.005, max=1)
    pred_s = torch.log(pred_s)
    kl_loss = temp * temp * torch.mul(pred_t, torch.log(pred_t)-pred_s)
    kl_loss = torch.mean(kl_loss, dim=(1,2,3,4))

    return kl_loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        loss = -(one_hot * F.log_softmax(pred, dim=1)).sum(dim=1)
        return loss.mean()