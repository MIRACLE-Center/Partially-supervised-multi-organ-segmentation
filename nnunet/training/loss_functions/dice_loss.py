#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch
from nnunet.training.loss_functions.ND_Crossentropy import CrossentropyND
from nnunet.training.loss_functions.TopK_loss import TopKLoss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from nnunet.utilities.mk_utils import get_tag_index
from torch import nn
import numpy as np


class GDL(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False, square_volumes=False):
        """
        square_volumes will square the weight term. The paper recommends square_volumes=True; I don't (just an intuition)
        """
        super(GDL, self).__init__()

        self.square_volumes = square_volumes
        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        shp_y = y.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if len(shp_x) != len(shp_y):
            y = y.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(x.shape, y.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = y
        else:
            gt = y.long()
            y_onehot = torch.zeros(shp_x)
            if x.device.type == "cuda":
                y_onehot = y_onehot.cuda(x.device.index)
            y_onehot.scatter_(1, gt, 1)

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if not self.do_bg:
            x = x[:, 1:]
            y_onehot = y_onehot[:, 1:]

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y_onehot, axes, loss_mask, self.square)

        # GDL weight computation, we use 1/V
        volumes = sum_tensor(y_onehot, axes) + 1e-6 # add some eps to prevent div by zero

        if self.square_volumes:
            volumes = volumes ** 2

        # apply weights
        tp = tp / volumes
        fp = fp / volumes
        fn = fn / volumes

        # sum over classes
        if self.batch_dice:
            axis = 0
        else:
            axis = 1

        tp = tp.sum(axis, keepdim=False)
        fp = fp.sum(axis, keepdim=False)
        fn = fn.sum(axis, keepdim=False)

        # compute dice
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        dc = dc.mean()

        return -dc


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class MCCLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_mcc=False, do_bg=True, smooth=0.0):
        """
        based on matthews correlation coefficient
        https://en.wikipedia.org/wiki/Matthews_correlation_coefficient

        Does not work. Really unstable. F this.
        """
        super(MCCLoss, self).__init__()

        self.smooth = smooth
        self.do_bg = do_bg
        self.batch_mcc = batch_mcc
        self.apply_nonlin = apply_nonlin

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        voxels = np.prod(shp_x[2:])

        if self.batch_mcc:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, tn = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)
        tp /= voxels
        fp /= voxels
        fn /= voxels
        tn /= voxels

        nominator = tp * tn - fp * fn + self.smooth
        denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5 + self.smooth

        mcc = nominator / denominator

        if not self.do_bg:
            if self.batch_mcc:
                mcc = mcc[1:]
            else:
                mcc = mcc[:, 1:]
        mcc = mcc.mean()

        return -mcc


class SoftDiceLossSquared(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        squares the terms in the denominator as proposed by Milletari et al.
        """
        super(SoftDiceLossSquared, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        shp_y = y.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(x.shape, y.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y = y.long()
                y_onehot = torch.zeros(shp_x)
                if x.device.type == "cuda":
                    y_onehot = y_onehot.cuda(x.device.index)
                y_onehot.scatter_(1, y, 1).float()

        intersect = x * y_onehot
        # values in the denominator get smoothed
        denominator = x ** 2 + y_onehot ** 2

        # aggregation was previously done in get_tp_fp_fn, but needs to be done here now (needs to be done after
        # squaring)
        intersect = sum_tensor(intersect, axes, False) + self.smooth
        denominator = sum_tensor(denominator, axes, False) + self.smooth

        dc = 2 * intersect / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc



class pann_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", ex=True):
        super(pann_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = CrossentropyND(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target, default_task, cur_task):
        if cur_task == default_task:#(task MALB) normal run
            # print(f"equal:{cur_task}/{default_task}")
            dc_loss = self.dc(net_output, target)
            ce_loss = self.ce(net_output, target)
            if self.aggregate == "sum":
                result = ce_loss + dc_loss
            elif self.aggregate == "ce":
                result = ce_loss
            elif self.aggregate == "dc":
                result = dc_loss
            else:
                # reserved for other stuff (later?)
                raise NotImplementedError("nah son")
        else:#unlabel_task
            # print(f"not_equal:{cur_task}/{default_task}")
            target_onehot = onehot_transform(target, len(cur_task)+1)
            new_label = torch.zeros_like(net_output)
            if net_output.device.type == "cuda":
                new_label = new_label.cuda(target_onehot.device.index)
  
            unlabel_task_index = []
            unlabel_task_index.append(0)#add bkg_index
            for i, task in enumerate(default_task):
                if task not in cur_task:
                    unlabel_task_index.append(i+1)
            
            for i, task in enumerate(default_task):
                if task in cur_task: #label_task
                    j = cur_task.index(task)
                    new_label[:, i+1, :, :] = target_onehot[:, j+1, :, :]
                else:#unlabel_task
                    max_index = torch.argmax(net_output[:,unlabel_task_index,:,:],dim=1,keepdim=True)
                    new_label.scatter_(1,max_index,1)

            clear_mask = torch.sum(new_label, dim=1,keepdim=True)>1
            #torch.masked_select(new_label, clear_mask)
            new_label.masked_fill_(mask=clear_mask, value=0)#tiaozheng conflict
            for i, task in enumerate(default_task):
                if task in cur_task: #label_task
                    j = cur_task.index(task)
                    new_label[:, i+1, :, :] = target_onehot[:, j+1, :, :]
                else:#unlabel_task
                    pass 
            #torch.masked_select(new_label, clear_mask)

            #new_lable(one-hot)-> general_label
            new_label = torch.topk(new_label, 1, dim=1)[1]
            dc_loss = self.dc(net_output, new_label)
            ce_loss = self.ce(net_output, new_label)       
            # epsilon=1
            # ex_loss = self.ex(net_output, not_gt)+self.ex_CE(net_output+epsilon,not_gt)
            if self.aggregate == "sum":
                result = ce_loss + dc_loss
            elif self.aggregate == "ce":
                result = ce_loss
            elif self.aggregate == "dc":
                result = dc_loss
            else:
                # reserved for other stuff (later?)
                raise NotImplementedError("nah son")
                # result = ex_loss
        return result



class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = CrossentropyND(**ce_kwargs)
        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) if self.weight_ce != 0 else 0
        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result


class GDL_and_CE_loss(nn.Module):
    def __init__(self, gdl_dice_kwargs, ce_kwargs, aggregate="sum"):
        super(GDL_and_CE_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = CrossentropyND(**ce_kwargs)
        self.dc = GDL(softmax_helper, **gdl_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False):
        super(DC_and_topk_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = TopKLoss(**ce_kwargs)
        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later?)
        return result


class Exclusion_loss(nn.Module):
    def __init__(self, union_func):
        super(Exclusion_loss, self).__init__()
        self.union = union_func

    def forward(self, network_output, target):
        return -self.union(network_output, target)#Intersection between prediction and En is as small as possible!!!(just completely contrary to the dc/crossEntropy loss)

class DC_CE_Marginal_Exclusion_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", ex=True):
        super(DC_CE_Marginal_Exclusion_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = CrossentropyND(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        self.ex = Exclusion_loss(self.dc)
        self.ex_CE = Exclusion_loss(self.ce)
        self.ex_choice = ex
        print(f"mode:{aggregate}/ weight:[1:1] with exclusion:{ex}")

    def forward(self, net_output, target, default_task, cur_task):
        if cur_task == default_task:
            # print(f"equal:{cur_task}/{default_task}")
            dc_loss = self.dc(net_output, target)
            ce_loss = self.ce(net_output, target)
            if self.aggregate == "sum":
                result = ce_loss + dc_loss
            elif self.aggregate == "ce":
                result = ce_loss
            elif self.aggregate == "dc":
                result = dc_loss
            else:
                # reserved for other stuff (later?)
                raise NotImplementedError("nah son")
        else:
            # print(f"not_equal:{cur_task}/{default_task}")
            target_onehot = onehot_transform(target, len(cur_task)+1)
            merged_pre = merge_prediction(
                net_output, target_onehot, cur_task, default_task)
            not_gt = expand_gt(net_output, target_onehot,
                               cur_task, default_task)
            dc_loss = self.dc(merged_pre, target)
            ce_loss = self.ce(merged_pre, target)
            ex_loss = self.ex(net_output, not_gt)
            # epsilon=1
            # ex_loss = self.ex(net_output, not_gt)+self.ex_CE(net_output+epsilon,not_gt)
            if self.aggregate == "sum":
                result = ce_loss + dc_loss
            elif self.aggregate == "ce":
                result = ce_loss
            elif self.aggregate == "dc":
                result = dc_loss
            else:
                # reserved for other stuff (later?)
                raise NotImplementedError("nah son")
            if self.ex_choice:
                result = result+2*ex_loss
                # result = ex_loss
        return result


def onehot_transform(tensor, depth):
    assert torch.max(tensor) < depth, f"{torch.max(tensor)}/{depth}"
    shp_tensor = list(tensor.shape)
    shp_tensor[1] = depth
    tensor = tensor.long()
    tensor_onehot = torch.zeros(shp_tensor)
    if tensor.device.type == "cuda":
        tensor_onehot = tensor_onehot.cuda(tensor.device.index)
    tensor_onehot.scatter_(1, tensor, 1)
    return tensor_onehot


def merge_prediction(net_output, target_onehot, cur_task, default_task):
    '''
        cur_task: GT task
        default_task: net_output task
    '''
    new_prediction = torch.zeros_like(target_onehot)
    if net_output.device.type == "cuda":
        new_prediction = new_prediction.cuda(net_output.device.index)
    new_prediction[:, 0, :, :] = net_output[:, 0, :, :]#先把bkg赋值(bkg不属于任何task)
  

    for i, task in enumerate(default_task):
        if task in cur_task: 
            j = cur_task.index(task)
            new_prediction[:, j+1, :, :] += net_output[:, i+1, :, :]
        else:
            new_prediction[:, 0, :, :] += net_output[:, i+1, :, :]
    return new_prediction


def expand_gt(net_output, target_onehot, cur_task, default_task):
    new_gt = torch.zeros_like(net_output)
    if net_output.device.type == "cuda":
        new_gt = new_gt.cuda(net_output.device.index)
    new_gt[:, 0, :, :] = 1-target_onehot[:, 0, :, :]
    for i, task in enumerate(default_task):
        if task in cur_task:
            j = cur_task.index(task)
            new_gt[:, i+1, :, :] = 1-target_onehot[:, j+1, :, :]
        else:
            new_gt[:, i+1, :, :] = 1-target_onehot[:, 0, :, :]
    return new_gt
