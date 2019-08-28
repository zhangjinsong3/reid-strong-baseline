# -*- coding: utf-8 -*-
"""
Created on 19-8-27

@author: 01376022

Focal loss from https://github.com/clcarwin/focal_loss_pytorch

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

# ====================Another version of focal loss =================================================
# # !/usr/bin/env python
# # -*- coding: utf-8 -*-
# # --------------------------------------------------------
# # Licensed under The MIT License [see LICENSE for details]
# # Written by Chao CHEN (chaochancs@gmail.com)
# # Created On: 2017-08-11
# # --------------------------------------------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
#
#
# class FocalLoss(nn.Module):
#     r"""
#         This criterion is a implemenation of Focal Loss, which is proposed in
#         Focal Loss for Dense Object Detection.
#
#             Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
#
#         The losses are averaged across observations for each minibatch.
#         Args:
#             alpha(1D Tensor, Variable) : the scalar factor for this criterion
#             gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
#                                    putting more focus on hard, misclassiﬁed examples
#             size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
#                                 However, if the field size_average is set to False, the losses are
#                                 instead summed for each minibatch.
#     """
#
#     def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
#         super(FocalLoss, self).__init__()
#         if alpha is None:
#             self.alpha = Variable(torch.ones(class_num, 1))
#         else:
#             if isinstance(alpha, Variable):
#                 self.alpha = alpha
#             else:
#                 self.alpha = Variable(alpha)
#         self.gamma = gamma
#         self.class_num = class_num
#         self.size_average = size_average
#
#     def forward(self, inputs, targets):
#         N = inputs.size(0)
#         print(N)
#         C = inputs.size(1)
#         P = F.softmax(inputs)
#
#         class_mask = inputs.data.new(N, C).fill_(0)
#         class_mask = Variable(class_mask)
#         ids = targets.view(-1, 1)
#         class_mask.scatter_(1, ids.data, 1.)
#         # print(class_mask)
#
#         if inputs.is_cuda and not self.alpha.is_cuda:
#             self.alpha = self.alpha.cuda()
#         alpha = self.alpha[ids.data.view(-1)]
#
#         probs = (P * class_mask).sum(1).view(-1, 1)
#
#         log_p = probs.log()
#         # print('probs size= {}'.format(probs.size()))
#         # print(probs)
#
#         batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
#         # print('-----bacth_loss------')
#         # print(batch_loss)
#
#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#         return loss
#
#
# if __name__ == "__main__":
#     alpha = torch.rand(21, 1)
#     print(alpha)
#     FL = FocalLoss(class_num=5, gamma=0)
#     CE = nn.CrossEntropyLoss()
#     N = 4
#     C = 5
#     inputs = torch.rand(N, C)
#     targets = torch.LongTensor(N).random_(C)
#     inputs_fl = Variable(inputs.clone(), requires_grad=True)
#     targets_fl = Variable(targets.clone())
#
#     inputs_ce = Variable(inputs.clone(), requires_grad=True)
#     targets_ce = Variable(targets.clone())
#     print('----inputs----')
#     print(inputs)
#     print('---target-----')
#     print(targets)
#
#     fl_loss = FL(inputs_fl, targets_fl)
#     ce_loss = CE(inputs_ce, targets_ce)
#     print('ce = {}, fl ={}'.format(ce_loss.data[0], fl_loss.data[0]))
#     fl_loss.backward()
#     ce_loss.backward()
#     # print(inputs_fl.grad.data)
#     print(inputs_ce.grad.data)
# ====================Another version of focal loss =================================================

