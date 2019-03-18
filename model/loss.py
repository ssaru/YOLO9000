import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def nll_loss(output, target):
    return F.nll_loss(output, target)


class Loss():
    def __init__(self, model):
        self.model = model

    def forward(self, prediction, y_hat, model):

        p_shape = prediction.shape
        y_shape = y_hat.shape

        p_b, p_h, p_w, p_c = p_shape
        y_b, y_h, y_w, y_c = y_shape

        # slice tensor of y_hat
        y_t0 = y_hat[:, :, :, 0]
        y_bx = y_hat[:, :, :, 1]
        y_by = y_hat[:, :, :, 2]
        y_bw = y_hat[:, :, :, 3]
        y_bh = y_hat[:, :, :, 4]
        
        # TODO. 0. Implement one-hot encoding function
        y_cls = one_hot(y_hat[:, :, :, 5])

        p_t0 = list()
        p_tx = list()
        p_ty = list()
        p_tw = list()
        p_th = list()
        p_cls = list()

        # slice tensor of prediction
        for i in range(self.model.num_prior_boxes):
            idx = i*5
            p_t0.append(prediction[:, :, :, idx])
            p_tx.append(prediction[:, :, :, idx + 1])
            p_ty.append(prediction[:, :, :, idx + 2])
            p_tw.append(prediction[:, :, :, idx + 3])
            p_th.append(prediction[:, :, :, idx + 4])
            p_cls.append(prediction[:, :, :, idx + 5:])

        # TODO. 1. Getting Bw, Bh from tw, th
        # TODO. 2. Calc IOU
        # TODO. 3. Finding backprop Targets
        # TODO. 4. Apply zero gradient to non Target
        # TODO. 5. getting Pr(objness) * IOU(b, object) from IOU & t0
        # TODO. 6. getting loss of each elements

        pass

