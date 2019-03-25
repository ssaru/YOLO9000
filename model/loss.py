import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def nll_loss(output, target):
    return F.nll_loss(output, target)


class DetectionLoss(torch.nn.Module):
    def __init__(self):
        super(DetectionLoss, self).__init__()

        self.lambda_obj = 5.
        self.lambda_nonobj = .5

    def forward(self, prediction, y_hat, model):

        p_shape = prediction.shape
        print("prediction shape : {}".format(p_shape))
        y_shape = y_hat.shape

        p_b, p_c, p_h, p_w = p_shape
        y_b, y_h, y_w, y_c = y_shape

        # slice tensor of y_hat
        y_t0 = y_hat[:, :, :, 0]
        y_bx = y_hat[:, :, :, 1]
        y_by = y_hat[:, :, :, 2]
        y_bw = y_hat[:, :, :, 3]
        y_bh = y_hat[:, :, :, 4]


        objness = y_t0
        non_objness = self.build_nonobj_indicator_block(y_t0)

        # TODO seperate non object loss, object loss

        # TODO Apply only object cell in label tensor block
        # slice tensor of prediction

        boxes = list()
        for i in range(model.num_prior_boxes):
            idx = i*(5+model.num_classes)

            _t0 = prediction[:, idx, :, :]
            _tx = prediction[:, idx + 1, :, :]
            _ty = prediction[:, idx + 2, :, :]
            _tw = prediction[:, idx + 3, :, :]
            _th = prediction[:, idx + 4, :, :]
            _cls = prediction[:, idx + 5:idx + 5 + model.num_classes, :, :]

            boxes.append([_tx, _ty, _tw, _th])

        print()
        exit()

        # TODO. 1. IOU Calculation method should be rewrite. cause didn't consider that operate tensorblock
        # TODO. 2. Convert [xmin, ymin, xmax, ymax] Box style for iou calculation

        exit()
        # TODO. 3. Calc IOU

        # TODO. 4. Finding backprop Targets
        # TODO. 5. Apply zero gradient to non Target
        # TODO. 6. getting Pr(objness) * IOU(b, object) from IOU & t0
        # TODO. 7. getting loss of each elements

        # TODO. 0. Implement one-hot encoding function
        y_cls = self.onehot(y_hat[:, :, :, 5], model.num_classes)

        pass

    def onehot(self, label, num_of_cls):
        print(label)
        print(num_of_cls)
        pass

    def iou(self):
        # iou calculation should be considered tensor

        pass

    @staticmethod
    def build_nonobj_indicator_block(objness):
        # reference You Only Look Once Loss function
        # https://arxiv.org/pdf/1506.02640.pdf
        return torch.neg(torch.add(objness, -1))
