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

    def forward(self, pred, y_hat, model):

        p_shape = pred.shape
        print("prediction shape : {}".format(p_shape))
        print("Gound Truth shape : {}".format(y_hat.shape))
        y_shape = y_hat.shape

        prior_boxes = model.prior_boxes
        num_prior_boxes = model.num_prior_boxes
        num_classes = model.num_classes
        num_of_boxinfo = 5
        num_of_anchorbox_elem = num_of_boxinfo + num_classes

        image_W, image_H = model.input_size
        grid_S = p_shape[1]

        dx = image_W / grid_S
        dy = image_H / grid_S

        p_b, p_c, p_h, p_w = p_shape
        y_b, y_h, y_w, y_c = y_shape

        # slice tensor of y_hat
        y_t0 = y_hat[:, :, :, 0]
        y_bx = y_hat[:, :, :, 1]
        y_by = y_hat[:, :, :, 2]
        y_bw = y_hat[:, :, :, 3]
        y_bh = y_hat[:, :, :, 4]


        # 로스를 ojbect있는 구간과 없는 구간으로 나누어 계산한 후, 합칠 것.
        objness = y_t0
        non_objness = self.build_nonobj_indicator_block(y_t0)

        # find objness index
        objness_np = objness.numpy()
        objness_index = np.where(objness_np == 1.)
        print(objness_index)

        # calc loss condition as objness == 1
        obj_loss = list()
        num_of_objness = len(objness_index[0])
        for idx in range(num_of_objness):

            pred_in_obj_idx = pred[objness_index[0][idx], :, objness_index[1][idx], objness_index[2][idx]]
            gt_in_obj_idx = y_hat[objness_index[0][idx], objness_index[1][idx], objness_index[2][idx], :]
            # tensor([1.0000, 0.5884, 0.1257, 0.8982, 0.8568, 7.0000])
            ious = list()
            for i in range(num_prior_boxes):

                start_idx = i * (num_of_anchorbox_elem)

                cell_x_idx = objness_index[2][idx]
                cell_y_idx = objness_index[1][idx]
                prior_box = prior_boxes[i]

                anchor_box = pred_in_obj_idx[start_idx : (start_idx + num_of_anchorbox_elem)]

                p_objness = anchor_box[0]
                p_tx = anchor_box[1]
                p_ty = anchor_box[2]
                p_tw = anchor_box[3]
                p_th = anchor_box[4]
                p_cls = anchor_box[5:]

                p_bx = dx * cell_x_idx + int(dx * p_tx)
                p_by = dy * cell_y_idx + int(dy * p_ty)
                p_bw = int(prior_box[0] * p_tw * image_W)
                p_bh = int(prior_box[1] * p_th * image_H)

                g_objness = gt_in_obj_idx[0]
                g_tx = gt_in_obj_idx[1]
                g_ty = gt_in_obj_idx[2]
                g_tw = gt_in_obj_idx[3]
                g_th = gt_in_obj_idx[4]
                # it should be apply one-hot encoding
                g_cls = gt_in_obj_idx[5]

                g_bx = dx * cell_x_idx + int(dx * g_tx)
                g_by = dy * cell_y_idx + int(dy * g_ty)
                g_bw = int(g_tw * image_W)
                g_bh = int(g_th * image_H)

                # iou 계산식이 필요함.


                exit()
                #_objness =

            #print(pred_in_obj_idx)
            #print(pred_in_obj_idx.shape)

            pass
        exit()
        idx_length = len(objness_index[0])
        print(idx_length)
        for i in range(idx_length):
            print(objness[objness_index[0][i], objness_index[1][i], objness_index[2][i]])

        torch_index = torch.where(objness == 1., objness, torch.zeros(objness.shape))
        print(torch_index)
        print(torch_index[0])
        print(objness[0])
        exit()
        print(np.where(objness.numpy() == 1., objness.numpy()))
        print(objness[objness == 1.])

        exit()

        # TODO seperate non object loss, object loss

        # TODO Apply only object cell in label tensor block
        # slice tensor of prediction

        boxes = list()
        for i in range(model.num_prior_boxes):
            idx = i*num_of_anchorbox_elem

            _t0 = pred[:, idx, :, :]
            _tx = pred[:, idx + 1, :, :]
            _ty = pred[:, idx + 2, :, :]
            _tw = pred[:, idx + 3, :, :]
            _th = pred[:, idx + 4, :, :]
            _cls = pred[:, idx + 5:idx + 5 + model.num_classes, :, :]

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
