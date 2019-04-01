import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from utils.util import calculate_intersection_over_union


def nll_loss(output, target):
    return F.nll_loss(output, target)


class DetectionLoss(torch.nn.Module):
    def __init__(self):
        super(DetectionLoss, self).__init__()

        self.lambda_obj = 5.
        self.lambda_nonobj = .5

    def forward(self, pred, y_hat, model, input):

        p_shape = pred.shape
        y_shape = y_hat.shape

        prior_boxes = model.prior_boxes
        num_prior_boxes = model.num_prior_boxes
        num_classes = model.num_classes
        num_of_boxinfo = 5
        num_of_anchorbox_elem = num_of_boxinfo + num_classes

        image_W, image_H = model.input_size
        grid_S = p_shape[2]

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

        loss = list()
        # 로스를 ojbect있는 구간과 없는 구간으로 나누어 계산한 후, 합칠 것.
        objness = y_t0
        non_objness = self.build_nonobj_indicator_block(y_t0)
        basis = non_objness.view(non_objness.shape[0], 1, non_objness.shape[1], non_objness.shape[2])
        non_objness = non_objness.view(non_objness.shape[0], 1, non_objness.shape[1], non_objness.shape[2])
        for i in range(num_classes - 1):
            non_objness = torch.cat((non_objness, basis), 1)

        # calc loss condition as objness = 0
        for i in range(num_prior_boxes):
            start_idx = i * (num_of_anchorbox_elem)
            anchor_box = pred[:, start_idx: (start_idx + num_of_anchorbox_elem), :, :]
            _cls = anchor_box[:, 5:, :, :]
            y_onehot_cls = torch.zeros(_cls.shape)
            _noobj_cls_loss = torch.pow(_cls - y_onehot_cls, 2)
            _loss = torch.sum(_noobj_cls_loss * non_objness)

            loss.append(_loss)

        # calc loss condition as objness == 1
        # find objness index
        objness_np = objness.numpy()
        objness_index = np.where(objness_np == 1.)

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

                p_center_x = dx * cell_x_idx + int(dx * p_tx)
                p_center_y = dy * cell_y_idx + int(dy * p_ty)
                p_w = int(prior_box[0] * p_tw * image_W)
                p_h = int(prior_box[1] * p_th * image_H)

                p_bx = p_center_x - (p_w // 2)
                p_by = p_center_y - (p_h // 2)
                p_bw = p_bx + p_w
                p_bh = p_by + p_h

                g_objness = gt_in_obj_idx[0]
                g_tx = gt_in_obj_idx[1]
                g_ty = gt_in_obj_idx[2]
                g_tw = gt_in_obj_idx[3]
                g_th = gt_in_obj_idx[4]
                # it should be apply one-hot encoding
                g_cls = gt_in_obj_idx[5]

                g_center_x = dx * cell_x_idx + int(dx * g_tx)
                g_center_y = dy * cell_y_idx + int(dy * g_ty)
                g_w = int(g_tw * image_W)
                g_h = int(g_th * image_H)

                g_bx = g_center_x - (g_w // 2)
                g_by = g_center_y - (g_h // 2)
                g_bw = g_bx + g_w
                g_bh = g_by + g_h

                _pred_box = [p_bx, p_by, p_bw, p_bh]
                _gt_box = [g_bx, g_by, g_bw, g_bh]
                iou = calculate_intersection_over_union(_pred_box, _gt_box)
                ious.append(iou)

                # sanity check using visualization
                """
                import torchvision.transforms as transforms
                from PIL import Image, ImageDraw
                import matplotlib.pyplot as plt
                # iou 계산식이 필요함.

                img = input[objness_index[0][idx], :, :, :]

                print("predicted boxes : {}".format([p_bx, p_by, p_bw, p_bh]))
                print("gt boxes : {}".format([g_bx, g_by, g_bw, g_bh]))
                print("iou : {}".format(iou))

                img = transforms.ToPILImage()(img)
                draw = ImageDraw.Draw(img)
                dx = int(dx)
                dy = int(dy)
                y_start = 0
                y_end = image_H

                for i in range(0, image_W, dx):
                    line = ((i, y_start), (i, y_end))
                    draw.line(line, fill="red")

                x_start = 0
                x_end = image_W
                for i in range(0, image_H, dy):
                    line = ((x_start, i), (x_end, i))
                    draw.line(line, fill="red")

                print("grid_S : {}, dx : {}, dy : {}".format(grid_S, dx, dy))
                print("image_W : {}, image_H : {}".format(image_W, image_H))
                print("image size : {}".format(img.size))
                print("cls : {}".format(g_cls))

                draw.rectangle(((int(p_bx), int(p_by)), (int(p_bw), int(p_bh))), outline="blue")
                draw.rectangle(((int(g_bx), int(g_by)), (int(g_bw), int(g_bh))), outline="green")
                draw.ellipse(((g_center_x - 2, g_center_y - 2),
                              (g_center_x + 2, g_center_y + 2)),
                             fill='green')

                cls_list = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
                draw.text((g_bx, g_by), cls_list[int(g_cls)])

                plt.figure()
                plt.imshow(img)
                plt.show()
                """
            max_iou_value = max(ious)

            if max_iou_value >= .5:
                #print("IOU : {}".format(ious))
                #print("Max IOU : {}, index : {}".format(max(ious), ious.index(max(ious))))

                max_iou_idx = ious.index(max_iou_value)
                start_idx = max_iou_idx * (num_of_anchorbox_elem)
                anchor_box = pred_in_obj_idx[start_idx: (start_idx + num_of_anchorbox_elem)]

                p_objness = anchor_box[0]
                p_tx = anchor_box[1]
                p_ty = anchor_box[2]
                p_tw = anchor_box[3]
                p_th = anchor_box[4]
                p_cls = anchor_box[5:]

                g_objness = gt_in_obj_idx[0]
                g_tx = gt_in_obj_idx[1]
                g_ty = gt_in_obj_idx[2]
                g_tw = gt_in_obj_idx[3]
                g_th = gt_in_obj_idx[4]

                # ONE HOT Encoding
                g_cls = gt_in_obj_idx[5].type(torch.LongTensor)
                num_digits = p_cls.shape[-1]
                y_onehot_cls = torch.zeros(1, num_digits).squeeze()
                y_onehot_cls[g_cls-1] = 1.

                _objness = p_objness * max_iou_value

                _obj_loss = torch.pow(_objness - g_objness, 2)
                _tx_loss = torch.pow(p_tx - g_tx, 2)
                _ty_loss = torch.pow(p_ty - g_ty, 2)
                _tw_loss = torch.pow(p_tw - g_tw, 2)
                _th_loss = torch.pow(p_th - g_th, 2)
                _cls_loss = torch.sum(torch.pow(p_cls - y_onehot_cls, 2))

                box_loss = self.lambda_obj * (_tx_loss + _ty_loss + _tw_loss + _th_loss)

                _loss = box_loss + _obj_loss + _cls_loss

                loss.append(_loss)


        Loss = loss[0]
        for i in range(len(loss)):
            Loss += loss[i] if i != 0 else 0

        Loss = Loss / p_shape[0]

        return Loss

    @staticmethod
    def build_nonobj_indicator_block(objness):
        # reference You Only Look Once Loss function
        # https://arxiv.org/pdf/1506.02640.pdf
        return torch.neg(torch.add(objness, -1))
