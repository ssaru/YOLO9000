import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from typing import List, Tuple, Dict
from utils.util import get_iou

class DetectionLoss(torch.nn.Module):
    def __init__(self, model):
        super(DetectionLoss, self).__init__()

        self._lambda_obj = 5.
        self._lambda_nonobj = .5
        self.prior_boxes = model.prior_boxes
        self.num_prior_boxes = model.num_prior_boxes
        self.num_classes = model.num_classes
        num_of_boxinfo = 5
        self.num_of_anchorbox_elem = num_of_boxinfo + self.num_classes
        self.image_width, self.image_height = model.input_size


    def forward(self, pred, y_hat, model, input):

        width_S, height_S = get_pred_resolution(pred)

        x_interval = get_interval(self.image_width, width_S)
        y_interval = get_interval(self.image_height, height_S)

        obj_index_map = get_obj_index_map(y_hat)
        nonobj_index_map = get_nonobj_index_map(obj_index_map, self.num_classes)

        nonobj_loss_list = get_nonobj_loss_list(pred, self.num_prior_boxes, self.num_of_anchorbox_elem)

        # calc loss condition as objness == 1
        # find objness index
        objness_np = obj_index_map.numpy()
        objness_index = np.where(objness_np == 1.)
        print(objness_index)
        print(type(objness_index))
        exit()

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

def get_obj_loss_list(pred: torch.tensor, target: torch.tensor, obj_index_map: torch.tensor,
                      prior_boxes: List[List[int]], x_interval: float, y_interval: float,
                      input_image_width: int, input_image_height: int, num_classes: int,
                      num_anchor_boxes: int, anchor_channels: int, lambda_obj: float) -> List[torch.tensor]:
    """get obj loss list


    Args:
        pred (torch.tensor) : result of inference
        target (torch.tensor) : label for detection consist of torch.tensor
        obj_index_map (torch.tensor) : object location indices map consist of torch.tensor.
        prior_boxes (List[List[int]]) : prior boxes
        x_interval (float) : ratio between image width with prediction width
        y_interval (float) : ratio between image height with prediction height
        input_image_width (int) : input image width
        input_image_height (int) : input image height
        num_classes (int) : number of classes
        num_anchor_boxes (int) : number of anchor boxes
        anchor_channels (int) : number of anchor channels
        lambda_obj (float) : loss weight about object existence case

    Retruns:
        obj_loss_list (List[torch.tensor]) : list of obj loss
    """


    obj_loss_list = list()

    object_map = get_obj_location_index(obj_index_map)
    num_of_obj = len(object_map[0])

    for idx in range(num_of_obj):
        pred_on_obj = pred[object_map[0][idx], :, object_map[1][idx], object_map[2][idx]]
        gt_on_obj = target[object_map[0][idx], object_map[1][idx], object_map[2][idx], :]

        ious = get_ious(pred_on_obj, gt_on_obj, object_map, prior_boxes,
                        x_interval, y_interval, input_image_width, input_image_height, num_anchor_boxes,
                        anchor_channels)

        max_iou = max(ious)


        if max_iou >= .5:
            max_iou_idx = ious.index(max_iou)
            anchor_box = get_anchor(pred, max_iou_idx, anchor_channels)

            pred_objness = anchor_box[0] * max_iou
            pred_tx = anchor_box[1]
            pred_ty = anchor_box[2]
            pred_tw = anchor_box[3]
            pred_th = anchor_box[4]
            pred_cls = anchor_box[5:]

            target_objness = target[0]
            target_tx = target[1]
            target_ty = target[2]
            target_tw = target[3]
            target_th = target[4]
            target_cls = onehot(target[5], num_classes)

            obj_loss = torch.pow(pred_objness - target_objness, 2)
            tx_loss = torch.pow(pred_tx - target_tx, 2)
            ty_loss = torch.pow(pred_ty - target_ty, 2)
            tw_loss = torch.pow(pred_tw - target_tw, 2)
            th_loss = torch.pow(pred_th - target_th, 2)
            cls_loss = torch.pow(pred_cls - target_cls, 2)

            box_loss = lambda_obj * (tx_loss + ty_loss + tw_loss + th_loss)
            loss = box_loss + obj_loss + cls_loss
            obj_loss_list.append(loss)

    return obj_loss_list

def onehot(class_block: torch.tensor, num_classes: int) -> torch.tensor:
    """get cls as result of onehot encoding for classes loss

        Args:
            class_block (torch.tensor) : number of classes consist of torch.tensor.
            num_classes (int) : number of classes

        Retruns:
            onehot_cls (torch.tensor) : encoded classes target tensor
    """
    cls = class_block.type(torch.LongTensor)
    onehot_cls = torch.zeros(1, num_classes).squeeze()
    onehot_cls[cls] = 1.

    return onehot_cls

def get_ious(pred: torch.tensor, target: torch.tensor, object_map: np.ndarray,
             prior_boxes: List[List[int]], x_interval: float, y_interval: float,
             input_image_width: int, input_image_height: int, num_anchor_boxes: int, anchor_channels: int) -> List[float]:
    """get iou between each anchor box with target


    Args:
        pred (torch.tensor) : result of inference
        target (torch.tensor) : label for detection consist of torch.tensor
        object_map (numpy.ndarray) : object location indices map consist of torch.tensor.
        prior_boxes (List[List[int]]) : prior boxes
        x_interval (float) : ratio between image width with prediction width
        y_interval (float) : ratio between image height with prediction height
        input_image_width (int) : input image width
        input_image_height (int) : input image height
        num_anchor_boxes (int) : number of anchor boxes
        anchor_channels (int) : number of anchor channels

    Retruns:
        ious (List[float]) : list of iou consist of List[float]
    """

    ious = list()

    for anchor_idx in range(num_anchor_boxes):

        x_idx = object_map[2][anchor_idx]
        y_idx = object_map[1][anchor_idx]
        prior_box = prior_boxes[anchor_idx]

        anchor_box = get_anchor(pred, anchor_idx, anchor_channels).numpy()

        pred_box = boxinfo_convert_xywh_stype(anchor_box, x_idx, y_idx, x_interval, y_interval,
                                              input_image_width, input_image_height, prior_box)
        target_box = boxinfo_convert_xywh_stype(target, x_idx, y_idx, x_interval, y_interval,
                                              input_image_width, input_image_height, prior_box)

        iou = get_iou(pred_box, target_box)
        ious.append(iou)

    return ious

def boxinfo_convert_xywh_stype(box: np.ndarray, x_idx: int, y_idx: int,
                               x_interval: float, y_interval: float,
                               image_width: int, image_height: int,
                               prior_box: List[int]) -> List[float]:
    """Yolo style as [tx, ty, tw, th] convert to box style as [x, y, w, h]

    Args:
        box (numpy.ndarray) : Yolo style as [tx, ty, tw, th]
        x_idx (int) : cell x coordinates
        y_idx (int) : cell y coordinates
        x_interval (float) : ratio between image width with prediction width
        y_interval (float) : ratio between image height with prediction height
        image_width (int) : input image width
        image_height (int) : input image height
        prior_box (List[int]) : prior box

    Retruns:
        box_style (List[float]) : converted boxes style as [x, y, w, h]
    """

    tx = box[1]
    ty = box[2]
    tw = box[3]
    th = box[4]

    center_x = x_interval * x_idx + int(x_interval * tx)
    center_y = y_interval * y_idx + int(y_interval * ty)
    p_w = int(prior_box[0] * tw * image_width)
    p_h = int(prior_box[1] * th * image_height)

    bx = center_x - (p_w // 2)
    by = center_y - (p_h // 2)
    bw = bx + p_w
    bh = by + p_h

    box_style = [bx, by, bw, bh]

    return box_style

def get_obj_location_index(obj_index_map: torch.tensor) -> np.ndarray:
    """get object location index

    Args:
        obj_index_map (torch.tensor) : object location indices map consist of torch.tensor.

    Retruns:
        object_indexmap_tuple (Tuple[numpy.array]) : A three-dimensional index map of whether
                                                  or not an object exists consists of a numpy array.
    """

    np_obj_index_map = obj_index_map.numpy()
    object_indexmap_tuple = np.where(np_obj_index_map == 1.)
    return object_indexmap_tuple

def get_nonobj_loss_list(pred: torch.tensor, num_anchor_boxes: int,
                         anchor_channels: int, lambda_noobj: float) -> List[torch.tensor]:
    """get loss as non-object loss

    Args:
        pred (torch.tensor) : result of inference.
                              shape of pred as [batch, channels, S, S]
        num_anchor_boxes (int) : number of anchor boxes
        anchor_channels (int) : number of anchor channels
        lambda_noobj (float) : loss weight about object not existence case

    Retruns:
        nonobj_loss_list (List[torch.tensor]) : non-object loss consist of torch.tensor inside list
    """
    nonobj_loss_list = list()
    for anchor_idx in range(num_anchor_boxes):
        anchor_box = get_anchor(pred, anchor_idx, anchor_channels)
        class_block = get_class_block(anchor_box)
        target = torch.zeros(class_block.shape)
        anchor_nonobj_cls_loss = lambda_noobj * torch.pow(class_block - target, 2)
        nonobj_loss_list.append(anchor_nonobj_cls_loss)

    return nonobj_loss_list

def get_anchor(pred: torch.tensor, idx: int, anchor_box_channels: int) -> torch.tensor:
    """get the anchor box for a specific index

    Args:
        pred (torch.tensor) : result of inference.
                              shape of pred as [batch, channels, S, S]
        idx (int) : value that indicates where located anchor boxes you want to fetch.
        anchor_box_channels (int) : channels of anchor boxes

    Retruns:
        anchor_box (torch.tensor) : detection result contain box and class information.
                                    shape of classes as [batch, [boxinfo(5) + number of classes], S, S]
    """

    start_idx = idx * (anchor_box_channels)
    anchor_box = pred[:, start_idx: (start_idx + anchor_box_channels), :, :]

    return anchor_box

def get_class_block(anchor: torch.tensor) -> torch.tensor:
    """get class tensor from each anchor box using python slicing

    Args:
        anchor (torch.tensor) : anchor box consist of torch.tensor.
                                shape of anchor as [batch, [boxinfo(5) + number of classes], S, S]

    Retruns:
        classes (torch.tensor) : number of classes consist of torch.tensor.
                                 shape of classes as [batch, number of classes, S, S]
    """

    return anchor[:, 5:, :, :]

def get_nonobj_index_map(objness: torch.tensor, num_classes: int) -> torch.tensor:
    """get non-object location map.

    get detection loss part have a two part.
    get object loss and non-object loss. It can be calculate index map as objectness map, non-objectness map
    practically First Calculate loss do not consider objness and non-objness area.
    Second result of loss multiply with objectness map or non-objectness map

    Args:
        objness (torch.tensor) : objectness map consist of torch.tensor.
        num_classes (int) : number of classes

    Retruns:
        noobj_index_map (torch.tensor) : non-object location indices map consist of torch.tensor.
                                         shape of noobj_index_map is same with result of class block as [batch, number of classes, S, S]

    # reference You Only Look Once Loss function
    # https://arxiv.org/pdf/1506.02640.pdf
    """
    non_objness = torch.neg(torch.add(objness, -1))

    basis = non_objness.view(non_objness.shape[0], 1, non_objness.shape[1], non_objness.shape[2])
    noobj_index_map = non_objness.view(non_objness.shape[0], 1, non_objness.shape[1], non_objness.shape[2])

    for i in range(num_classes - 1):
        noobj_index_map = torch.cat((noobj_index_map, basis), 1)

    return noobj_index_map

def get_obj_index_map(yhat: torch.tensor) -> torch.tensor:
    """get object location map.

    get detection loss part have a two part.
    get object loss and non-object loss. It can be calculate index map as objectness map, non-objectness map
    practically First Calculate loss do not consider objness and non-objness area.
    Second result of loss multiply with objectness map or non-objectness map

    Args:
        yhat (torch.tensor) : target information consist of torch.tensor.

    Retruns:
        obj_index_map (torch.tensor) : object location indices map consist of torch.tensor.

    # reference You Only Look Once Loss function
    # https://arxiv.org/pdf/1506.02640.pdf
    """
    obj_index_map = yhat[:, :, :, 0]

    return obj_index_map

def get_interval(input_size: int, pred_size: int) -> float:
    """get each direction interval

    Calculation of Loss of YOLO is based on grid cell
    therefore should be know that interval as each grid cell
    interval is ratio between input resolution with output resolution

    Args:
        input_size (int) : input image shape (width or height)
        pred_size (int) : output tensor shape (width or height)

    Retruns:
        interval (float) : ratio between input_size with pred_size
    """

    return input_size / pred_size

def get_pred_resolution(pred: torch.tensor) -> [int, int]:
    """get output feature map resolution

    shape of predicted torch tensor is [batch, channels, height, width]
    output feature map resolution is height & width from predicted torch tensor

    Args:
        pred (int) : feature map as result of inference

    Retruns:
        width, height [int, int] : width & height of feature map
    """

    return pred.shape[2], pred.shape[3]