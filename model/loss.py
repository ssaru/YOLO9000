import torch
import numpy as np

from typing import List
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


    def forward(self, pred, y_hat, device):

        width_S, height_S = get_pred_resolution(pred)

        batch_size = pred.shape[0]
        x_interval = get_interval(self.image_width, width_S)
        y_interval = get_interval(self.image_height, height_S)

        obj_index_map = get_obj_index_map(y_hat)
        nonobj_index_map = get_nonobj_index_map(obj_index_map, self.num_classes)

        nonobj_loss = get_nonobj_loss(pred, nonobj_index_map, self.num_prior_boxes,
                                      self.num_of_anchorbox_elem, self._lambda_nonobj, device)

        obj_loss = get_obj_loss(pred, y_hat, obj_index_map, self.prior_boxes,
                                      x_interval, y_interval, self.image_width, self.image_height,
                                      self.num_classes, self.num_prior_boxes,
                                      self.num_of_anchorbox_elem, self._lambda_obj, self._lambda_nonobj, device)

        total_loss = (obj_loss + nonobj_loss) / batch_size

        return total_loss


def get_obj_loss(pred: torch.tensor, target: torch.tensor, obj_index_map: torch.tensor,
                 prior_boxes: List[List[int]], x_interval: float, y_interval: float, input_image_width: int,
                 input_image_height: int, num_classes: int, num_anchor_boxes: int, anchor_channels: int,
                 lambda_obj: float, lambda_nonobj: float, device: str) -> List[torch.tensor]:
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
        lambda_noobj (float) : loss weight about object not existence case
        device (str) : train device

    Retruns:
        obj_loss_list (List[torch.tensor]) : list of obj loss

    Test: v
    """


    obj_loss_list = list(torch.zeros(1))
    nonobj_loss_list = list(torch.zeros(1))

    object_map = get_obj_location_index(obj_index_map)
    num_of_obj = len(object_map[0])

    for idx in range(num_of_obj):
        # shape of pred is [batch, channels, height, width]
        # shape of target is [batch, width, height, channels]
        # should be consider that
        pred_on_obj = pred[object_map[0][idx], :, object_map[2][idx], object_map[1][idx]]
        gt_on_obj = target[object_map[0][idx], object_map[1][idx], object_map[2][idx], :]
        specific_object_map = [object_map[1][idx], object_map[2][idx]]

        ious = get_ious(pred_on_obj, gt_on_obj, specific_object_map, prior_boxes,
                        x_interval, y_interval, input_image_width, input_image_height, num_anchor_boxes,
                        anchor_channels)

        max_iou = max(ious)
        max_iou_idx = ious.index(max_iou)

        if max_iou >= .5:
            anchor_box = get_anchor(pred_on_obj, max_iou_idx, anchor_channels)

            pred_objness = anchor_box[0] * max_iou
            pred_tx = anchor_box[1]
            pred_ty = anchor_box[2]
            pred_tw = anchor_box[3]
            pred_th = anchor_box[4]
            pred_cls = anchor_box[5:]

            target_objness = gt_on_obj[0]
            target_tx = gt_on_obj[1]
            target_ty = gt_on_obj[2]
            target_tw = gt_on_obj[3]
            target_th = gt_on_obj[4]
            target_cls = onehot(gt_on_obj[5], num_classes)

            obj_loss = torch.sum(torch.pow(pred_objness - target_objness, 2))
            tx_loss = lambda_obj * torch.sum(torch.pow(pred_tx - target_tx, 2))
            ty_loss = lambda_obj * torch.sum(torch.pow(pred_ty - target_ty, 2))
            tw_loss = lambda_obj * torch.sum(torch.pow(pred_tw - target_tw, 2))
            th_loss = lambda_obj * torch.sum(torch.pow(pred_th - target_th, 2))
            cls_loss = lambda_obj * torch.sum(torch.pow(pred_cls - target_cls, 2))

            loss = obj_loss + tx_loss + ty_loss + tw_loss + th_loss + cls_loss
            obj_loss_list.append(loss)

        # already calculated non-obj loss
        # part of max iou less than .5 case should be calculate like a non-obj
        else:

            for idx in range(num_anchor_boxes):
                anchor_box = get_anchor(pred_on_obj, idx, anchor_channels)
                class_block = get_class_block(anchor_box)
                cls_target = torch.zeros(class_block.shape).to(device)
                anchor_cls_loss = torch.sum(lambda_nonobj * torch.pow(class_block - cls_target, 2))
                nonobj_loss_list.append(anchor_cls_loss)

    nonobj_losses = torch.stack(nonobj_loss_list)
    obj_losses = torch.stack(obj_loss_list)

    return torch.sum(obj_losses) + torch.sum(nonobj_losses)

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
    onehot_cls.requires_grad = False
    onehot_cls[cls] = 1.

    return onehot_cls

def get_ious(pred: torch.tensor, target: torch.tensor, object_map: np.ndarray,
             prior_boxes: List[List[int]], x_interval: float, y_interval: float,
             input_image_width: int, input_image_height: int, num_anchor_boxes: int, anchor_channels: int) -> List[float]:
    """get iou between each anchor box with target


    Args:
        pred (torch.tensor) : result of inference
        target (torch.tensor) : label for detection consist of torch.tensor
        object_map (list) : object location indices map. shape is [height, width]
                            it already has been chosen specific channels
        prior_boxes (List[List[int]]) : prior boxes
        x_interval (float) : ratio between image width with prediction width
        y_interval (float) : ratio between image height with prediction height
        input_image_width (int) : input image width
        input_image_height (int) : input image height
        num_anchor_boxes (int) : number of anchor boxes
        anchor_channels (int) : number of anchor channels

    Retruns:
        ious (List[float]) : list of iou consist of List[float]

    Test: v
    """

    ious = list()
    for anchor_idx in range(num_anchor_boxes):

        x_idx = object_map[1]
        y_idx = object_map[0]
        prior_box = prior_boxes[anchor_idx]

        anchor_box = get_anchor(pred, anchor_idx, anchor_channels).detach().numpy()
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

    Test: v
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
    Test: v
    """

    np_obj_index_map = obj_index_map.cpu().detach().numpy()
    object_indexmap_tuple = np.where(np_obj_index_map == 1.)

    return object_indexmap_tuple

def get_nonobj_loss(pred: torch.tensor, nonobj_index_map: torch.tensor, num_anchor_boxes: int,
                         anchor_channels: int, lambda_noobj: float, device: str) -> List[torch.tensor]:
    """get loss as non-object loss

    Args:
        pred (torch.tensor) : result of inference.
                              shape of pred as [batch, channels, S, S]
        nonobj_index_map (torch.tensor) : non-object location indices map consist of torch.tensor.
                                          shape of noobj_index_map is same with result of class block
                                          as [batch, number of classes, S, S]
        num_anchor_boxes (int) : number of anchor boxes
        anchor_channels (int) : number of anchor channels
        lambda_noobj (float) : loss weight about object not existence case
        device (str) : train device

    Retruns:
        nonobj_loss (List[torch.tensor]) : non-object loss consist of torch.tensor

    Test: v
    """
    nonobj_loss_list = list()

    for anchor_idx in range(num_anchor_boxes):
        anchor_box = get_anchor(pred, anchor_idx, anchor_channels)
        class_block = get_class_block(anchor_box)
        target = torch.zeros(class_block.shape).to(device)
        anchor_cls_loss = lambda_noobj * torch.pow(class_block - target, 2)
        anchor_nonobj_cls_loss = torch.sum(anchor_cls_loss * nonobj_index_map)
        nonobj_loss_list.append(anchor_nonobj_cls_loss)

    nonobj_losses = torch.stack(nonobj_loss_list)

    return torch.sum(nonobj_losses)

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

    Test: v
    """
    shape = len(pred.shape)

    start_idx = idx * (anchor_box_channels)
    end_idx = (start_idx + anchor_box_channels)

    if shape == 1:
        anchor_box = pred[start_idx: end_idx]
    elif shape == 4:
        anchor_box = pred[:, start_idx: end_idx, :, :]
    else:
        raise Exception("shape of input parameter `pred` wrong. It should be 1 or 4")

    return anchor_box

def get_class_block(anchor: torch.tensor) -> torch.tensor:
    """get class tensor from each anchor box using python slicing

    Args:
        anchor (torch.tensor) : anchor box consist of torch.tensor.
                                shape of anchor as [batch, [boxinfo(5) + number of classes], S, S]

    Retruns:
        classes (torch.tensor) : number of classes consist of torch.tensor.
                                 shape of classes as [batch, number of classes, S, S]

    Test: v
    """
    shape = len(anchor.shape)

    if shape == 1:
        return anchor[5:]
    elif shape == 4:
        return anchor[:, 5:, :, :]
    else:
        raise Exception("shape of input parameter `pred` wrong. It should be 1 or 4")


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

    Test: v
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

    Test: v
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

    Test: v
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

    Test: v
    """

    return pred.shape[2], pred.shape[3]