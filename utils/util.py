import os
import numpy as np
import torch

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def non_max_suppresssion(boxes, probs, threshold=0.5):

    '''

    :param boxes:
    :param probs:
    :param threshold:
    :return:

    1. 주어진 Box들 중 가장 높은 Score를 가진 Box 선택
    2. 선택된 Box와 나머지 Box들 간의 IOU를 계산하고 threshold 이상이면 제거(동일한 객체에 대한 검출이기 때문에 겹치는 부분이 많을 것이라고 예상됨.)
    3. 특정한 Box의 숫자가 남을 때까지 or 더 이상 선택할 Box가 없을 때 까지 위의 과정 반복
    '''

    # Init the picked box info
    pick = []

    # Box coordinate
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # Compute area of each boxes
    area = (x2 - x1) * (y2 - y1)

    # Sort
    idxs = np.argsort(probs)

    while idxs.size:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.maximum(x2[i], x2[idxs[:last]])
        yy2 = np.maximum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        iou = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(iou > threshold)[0])))

    return boxes[pick].astype("int")

def get_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 1e-6

    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return iou