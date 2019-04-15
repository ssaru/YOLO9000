import numpy as np
import torch
import torch.nn as nn
from torchsummary.torchsummary import summary
from base import BaseModel

from model.loss import get_anchor
from model.loss import get_obj_location_index
from model.loss import boxinfo_convert_xywh_type
from model.loss import get_interval


class Yolo9000(BaseModel):
    def __init__(self,
                 num_classes=20,
                 num_prior_boxes=5,
                 prior_boxes=None,
                 device="cpu",
                 input_size=(416, 416)):
        assert isinstance(num_prior_boxes, int)

        super(Yolo9000, self).__init__()

        self.device = device
        self.input_size = input_size
        self.num_coordinates = 5
        self.num_prior_boxes = num_prior_boxes
        self.prior_boxes = None
        self.num_classes = num_classes
        self.output_shape = (self.num_coordinates + self.num_classes) * self.num_prior_boxes

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU())

        self.layer7 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, self.output_shape, kernel_size=1, stride=1, padding=0))

    def forward(self, *input):
        assert self.num_prior_boxes == len(self.prior_boxes)
        for box in self.prior_boxes:
            assert len(box) == 2
            for element in box:
                assert isinstance(element, float)
        """
        output structure
        [objness, tx, ty, tw, th, c1, ..., cn] x 5 = 125

        shape : [batch, n, n, 125]
        """
        x = input[0]
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.post_processing(out)

        return out

    def post_processing(self, out):

        for box_idx in range(self.num_prior_boxes):
            idx = box_idx * (self.num_coordinates + self.num_classes)

            # becareful!! if didn't use `.clone()` in slicing tensor
            # raise gradient in-place operation error at `.backward()`
            # t0
            out[:, idx, :, :] = nn.Sigmoid()(out[:, idx, :, :].clone())

            # bx, by
            out[:, idx + 1, :, :] = nn.Sigmoid()(out[:, idx + 1, :, :].clone())
            out[:, idx + 2, :, :] = nn.Sigmoid()(out[:, idx + 2, :, :].clone())

            # bw, bh
            out[:, idx + 3, :, :] = out[:, idx + 3, :, :].clone().exp()
            out[:, idx + 4, :, :] = out[:, idx + 4, :, :].clone().exp()

            # classes
            out[:, idx + 5:idx + 5 + self.num_classes, :, :] = \
                nn.Sigmoid()(out[:, idx + 5:idx + 5 + self.num_classes, :, :].clone())

        return out

    def detect(self, output: torch.tensor, threshold: float)-> torch.tensor:

        _, _, height_S, width_S = output.shape

        x_interval = get_interval(self.input_size[0], width_S)
        y_interval = get_interval(self.input_size[1], height_S)

        boxes = list()
        for anchor_idx in range(self.num_prior_boxes):
            anchor_channels = self.num_prior_boxes + self.num_classes
            anchor_box = get_anchor(output, anchor_idx, anchor_channels).squeeze()
            t0 = anchor_box[0, :, :]
            obj_index_map = get_obj_location_index(t0, threshold)
            len_index_map = len(obj_index_map[0])
            prior_box = self.prior_boxes[anchor_idx]

            for obj_idx in range(len_index_map):
                _x = obj_index_map[1][obj_idx]
                _y = obj_index_map[0][obj_idx]

                _obj_block = anchor_box[:, _y, _x]
                print(_obj_block.shape)
                # convert bx, by, bw, bh style
                xywh = boxinfo_convert_xywh_type(_obj_block, _x, _y, x_interval, y_interval,
                                                 self.input_size[0], self.input_size[1], prior_box, "pred")
                _cls = anchor_box[5:, _y, _x].cpu().detach().numpy()
                _max_cls_value = max(_cls)
                _cls_idx = int(np.where(_cls == _max_cls_value)[0])

                boxes.append([xywh, _cls_idx])
                print(boxes)
                exit()

                pass
        pass

    def summary(self):
        summary(self, input_size=(3, self.input_size[0], self.input_size[1]), device=self.device)

    def get_output_shape(self):
        x = torch.rand(1, 3, self.input_size[0], self.input_size[1])
        out = self.forward(x)
        b, c, w, h = out.shape

        return c, w, h

    def build(self, prior_boxes):
        self.set_prior_boxes(prior_boxes=prior_boxes)

    def set_prior_boxes(self, prior_boxes):
        self.prior_boxes = prior_boxes
