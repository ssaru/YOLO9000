import torch
import numpy as np
from torchvision import transforms
from base import BaseDataLoader
from utils.augmentator import Augmenter
from imgaug import augmenters as iaa
from data_loader.voc_dataset import VocDetection

class DetectionDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, output_shape, training=True):
        self.output_shape = output_shape

        seq = iaa.SomeOf(2, [
            iaa.Multiply((1.2, 1.5)),
            iaa.Affine(
                translate_px={"x": 3, "y": 10},
                scale=(0.9, 0.9)
            ),
            iaa.AdditiveGaussianNoise(scale=0.1 * 255),
            iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),
            iaa.Affine(rotate=(-45, 45)),
            iaa.Sharpen(alpha=0.5)
        ])
        composed = transforms.Compose([Augmenter(seq)])

        self.data_dir = data_dir
        self.dataset = VocDetection(self.data_dir, transform=composed)

        super(DetectionDataLoader, self).__init__(self.dataset,
                                                  batch_size,
                                                  shuffle,
                                                  validation_split,
                                                  num_workers,
                                                  collate_fn=self.collate_fn)

    def collate_fn(self, batch):

        targets = []
        imgs = []

        c, w, h = self.output_shape
        S = w

        for sample in batch:
            imgs.append(sample[0])

            np_label = np.zeros((w, h, 6), dtype=np.float32)
            for _object in sample[1]:
                objectness = 1.
                cls = _object[0]
                bx = _object[1]
                by = _object[2]
                bw = _object[3]
                bh = _object[4]

                # can be acuqire grid (x,y) index when divide (1/S) of x_ratio
                scale_factor = (1 / S)
                cx = int(bx // scale_factor)
                cy = int(by // scale_factor)
                sigmoid_tx = (bx / scale_factor) - cx
                sigmoid_ty = (by / scale_factor) - cy

                # insert object row in specific label tensor index as (x,y)
                # object row follow as
                # [objectness, class, x offset, y offset, width ratio, height ratio]
                np_label[cx][cy] = np.array([objectness, sigmoid_tx, sigmoid_ty, bw, bh, cls])

            label = torch.from_numpy(np_label)
            targets.append(label)

        return torch.stack(imgs, 0), torch.stack(targets, 0)
