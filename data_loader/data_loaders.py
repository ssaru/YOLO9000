import torch
import numpy as np
from torchvision import transforms
from base import BaseDataLoader
from utils.augmentator import Augmenter
from imgaug import augmenters as iaa
from data_loader.voc_dataset import VocDetection

class DetectionDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):

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
        super(DetectionDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    @staticmethod
    def detection_collate(batch):
        """ `Puts each data field into a tensor with outer dimension batch size`
        Args:
            batch : batch data ``batch[0]`` : image, ``batch[1]`` : label, ``batch[3]`` : size
        Return:
            image tensor, label tensor
        Future work:
            return value(torch.stack) change to Torch.FloatTensor()
        """

        targets = []
        imgs = []
        sizes = []

        for sample in batch:
            imgs.append(sample[0])

            # for drawing box
            # if using batch it should keep original image size.
            sizes.append(sample[2])

            np_label = np.zeros((7, 7, 6), dtype=np.float32)
            for object in sample[1]:
                objectness = 1
                classes = object[0]
                x_ratio = object[1]
                y_ratio = object[2]
                w_ratio = object[3]
                h_ratio = object[4]

                # can be acuqire grid (x,y) index when divide (1/S) of x_ratio
                scale_factor = (1 / 7)
                grid_x_index = int(x_ratio // scale_factor)
                grid_y_index = int(y_ratio // scale_factor)
                x_offset = (x_ratio / scale_factor) - grid_x_index
                y_offset = (y_ratio / scale_factor) - grid_y_index

                # insert object row in specific label tensor index as (x,y)
                # object row follow as
                # [objectness, class, x offset, y offset, width ratio, height ratio]
                np_label[grid_x_index][grid_y_index] = np.array(
                    [objectness, x_offset, y_offset, w_ratio, h_ratio, classes])

            label = torch.from_numpy(np_label)
            targets.append(label)

        return torch.stack(imgs, 0), torch.stack(targets, 0), sizes