from data_loader.voc_dataset import VocDetection
from utils.label_visualizer import visualize_detection_label
from utils.augmentator import Augmenter
import torchvision.transforms as transforms
from imgaug import augmenters as iaa
from data_loader.data_loaders import DetectionDataLoader

import torch


def main():
    root = "/home/martin/Documents/dev/Deepbaksu_vision/_Datasets/VOC2012"

    # Dataset class demo
    voc = VocDetection(root)
    for i in range(1):
        image, target = voc.__getitem__(i)
        #print(image, target)
        print(target)
        #visualize_detection_label(image, target, voc.classes_list, (13, 13))

    train_loader = DetectionDataLoader(data_dir=root,
                                       batch_size=1,
                                       shuffle=True,
                                       output_shape=[125, 13, 13],
                                       validation_split=0.1,
                                       num_workers=1)

    for (images, labels) in train_loader:
        # print(images)
        print("label shape : {}".format(labels.shape))
        print("objness : {}".format(labels[0, :, :, 0]))
        print("cx : {}".format(labels[0, :, :, 1]))
        print("cy : {}".format(labels[0, :, :, 2]))
        print("cw : {}".format(labels[0, :, :, 3]))
        print("ch : {}".format(labels[0, :, :, 4]))
        print("cls : {}".format(labels[0, :, :, 5]))
        exit()


    # Augmentation Demo
    seq = iaa.SomeOf(2, [
                iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
                iaa.Affine(
                    translate_px={"x": 3, "y": 10},
                    scale=(0.9, 0.9)
                ),  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
                iaa.AdditiveGaussianNoise(scale=0.1 * 255),
                iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),
                iaa.Affine(rotate=(-45, 40)),
                iaa.Sharpen(alpha=0.5)
    ])

    composed = transforms.Compose([Augmenter(seq)])

    voc = VocDetection(root, transform=composed)
    for i in range(10):
        image, target = voc.__getitem__(i)
        print(image, target)
        visualize_detection_label(image, target, voc.classes_list, (13, 13))

if __name__ == "__main__":
    main()
