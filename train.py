import os
import json
import argparse
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch

from trainer import Trainer
from utils import Logger
from utils.dimension_cluster import DimensionCluster


def get_instance(module, name, config, *args):
    print(config[name]['type'])
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume):
    train_logger = Logger()

    # build model architecture
    model_args_options = config["arch"]["args"].keys()
    have_prior_boxes = "prior_boxes" in model_args_options
    if not have_prior_boxes:
        dimension_cluster = DimensionCluster(root=config["data_loader"]["args"]["data_dir"],
                                             display=False,
                                             distance_metric="iou")
        prior_boxes = dimension_cluster.process()
    elif have_prior_boxes:
        prior_boxes = config["arch"]["args"]["prior_boxes"]
    else:
        raise Exception("Calc prior boxes error")

    model = get_instance(module_arch, 'arch', config)
    model.build(prior_boxes)
    model.summary()
    print(model.get_output_shape())

    # setup data_loader instances
    data_loader = get_instance(module_data, 'data_loader', config)
    valid_data_loader = data_loader.split_validation()

    from torchvision import transforms
    import matplotlib.pyplot as plt
    from PIL import ImageDraw

    for batch_idx, (d, t) in enumerate(data_loader):
        c, _, _, _ = d.shape

        for idx in range(c):
            data = d[idx]
            target = t[idx]
            #print(data.shape)
            #print(data, end="\n\n\n")

            img = transforms.ToPILImage()(data)

            draw = ImageDraw.Draw(img)

            # Draw 13x13 Grid in Image
            W, H = img.size
            dx = W // 13
            dy = H // 13

            y_start = 0
            y_end = H

            for i in range(0, W, dx):
                line = ((i, y_start), (i, y_end))
                draw.line(line, fill="red")

            x_start = 0
            x_end = W
            for i in range(0, H, dy):
                line = ((x_start, i), (x_end, i))
                draw.line(line, fill="red")

            obj_coord = target[:, :, 0]
            x_shift = target[:, :, 1]
            y_shift = target[:, :, 2]
            w_ratio = target[:, :, 3]
            h_ratio = target[:, :, 4]
            cls = target[:, :, 5]

            # y
            for j in range(13):
                # x
                for i in range(13):
                    if obj_coord[j][i] == 1:

                        x_center = dx * i + int(dx * x_shift[j][i])
                        y_center = dy * j + int(dy * y_shift[j][i])
                        print("i : {}, j:{}".format(i, j))
                        width = int(w_ratio[j][i] * W)
                        height = int(h_ratio[j][i] * H)

                        xmin = x_center - (width // 2)
                        ymin = y_center - (height // 2)
                        xmax = xmin + width
                        ymax = ymin + height

                        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="blue")

                        draw.rectangle(((dx * i, dy * j), (dx * i + dx, dy * j + dy)), outline='#00ff88')
                        draw.ellipse(((x_center - 2, y_center - 2),
                                      (x_center + 2, y_center + 2)),
                                     fill='blue')
                        draw.text((dx * i, dy * j), data_loader.dataset.classes_list[int(cls[j][i])])

            plt.figure()
            plt.imshow(img)
            plt.show()

            print(target.shape)
            print(target, end="\n\n\n")
        exit()

    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]


    # build optimizer, learning rate scheduler.
    # delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)
    print("1")
    trainer = Trainer(model, loss, metrics, optimizer,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=None, #
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger)
    print("2")
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                           help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('-p', '--prior_boxes', default=None, type=str,
                        help='pre-calculated prior boxes txt file path(default: None)')
    args = parser.parse_args()
    config = None

    if args.config:
        # load config file
        config = json.load(open(args.config))
        print(config, end="\n\n")

        # TODO. what's purpose?
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint,
        # in case new config file is not given.
        # Use '--config' and '--resume' arguments together
        # to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. \
         Add '-c config.json', for example.")

    if torch.cuda.is_available():
        if args.device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        print("CUDA VISIBLE DEVICES : {}".format(os.environ["CUDA_VISIBLE_DEVICES"], end="\n\n"))

    main(config, args.resume)
