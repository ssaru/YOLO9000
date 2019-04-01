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

    # TODO. until line working check finisehd
    from model.loss import DetectionLoss
    model.to('cpu')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    detection_loss = DetectionLoss()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to('cpu'), target.to('cpu')

        optimizer.zero_grad()
        output = model(data)
        loss = detection_loss(output, target, model, data)
        loss.backward()
        optimizer.step()
        print("Loss : {}".format(loss.item()))


    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]


    # build optimizer, learning rate scheduler.
    # delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)
    trainer = Trainer(model, loss, metrics, optimizer,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=None, #
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger)
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
