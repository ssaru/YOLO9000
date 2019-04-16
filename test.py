import os
import json
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from train import get_instance


def main(config, resume):

    model_args_options = config["arch"]["args"].keys()
    have_prior_boxes = "prior_boxes" in model_args_options
    assert have_prior_boxes is True

    prior_boxes = config["arch"]["args"]["prior_boxes"]

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=1,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=0,
        output_shape=config['data_loader']['args']['output_shape']
    )

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    model.build(prior_boxes)
    model.summary()

    # get function handles of loss and metrics
    loss_fn = get_instance(module_loss, 'loss', config, model)
    # metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # load state dict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(resume, map_location=device)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    # total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for _, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            model.module.detect(output, threshold=.3)

            exit()
            #
            # save sample images, or do something with output here
            #
            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size

            #for j, metric in enumerate(metric_fns):
            #    total_metrics[j] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}

    #log.update({met.__name__ : total_metrics[i].item() /
    #                           n_samples for i, met in enumerate(metric_fns)})
    print(log)


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

        # TODO. what's purpose?
        path = os.path.join(config['trainer']['save_dir'], config['name'])

    if args.resume:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)
