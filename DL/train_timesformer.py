import argparse
import logging
import os
import time
from collections import OrderedDict
from datetime import datetime

import joblib
import torch
import torch.nn as nn
import wandb
from dataset import DatasetFor3D
from models.timesformer import timesformer
from timm import utils
from torchvision import transforms
from utils import MetricCalculator

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')


# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training', add_help=False)


# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='resnet101_tv', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50"')
group.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
group.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')
group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='Validation batch size override (default: None)')
group.add_argument('--device', type=str, default='cuda:0', metavar='DEVICE',
                   help='CPU or GPU (default: cuda:0)')
group.add_argument("--enhance", action="store_true", default=False)

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.05)')
group.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')


# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
group.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
group.add_argument('--eval-metric', default='acc', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "acc"')
group.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
group.add_argument('-v', '--view-num', type=int, default=1)


def main(index):
    utils.setup_default_logging()
    args = parser.parse_args()
    
    wandb.init(project=args.experiment, config=args)
    
    args.rank = 0
    _logger.info('Training with a single process on 1 GPUs.')
    utils.random_seed(args.seed, args.rank)
    
    
    
    
    
    model = timesformer()
    # move model to GPU
    # is_TimeSformer = True
    model.to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # setup learning rate schedule and starting epoch
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=args.lr_min)
    # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-3, step_size_up=25, mode="triangular2", gamma=0.5)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 180], gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 15], gamma=0.3)
    
    # create the train and eval transforms
    data_transforms = {
        "train": transforms.Compose([
            transforms.CenterCrop(1080),
            transforms.Resize(448),
            transforms.ToTensor(),
        ]),
        
        "eval": transforms.Compose([
            transforms.CenterCrop(1080),
            transforms.Resize(448),
            transforms.ToTensor()
        ])
    }
    
    root = "../image_enhance" if args.enhance else "../data/image"
    print(f"Using data from {root}")
    # dataset = BaseDataset(root=root, num_view=args.view_num, transform=data_transforms["train"])
    # n_val = int(len(dataset) * 0.2)
    # n_tarin = len(dataset) - n_val
    # train_dataset, eval_dataset = random_split(dataset, [n_tarin, n_val])
    # train_dataset = CrossObjectDataSet(root=root,
    #                                    num_view=args.view_num,
    #                                    num_people=index[0],
    #                                    transform=data_transforms["train"])
    # eval_dataset = CrossObjectDataSet(root="../data/image",
    #                                  num_view=args.view_num, 
    #                                  num_people=index[1],
    #                                  transform=data_transforms["eval"])
    train_dataset = DatasetFor3D(root=root,
                              num_view=args.view_num,
                              num_people=index[0],
                              transform=data_transforms["train"])
    eval_dataset = DatasetFor3D(root=root,
                             num_view=args.view_num,
                             num_people=index[1],
                             transform=data_transforms["eval"])

    # create data loaders w/ augmentation pipeiine
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=args.batch_size, 
                                                shuffle=True,)

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False,)

    # setup loss function
    train_loss_fn = nn.CrossEntropyLoss().to(args.device)
    validate_loss_fn = nn.CrossEntropyLoss().to(args.device)

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    
    exp_name = args.experiment
    t = datetime.now().strftime("%m%d-%H%M")
    output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name + '-' + args.model, t)
    decreasing = True if eval_metric == 'loss' else False
    saver = utils.CheckpointSaver(
        model=model, optimizer=optimizer, args=args,
        checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, )
    train_metrics_calculator = MetricCalculator()
    eval_metrics_calculator = MetricCalculator()
    
    try:
        for epoch in range(args.epochs):

            train_metrics = train_one_epoch(
                epoch, model, train_dataloader, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                metrics_calculator=train_metrics_calculator)


            eval_metrics = validate(model, eval_dataloader, validate_loss_fn, args,
                                    metrics_calculator=eval_metrics_calculator)


            _lr = optimizer.param_groups[0]["lr"]
            wandb.log({'lr': _lr})
            
            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step()

            if output_dir is not None:
                utils.update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=True)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info(f'*** Best metric: {best_metric} (epoch {best_epoch})')
        
    wandb.finish()


def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args,
                    lr_scheduler=None, saver=None, output_dir=None, metrics_calculator=None):

    batch_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        input, target = input.to(args.device), target.to(args.device)

        output = model(input)
        loss = loss_fn(output, target)
        losses_m.update(loss.item(), input.size(0))

        metrics = metrics_calculator(output.to('cpu'), target.to('cpu'))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            _logger.info(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                'Metrics: {acc: .3f}   {f1: .3f}   {recall: .3f}  '
                # 'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                # '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                'LR: {lr:.3e}  '.format(
                # 'Data: {data_time.val:.3f} ({data_time.avg:.3f})\n'.format(
                    epoch,
                    batch_idx, len(loader),
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    acc=metrics.get('acc', -1),
                    f1=metrics.get('f1', -1),
                    recall=metrics.get('recall', -1),
                    # batch_time=batch_time_m,
                    # rate=input.size(0) / batch_time_m.val,
                    # rate_avg=input.size(0) / batch_time_m.avg,
                    lr=lr,
                    data_time=data_time_m))  


        
        end = time.time()
        # end for

    train_metrics = OrderedDict(**metrics)
    train_metrics["loss"] = losses_m.avg
    return train_metrics


def validate(model, loader, loss_fn, args, log_suffix='', metrics_calculator=None):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input = input.to(args.device)
            target = target.to(args.device)

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            losses_m.update(loss.data.item(), input.size(0))
            batch_time_m.update(time.time() - end)
            
            metrics = metrics_calculator(output.to('cpu'), target.to('cpu'))

            end = time.time()
            if last_batch or batch_idx % args.log_interval == 0:
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    # 'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Metrics: {acc: .3f}   {f1: .3f}   {recall: .3f} \n'.format(
                        log_name, batch_idx, last_idx,
                        loss=losses_m, acc=metrics.get('acc', -1),
                        f1=metrics.get('f1', -1), recall=metrics.get('recall', -1)))

    eval_metrics = OrderedDict(**metrics)
    eval_metrics['loss'] = losses_m.avg
    return eval_metrics


if __name__ == '__main__':
    index = joblib.load("./index.pth")
    for idx in index:
        main(idx)    