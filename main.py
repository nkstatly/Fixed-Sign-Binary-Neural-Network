import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import data
import resnet

from torch.autograd import Variable
from datetime import datetime
from scalenet import make_model
from utils import *
from torchsummary import summary


parser = argparse.ArgumentParser(description='PyTorch ScaleNet Training')

parser.add_argument('--is_train', action='store_true',
                    help='If training')
parser.add_argument('--test_code', action='store_true',
                    help='Test code')
parser.add_argument('--no_flip', action='store_true',
                    help='No flip')
parser.add_argument('--dir_data', default='./data/CIFAR',
                    help='dataset directory')
parser.add_argument('--n_threads', type=int, default=10,
                    help='number of threads for data loading')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                    help='dataset name or folder')
parser.add_argument('--model', metavar='MODEL', default='resnet',
                    help='model name')
parser.add_argument('--depth', default=18, type=int, metavar='N',
                    help='model depth')
parser.add_argument('--input_bits', default=32, type=int, metavar='N',
                    help='input bits')
parser.add_argument('--scale_bits', default=16, type=int, metavar='N',
                    help='input bits')
parser.add_argument('--activation', default='relu', type=str, metavar='ACT',
                    help='activation function used')
parser.add_argument('--origin', action='store_true',
                    help='If origin')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')

def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, (inputs, target) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            target = target.cuda(async=True)
        input_var = Variable(inputs.type(args.type), volatile=not training)
        target_var = Variable(target)

        # Compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        if type(output) is list:
            output = output[0]

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if training:
            # Compute gradient and do SGD step
            # We can comment out the two loops below, since we need not train the sign.
            optimizer.zero_grad()
            loss.backward()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.data.copy_(p.org)
            optimizer.step()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.org.copy_(p.data.clamp_(-1,1))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer):
    # Switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, epoch):
    # Switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None)

if __name__ == '__main__':
    global args, best_prec1
    best_prec1 = 0.
    args = parser.parse_args()
    
    save_fold_name = [
        args.model, str(args.depth) if args.model=='resnet' else args.vgg_type,
        args.dataset, 'BS%d'%args.batch_size,
        'IBit%d'%args.input_bits, 'SBit%d'%args.scale_bits,
        args.activation, args.optimizer
    ]
    if args.origin:
        save_fold_name = save_fold_name[:4]
        save_fold_name.append('Origin')
        
    save_fold_name.append(datetime.now().strftime('%Y-%m-%d_%H-%M'))
    save_fold_name = '_'.join(save_fold_name)    
    save_path = os.path.join(args.results_dir, save_fold_name)
    os.makedirs(save_path)
    
    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')
    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)
    
    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None
    
    logging.info("creating model %s", args.model)
    if args.origin:
        if args.model == 'resnet':
            model = resnet.ResNet(args)
    else:
        model = make_model(args)
    shutil.copyfile('./bnn_dict.pth', os.path.join(save_path, 'bnn_dict.pth'))
    
    criterion = nn.CrossEntropyLoss()
    criterion.type(args.type)
    model.type(args.type)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    if args.test_code:
        if args.dataset == 'cifar10':
            summary(model,(3,32,32))
        elif args.dataset == 'imagenet':
            summary(model,(3,224,224))
        raise Exception(1)
    
    if args.dataset == 'cifar10':
        loader_train, loader_test = data.get_loader_cifar10(args)
        regime = {
                0: {'optimizer': args.optimizer, 'lr': 1e-1,
                    'weight_decay': 1e-4, 'momentum': 0.9},
                120: {'lr': 1e-2},
                240: {'lr': 1e-3, 'weight_decay': 0},
                400: {'lr': 1e-4}
         } 
    elif args.dataset == 'imagenet':
        loader_train, loader_test = data.get_loader_imagenet('/root/datas/', batch_size=args.batch_size)
        regime = {
                0: {'optimizer': args.optimizer, 'lr': 1e-1,
                    'weight_decay': 1e-4, 'momentum': 0.9},
                40: {'lr': 1e-2},
                70: {'lr': 1e-3, 'weight_decay': 0}, 
                100: {'lr': 1e-4} 
         } 
    
    for epoch in range(args.start_epoch, args.epochs):
        optimizer = adjust_optimizer(optimizer, epoch, regime)
                
        train_loss, train_prec1, train_prec5 = train(
            loader_train, model, criterion, epoch, optimizer)
        
        val_loss, val_prec1, val_prec5 = validate(
            loader_test, model, criterion, epoch)
    
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'regime': regime
        }, is_best, path=save_path)
        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5))
        results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                    train_error1=100 - train_prec1, val_error1=100 - val_prec1,
                    train_error5=100 - train_prec5, val_error5=100 - val_prec5)
        results.save()
