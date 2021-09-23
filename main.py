import argparse
import time

import utils

import numpy as np
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import DCASE
from model import DCASEModel


parser = argparse.ArgumentParser(description='DCASE CNN')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Size of batches')
parser.add_argument('-p', '--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--eval_freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 1)')
args = parser.parse_args()

torchaudio.set_audio_backend("sox_io")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 15

def main():
    dataset = DCASE(args.data, 4, 30)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    _, _, freq_dim, time_dim = next(iter(dataloader))[0].size()
    model = DCASEModel(freq_dim, time_dim)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.epochs):
        run_phase(dataloader, dataset, model, criterion, optimizer, epoch, args, phase='train')

        #if epoch % args.eval_freq == 0:
            #run_phase(dataloader, dataset, model, criterion, optimizer, epoch, args, phase='val')


def run_phase(loader, dataset, model, criterion, optimizer, epoch, args, phase='train'):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(
        len(loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}] | {} | ".format(epoch, phase))

    # switch to train mode
    model.train(phase == 'train')

    end = time.time()
    for i, (specs, target) in enumerate(loader):
        data_time.update(time.time() - end)
        specs = specs.to(device)
        target = target.to(device)
        specs = specs.contiguous().view(-1, 1, np.shape(specs)[2], np.shape(specs)[3])

        output = model(specs)
        num_clips = dataset.get_num_clips()
        output = output.reshape(-1, num_clips, num_classes)
        output = torch.mean(output, 1)
        loss = criterion(output, target)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), specs.size(0))
        top1.update(acc1[0], specs.size(0))
        top5.update(acc5[0], specs.size(0))

        if phase == 'train':
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    main()