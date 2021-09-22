import argparse
import time

import utils

import torchaudio
from torch.utils.data import DataLoader

from DCASE_Dataset import DCASE_Dataset


parser = argparse.ArgumentParser(description='DCASE CNN')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('epochs', type=int, default=10,
                    help='number of epochs')
args = parser.parse_args()

torchaudio.set_audio_backend("sox_io")

def main():
    dataset = DCASE_Dataset(args.data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in args.epochs:
        train(dataloader)

def train(loader, model, criterion, optimizer, epoch, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(
        len(loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(dataloader):
        data_time.update(time.time() - end)


if __name__ == '__main__':
    main()