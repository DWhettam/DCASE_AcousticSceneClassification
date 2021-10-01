import argparse
import time
from pathlib import Path

import wandb
import numpy as np
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

import utils
from dataset import DCASE
from model import DCASEModel


parser = argparse.ArgumentParser(description='DCASE CNN')
parser.add_argument('data', type=str,
                    help='path to dataset')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Size of batches')
parser.add_argument('-p', '--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--eval_freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 1)')
args = parser.parse_args()

wandb.init(project="DCASE-CNN")

torchaudio.set_audio_backend("sox_io")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 15

def main():
    dataset = DCASE(args.data, 3)
    length = len(dataset)
    train_len = int(round(length * 0.8))
    val_len = length - train_len
    train_data, val_data = torch.utils.data.random_split(dataset, [train_len, val_len])
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = DCASEModel()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.epochs):
        run_phase(train_dataloader, dataset, model, criterion, optimizer, epoch, args, phase='train')

        if epoch % args.eval_freq == 0:
            run_phase(val_dataloader, dataset, model, criterion, optimizer, epoch, args, phase='val')


def run_phase(loader, dataset, model, criterion, optimizer, epoch, args, phase='train'):
    """
    Runs a full epoch of train or val phase
    :param loader: Pytorch DataLoader. Can be train or val
    :param dataset: DCASE Dataset. Used for getting the number of sub-clips within each audio clip
    :param model: Model object
    :param criterion: Criterion for calculating loss
    :param optimizer: Optimiser used for gradient updates
    :param epoch: number of current epoch
    :param args: arguments input by user
    :param phase: train/val phase
    :return: N/A
    """
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(
        len(loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}] | {} | ".format(epoch, phase))

    # Initialize the prediction and label lists(tensors)
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')

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

        # Append batch prediction results
        predlist = torch.cat([predlist, torch.argmax(output, dim=1).cpu()])
        lbllist = torch.cat([lbllist, target.cpu()])

        if i % args.print_freq == 0:
            progress.display(i)

        for meter in progress.meters:
            wandb.log({str(phase + '-' + meter.name): meter.val})

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    conf_mat = confusion_matrix(lbllist.detach().numpy(), predlist.detach().numpy())
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp.plot()
    plt.savefig(f"{phase}_cm.png")
    plt.close()

    accuracy = accuracy_score(lbllist.detach().numpy(), predlist.detach().numpy())
    print(f"{phase} accuracy of epoch {epoch}: {accuracy}")




if __name__ == '__main__':
    main()
