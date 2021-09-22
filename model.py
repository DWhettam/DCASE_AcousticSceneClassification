import torch
import torch.nn as nn
from typing import Any


class DCASEModel(nn.Module):

    def __init__(self) -> None:
        super(DCASEModel, self).__init__()

        self.layer1 = nn.sequential(nn.Conv2d(1, 128, kernel_size=(5, 5), stride=1, padding=0),
                                    nn.ReLu(),
                                    nn.MaxPool2d(5, 5),
                                    )

        self.layer2 = nn.sequential(nn.Conv2d(128, 256, kernel_size=(5, 5), stride=1, padding=0),
                                    nn.ReLu(),
                                    TimeDestructMaxPool(5),
                                    )

        self.fc_layer = nn.Sequential(nn.Linear(256, 15),
                                      nn.Softmax())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc_layer(x)
        return x


class TimeDestructMaxPool(nn.Module):

    def __init__(self, kernel_size) -> None:
        super(TimeDestructMaxPool, self).__init__()

        self.temporal_max_pool = nn.MaxPool1d(kernel_size)
        self.freq_max_pool = nn.MaxPool1d(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_max_pool(x)

        freq_dim_qtr = x.size()[2] / 4
        x = x.permute(0, 1, 3, 2)
        for idx in range(1, 4):
            dims = idx * freq_dim_qtr
            x = self.freq_max_pool(x[:, :, :, (dims - freq_dim_qtr):dims])
        x = x.permute(0, 1, 3, 2)
        return x
