import torch
import torch.nn as nn
from typing import Any


class DCASEModel(nn.Module):

    def __init__(self, freq_dim, time_dim) -> None:
        super(DCASEModel, self).__init__()

        freq_dim_qtr = freq_dim/ 4
        self.layer1 = nn.sequential(nn.Conv2d(1, 128, kernel_size=(5, 5), stride=1, padding=0),
                                    nn.ReLu(),
                                    nn.BatchNorm2d(128),
                                    nn.MaxPool2d((5, 5)),
                                    )

        self.layer2 = nn.sequential(nn.Conv2d(128, 256, kernel_size=(5, 5), stride=1, padding=0),
                                    nn.ReLu(),
                                    nn.BatchNorm2d(256),
                                    nn.MaxPool2d((freq_dim_qtr, time_dim)),
                                    )

        self.fc_layer = nn.Sequential(nn.Linear(256, 15),
                                      nn.Softmax())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc_layer(x)
        return x

