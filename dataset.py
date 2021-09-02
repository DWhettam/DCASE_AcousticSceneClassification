from torch.utils.data.dataset import Dataset
import torch
import torchaudio

from pathlib import Path
import pandas as pd

from PIL import Image
import librosa
import numpy as np
import random
import pickle
import PIL
import os
import json
import torch
import torchaudio
import torchvision
import h5py
import datetime

class DCASE(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.labels = pd.read_csv((root_dir / 'labels.csv'))

        self.spec_transform =

        self.data_len = len(self.labels)

    def __getitem__(self, index):
        filename, label = self.labels[index]
        filepath = self.root_dir / filename
        data_array, sample_rate = torchaudio.load(filepath)

        spec = self.__make_spec__(data_array, sample_rate)


    def __make_spec__(self, data_array, sample_rate):
        win_size = 40 * sample_rate / 1e3
        spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            nfft=win_size,
            hop_length=win_size//2,
            window_fn=torch.hamming_window,
            power=2,

        )


    def __len__(self):
        return self.data_len
