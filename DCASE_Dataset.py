from pathlib import Path
import pandas as pd

import torch
import torchaudio
from torch.utils.data.dataset import Dataset

torchaudio.set_audio_backend("sox_io")

class DCASE_Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.labels = pd.read_csv(self.root_dir / 'labels.csv')

        self.data_len = len(self.labels)

    def __getitem__(self, index):
        filename, label = self.labels.iloc[index]
        filepath = self.root_dir / 'audio' / filename
        data_array, sample_rate = torchaudio.load(filepath)
        spec = self.__make_spec__(data_array, sample_rate)

        return spec, label

    def __make_spec__(self, data_array, sample_rate):
        win_size = int(round(40 * sample_rate / 1e3))
        spec_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=win_size,
            hop_length=win_size//2,
            window_fn=torch.hamming_window,
            power=2,
            n_mels=60
        )
        spec = spec_fn(data_array)
        return spec

    def __len__(self):
        return self.data_len
