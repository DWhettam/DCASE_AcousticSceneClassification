from torch.utils.data.dataset import Dataset
import torch
import torchaudio

from pathlib import Path
import pandas as pd

class DCASE(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.label_names = pd.read_csv((root_dir / 'labels.csv'))
        self.labels = self.label_names.astype('category').cat.codes

        self.sample_rate = 24000 #Confirm this

        win_size = 40 * self.ample_rate / 1e3
        self.spec_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            nfft=win_size,
            n_mels=60,
            hop_length=win_size//2,
            window_fn=torch.hamming_window,
            power=2,
        )

        self.data_len = len(self.labels)

    def __getitem__(self, index):
        filename, label = self.labels[index]
        filepath = self.root_dir / filename
        data_array, sample_rate = torchaudio.load(filepath)

        spec = self.__make_spec__(data_array, sample_rate)

        return spec, label


    def __make_spec__(self, data_array):
        spec = self.spec_fn(data_array).log2()

        return spec


    def __len__(self):
        return self.data_len
