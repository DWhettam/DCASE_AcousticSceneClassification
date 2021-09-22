import random

from torch.utils.data.dataset import Dataset
import torch
import torchaudio

from pathlib import Path
import pandas as pd

class DCASE(Dataset):
    def __init__(self, root_dir, clip_duration, total_duration):
        self.root_dir = Path(root_dir)
        self.label_names = pd.read_csv((root_dir / 'labels.csv'))
        self.labels = self.label_names.astype('category').cat.codes
        self.clip_duration = clip_duration
        self.total_duration = total_duration

        self.sample_rate = 44100 #Confirm this

        win_size = 40 * self.sample_rate / 1e3
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
        spec = self.__trim__(spec)

        return spec, label


    def __make_spec__(self, data_array):
        spec = self.spec_fn(data_array).log2()
        return spec

    def __trim__(self, spec):
        time_steps = spec.size(-1)
        self.num_clips = self.total_duration / self.clip_duration
        time_interval = time_steps // self.num_clips

        all_clips = []
        for clip_idx in range(self.num_clips):
            start = clip_idx * time_interval
            end = start + time_interval
            spec_clip = spec[start:end]
            all_clips.append(spec_clip)

        specs = torch.stack(all_clips)
        return specs

    def get_num_clips(self):
        return self.num_clips







    def __len__(self):
        return self.data_len
