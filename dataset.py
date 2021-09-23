import random

from torch.utils.data.dataset import Dataset
import torch
import torchaudio

from pathlib import Path
import pandas as pd

class DCASE(Dataset):
    def __init__(self, root_dir, clip_duration, total_duration):
        self._root_dir = Path(root_dir)
        self._labels = pd.read_csv((self._root_dir / 'labels.csv'), names=['file', 'label'])
        self._labels['label'] = self._labels.label.astype('category').cat.codes.astype('int')
        self._clip_duration = clip_duration
        self._total_duration = total_duration

        self._sample_rate = 44100 #Confirm this

        win_size = int(round(40 * self._sample_rate / 1e3))
        self._spec_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=self._sample_rate,
            n_fft=win_size,
            n_mels=60,
            hop_length=win_size//2,
            window_fn=torch.hamming_window,
            power=2,
        )

        self._data_len = len(self._labels)

    def __getitem__(self, index):
        filename, label = self._labels.iloc[index]
        filepath = self._root_dir / 'audio'/ filename
        data_array, sample_rate = torchaudio.load(filepath)

        spec = self.__make_spec__(data_array)
        spec = self.__trim__(spec)
        return spec, label


    def __make_spec__(self, data_array):
        spec = self._spec_fn(data_array).log2()
        return spec

    def __trim__(self, spec):
        time_steps = spec.size(-1)
        self._num_clips = self._total_duration // self._clip_duration
        time_interval = int(time_steps // self._num_clips)
        all_clips = []
        for clip_idx in range(self._num_clips):
            start = clip_idx * time_interval
            end = start + time_interval
            spec_clip = spec[:, :, start:end]
            spec_clip = torch.squeeze(spec_clip)
            all_clips.append(spec_clip)

        specs = torch.stack(all_clips)
        return specs

    def get_num_clips(self):
        return self._num_clips



    def __len__(self):
        return self._data_len
