import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, csv_path, audio_dir, sr=16000, n_mels=80, max_duration=1.0):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.sr = sr
        self.n_mels = n_mels
        self.max_length = int(sr * max_duration)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.audio_dir, row["audio_path"])

        signal, sr = librosa.load(file_path, sr=self.sr)

        if len(signal) < self.max_length:
            signal = np.pad(signal, (0, self.max_length - len(signal)))
        else:
            signal = signal[:self.max_length]

        mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=self.n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-6)
        features = torch.tensor(log_mel_spec.T, dtype=torch.float32)

        label_id = int(row["label_id"])
        label = torch.tensor(label_id, dtype=torch.long)

        return features, label