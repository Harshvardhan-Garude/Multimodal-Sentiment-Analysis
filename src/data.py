import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from .audio import AudioPreprocessor
from .text import Tokenizer

class MultimodalDataset(Dataset):
    def __init__(self, data_dir, tokenizer, split='train',
                 use_spec_augment=False, max_text_len=96,
                 audio_seconds=4.0, n_mels=64):
        self.data_dir = data_dir
        self.split = split
        self.csv_path = os.path.join(data_dir, split, 'data.csv')
        self.audio_dir = os.path.join(data_dir, split, 'audio')

        self.tokenizer = tokenizer

        # Load csv safely
        try:
            self.df = pd.read_csv(self.csv_path)
            # Ensure proper columns exist
            if 'FileID' not in self.df.columns:
                raise ValueError(f"Missing FileID in {self.csv_path}")
            if 'Text' not in self.df.columns:
                self.df['Text'] = ""
            if split != 'test' and 'Label' not in self.df.columns:
                 raise ValueError(f"Missing Label in {self.csv_path} for split {split}")

            # fill missing text with empty string
            self.df['Text'] = self.df['Text'].fillna("")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from {self.csv_path}: {e}")

        self.audio_processor = AudioPreprocessor(
            use_spec_augment=use_spec_augment,
            duration=audio_seconds,
            n_mels=n_mels
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = str(row['FileID'])
        if file_id.endswith('.wav'):
            file_id = file_id[:-4]
        text = str(row['Text'])

        # Audio
        audio_path = os.path.join(self.audio_dir, f"{file_id}.wav")
        audio_feat = self.audio_processor.process_file(audio_path)

        # Text
        text_feat = self.tokenizer.numericalize(text)

        item = {
            'file_id': file_id,
            'audio': audio_feat,
            'text': text_feat
        }

        if self.split != 'test':
            item['label'] = torch.tensor(row['Label'], dtype=torch.long)

        return item

def collate_fn(batch):
    file_ids = [item['file_id'] for item in batch]

    out = {
        'file_id': file_ids,
    }

    if 'audio' in batch[0] and batch[0]['audio'] is not None:
        out['audio'] = torch.stack([item['audio'] for item in batch])

    if 'text' in batch[0] and batch[0]['text'] is not None:
        out['text'] = torch.stack([item['text'] for item in batch])

    if 'label' in batch[0]:
        out['label'] = torch.stack([item['label'] for item in batch])

    return out
