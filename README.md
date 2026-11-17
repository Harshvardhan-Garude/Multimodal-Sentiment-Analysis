# Multimodal Sentiment Classifier

This repository contains a PyTorch sentiment classifier that trains on audio, text, or both modalities from scratch.

## Setup

Use a virtual environment and install [compatible PyTorch and TorchAudio builds](https://docs.pytorch.org/audio/stable/installation.html) for your platform. For TorchAudio releases that use TorchCodec, follow the linked decoder installation instructions and check that `torchaudio.load` can open one of your WAV files before training.

```bash
python -m pip install numpy pandas scikit-learn
```

## Data

Pass the dataset directory as `--data_dir`. Each split uses a CSV and a
directory of WAV files.

| Split | CSV | WAV directory |
| --- | --- | --- |
| Training | `train/data.csv` | `train/audio/` |
| Validation | `val/data.csv` | `val/audio/` |
| Test | `test/data.csv` | `test/audio/` |

Use `FileID` and `Text` columns. Audio files are named `<FileID>.wav`.
Include integer `Label` values 0, 1, and 2 in the training and validation CSVs
for the three-class workflow. Inference uses training text to rebuild the
vocabulary, so keep the training CSV next to the test split.

## Training

### Single Run
Train a single model (e.g., using seed 42) with both modalities, text-only, or audio-only:

```bash
# Both modalities (default)
python -m src.train --data_dir ./data --seeds 42 --modality both --epochs 30

# Text only
python -m src.train --data_dir ./data --seeds 42 --modality text --epochs 30

# Audio only
python -m src.train --data_dir ./data --seeds 42 --modality audio --epochs 30
```

### Ensemble Training
Train multiple models with different seeds to average their predictions later:

```bash
python -m src.train --data_dir ./data --seeds 42,43,44 --modality both --epochs 30
```

## Inference

Run inference and generate `test.txt` using saved checkpoints.

### Single Model Inference
```bash
python -m src.infer --data_dir ./data --ckpts checkpoints/best_model_seed_42.pt --output test.txt
```

### Ensemble Inference
Provide a comma-separated list of checkpoints to average probabilities across models:
```bash
python -m src.infer --data_dir ./data --ckpts checkpoints/best_model_seed_42.pt,checkpoints/best_model_seed_43.pt,checkpoints/best_model_seed_44.pt --output test.txt
```
