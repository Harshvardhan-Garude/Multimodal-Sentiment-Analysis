# Multimodal Sentiment Classifier

This repository contains a PyTorch implementation of a multimodal sentiment classifier (audio + text) that requires NO pretrained models.

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
