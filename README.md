# Multimodal Sentiment Classifier

This repository contains a PyTorch implementation of a multimodal sentiment classifier (audio + text) that requires NO pretrained models.

## Training

### 1-Epoch Sanity Check
```bash
python -m src.train --data_dir ./data --epochs 1 --min_epochs 1
```

### 60-Trial Random Sweep
```bash
python -m src.train --data_dir ./data --sweep --sweep_type random --num_trials 60
```

### 150-Trial Random Sweep
```bash
python -m src.train --data_dir ./data --sweep --sweep_type random --num_trials 150
```

### Multi-Seed Retrain (e.g. from Sweep)
You can provide `--final_seeds` when sweeping to automatically retrain the best config:
```bash
python -m src.train --data_dir ./data --sweep --num_trials 60 --final_seeds "42,43,44"
```
Or run a single configuration manually over multiple seeds:
```bash
python -m src.train --data_dir ./data --seeds "42,43,44" --lr 0.001 --dropout 0.2
```

## Inference

Run inference and generate `test.txt` using saved checkpoints.

### Single Model Inference
```bash
python -m src.infer --data_dir ./data --ckpts runs/my_run/best_model_seed_42.pt --out test.txt
```

### Ensemble Inference
Provide a comma-separated list of checkpoints to average probabilities across models:
```bash
python -m src.infer --data_dir ./data --ckpts runs/final/seed_42/best_model_seed_42.pt,runs/final/seed_43/best_model_seed_43.pt,runs/final/seed_44/best_model_seed_44.pt --out test.txt
```
