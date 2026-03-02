import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter

from src.data import MultimodalDataset, collate_fn
from src.text import Tokenizer
from src.models import MultimodalClassifier
from src.metrics import compute_metrics

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--modality', type=str, choices=['text', 'audio', 'both'], default='both')
    parser.add_argument('--text_encoder', type=str, choices=['lstm', 'transformer'], default='lstm')
    parser.add_argument('--seeds', type=str, default='42', help='Comma separated seeds for ensembling')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--min_epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--vocab_size', type=int, default=20000)
    parser.add_argument('--max_text_len', type=int, default=96)
    parser.add_argument('--use_spec_augment', action='store_true', default=False)

    # Audio preprocessor config
    parser.add_argument('--audio_seconds', type=float, default=4.0)
    parser.add_argument('--n_mels', type=int, default=64)

    parser.add_argument('--runs_dir', type=str, default='runs')
    parser.add_argument('--run_name', type=str, default='')

    # Sweep config
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--sweep_type', type=str, choices=['random', 'grid'], default='random')
    parser.add_argument('--num_trials', type=int, default=60)
    parser.add_argument('--sweep_out', type=str, default='')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--final_seeds', type=str, default='')
    return parser.parse_args()

import json
import time
from datetime import datetime

import itertools

def run_sweep(args):
    sweep_out = args.sweep_out if args.sweep_out else os.path.join(args.runs_dir, 'sweeps', datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(sweep_out, exist_ok=True)
    results_path = os.path.join(sweep_out, 'results.csv')

    # Define search space
    space_lr = [10**np.random.uniform(-4, np.log10(2e-3)) for _ in range(1000)]
    space_wd = [10**np.random.uniform(-6, -3) for _ in range(1000)]
    space_do = [np.random.uniform(0.1, 0.5) for _ in range(1000)]

    space = {
        'batch_size': [8, 16, 32],
        'hidden_dim': [128, 256, 384],
        'use_spec_augment': [False, True],
        'max_text_len': [64, 96, 128],
        'audio_seconds': [2.0, 3.0, 4.0, 5.0],
        'n_mels': [64, 80, 96]
    }

    # Initialize results.csv
    import csv
    if not os.path.exists(results_path):
        with open(results_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['trial_id', 'val_macro_f1', 'lr', 'weight_decay', 'dropout', 'batch_size', 'hidden_dim', 'use_spec_augment', 'max_text_len', 'audio_seconds', 'n_mels'])

    results = []

    # Generate configs
    configs = []
    if args.sweep_type == 'grid':
        keys, values = zip(*space.items())
        grid_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
        random.shuffle(grid_configs)

        for i, config in enumerate(grid_configs):
            if i >= args.num_trials:
                break
            config['lr'] = random.choice(space_lr[:3]) # simplified grid
            config['weight_decay'] = random.choice(space_wd[:3])
            config['dropout'] = random.choice(space_do[:3])
            configs.append(config)
    else:
        for _ in range(args.num_trials):
            configs.append({
                'lr': random.choice(space_lr),
                'weight_decay': random.choice(space_wd),
                'dropout': random.choice(space_do),
                'batch_size': random.choice(space['batch_size']),
                'hidden_dim': random.choice(space['hidden_dim']),
                'use_spec_augment': random.choice(space['use_spec_augment']),
                'max_text_len': random.choice(space['max_text_len']),
                'audio_seconds': random.choice(space['audio_seconds']),
                'n_mels': random.choice(space['n_mels']),
            })

    for i, config in enumerate(configs):
        print(f"\\n{'='*40}\\nStarting sweep trial {i+1}/{len(configs)}\\n{'='*40}")
        trial_args = argparse.Namespace(**vars(args))
        for k, v in config.items():
            setattr(trial_args, k, v)

        trial_dir = os.path.join(sweep_out, f'trial_{i}')
        os.makedirs(trial_dir, exist_ok=True)

        with open(os.path.join(trial_dir, "config.json"), "w") as f:
            json.dump(vars(trial_args), f, indent=4)

        seed = 42 # Default single seed for sweep trials
        set_seed(seed)
        best_f1 = train_seed(trial_args, seed, trial_dir)

        result_row = {
            'trial_id': i,
            'val_macro_f1': best_f1,
            **config
        }
        results.append(result_row)

        with open(results_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([result_row[k] for k in ['trial_id', 'val_macro_f1', 'lr', 'weight_decay', 'dropout', 'batch_size', 'hidden_dim', 'use_spec_augment', 'max_text_len', 'audio_seconds', 'n_mels']])

    # Top K
    results.sort(key=lambda x: x['val_macro_f1'], reverse=True)
    topk_results = results[:args.topk]

    print(f"\\n{'='*40}\\nTop {args.topk} configs by val_macro_f1\\n{'='*40}")
    with open(os.path.join(sweep_out, 'top10.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['trial_id', 'val_macro_f1', 'lr', 'weight_decay', 'dropout', 'batch_size', 'hidden_dim', 'use_spec_augment', 'max_text_len', 'audio_seconds', 'n_mels'])
        for r in topk_results:
            print(f"Trial {r['trial_id']}: F1={r['val_macro_f1']:.4f} | {r}")
            writer.writerow([r[k] for k in ['trial_id', 'val_macro_f1', 'lr', 'weight_decay', 'dropout', 'batch_size', 'hidden_dim', 'use_spec_augment', 'max_text_len', 'audio_seconds', 'n_mels']])

    if args.final_seeds and len(topk_results) > 0:
        best_config = topk_results[0]
        final_seeds = [int(s) for s in args.final_seeds.split(',')]

        print(f"\\n{'='*40}\\nRetraining BEST config on seeds: {final_seeds}\\n{'='*40}")
        best_args = argparse.Namespace(**vars(args))
        for k in ['lr', 'weight_decay', 'dropout', 'batch_size', 'hidden_dim', 'use_spec_augment', 'max_text_len', 'audio_seconds', 'n_mels']:
            setattr(best_args, k, best_config[k])

        final_dir = os.path.join(sweep_out, 'final')
        os.makedirs(final_dir, exist_ok=True)

        final_f1s = []
        for seed in final_seeds:
            seed_dir = os.path.join(final_dir, f'seed_{seed}')
            os.makedirs(seed_dir, exist_ok=True)
            set_seed(seed)
            f1 = train_seed(best_args, seed, seed_dir)
            final_f1s.append(f1)

        mean_f1 = np.mean(final_f1s)
        std_f1 = np.std(final_f1s)
        print(f"Final Retraining Results: Mean F1 = {mean_f1:.4f} +- {std_f1:.4f}")
        with open(os.path.join(final_dir, 'final_results.txt'), 'w') as f:
            f.write(f"Mean F1 = {mean_f1:.4f} +- {std_f1:.4f}\\n")
            f.write(f"Seeds: {final_seeds}\\n")
            f.write(f"F1s: {final_f1s}\\n")


def main():
    args = parse_args()

    if args.sweep:
        run_sweep(args)
        return

    run_name = args.run_name if args.run_name else datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    seeds = [int(s) for s in args.seeds.split(',')]

    for seed in seeds:
        print(f"\\n{'='*40}\\nStarting training for seed {seed}\\n{'='*40}")
        set_seed(seed)
        train_seed(args, seed, run_dir)

def train_seed(args, seed, run_dir):
    # Setup tokenizer
    tokenizer = Tokenizer(vocab_size=args.vocab_size, max_len=args.max_text_len)

    # Load raw text to fit tokenizer
    print("Building vocabulary...")
    import pandas as pd
    try:
        train_df = pd.read_csv(os.path.join(args.data_dir, 'train', 'data.csv'))
        train_texts = train_df['Text'].fillna("").tolist()
        tokenizer.build_vocab(train_texts)
    except Exception as e:
        print(f"Warning: could not read train texts for vocab building. {e}")
        # dummy fit
        tokenizer.build_vocab(["dummy text to fit tokenizer"])

    # Datasets
    print("Loading datasets...")
    train_dataset = MultimodalDataset(
        args.data_dir, tokenizer, split='train',
        use_spec_augment=args.use_spec_augment, max_text_len=args.max_text_len,
        audio_seconds=args.audio_seconds, n_mels=args.n_mels
    )
    val_dataset = MultimodalDataset(
        args.data_dir, tokenizer, split='val',
        use_spec_augment=False, max_text_len=args.max_text_len,
        audio_seconds=args.audio_seconds, n_mels=args.n_mels
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Class weights (inverse frequency)
    train_labels = [train_dataset.df.iloc[i]['Label'] for i in range(len(train_dataset))]
    label_counts = Counter(train_labels)
    total_samples = len(train_labels)
    num_classes = len(label_counts)

    weights = [0.0] * num_classes
    for c in range(num_classes):
        weights[c] = total_samples / (num_classes * label_counts.get(c, 1))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Computed class weights: {class_weights}")

    # Model
    model = MultimodalClassifier(
        vocab_size=len(tokenizer.vocab),
        modality=args.modality,
        text_encoder_type=args.text_encoder,
        num_classes=num_classes,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim
    ).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return train_loop(args, seed, model, train_loader, val_loader, criterion, optimizer, device, run_dir)

def train_loop(args, seed, model, train_loader, val_loader, criterion, optimizer, device, run_dir):
    import csv
    best_val_f1 = -1
    epochs_no_improve = 0
    ckpt_path = os.path.join(run_dir, f'best_model_seed_{seed}.pt')
    metrics_path = os.path.join(run_dir, f'metrics_seed_{seed}.csv')

    # Initialize metrics file
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_macro_f1', 'val_per_class_f1'])

    for epoch in range(1, args.epochs + 1):
        print(f"\\nEpoch {epoch}/{args.epochs}")

        # Train
        model.train()
        train_loss = 0
        train_preds, train_targets = [], []

        for batch in train_loader:
            optimizer.zero_grad()

            audio = batch.get('audio', None)
            if audio is not None:
                audio = audio.to(device)

            text = batch.get('text', None)
            if text is not None:
                text = text.to(device)

            labels = batch['label'].to(device)

            logits = model(audio, text)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        train_metrics = compute_metrics(train_targets, train_preds)
        train_loss /= len(train_loader)

        print(f"Train | Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f} | Macro F1: {train_metrics['macro_f1']:.4f}")

        # Validate
        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                audio = batch.get('audio', None)
                if audio is not None:
                    audio = audio.to(device)

                text = batch.get('text', None)
                if text is not None:
                    text = text.to(device)

                labels = batch['label'].to(device)

                logits = model(audio, text)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                val_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_metrics = compute_metrics(val_targets, val_preds)
        val_loss /= len(val_loader)

        print(f"Val   | Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"Per-Class F1: {val_metrics['per_class_f1']}")

        with open(metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_metrics['macro_f1'], str(val_metrics['per_class_f1'])])

        # Save best model
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            epochs_no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'args': args,
                'metrics': val_metrics,
                'seed': seed
            }, ckpt_path)

            # Save confusion matrix
            np.savetxt(os.path.join(run_dir, f'best_cm_seed_{seed}.txt'), val_metrics['confusion_matrix'], fmt='%d')

            print(f"*** New best val macro-F1! Saved model. ***")
            print(f"Confusion Matrix:\\n{val_metrics['confusion_matrix']}")
        else:
            epochs_no_improve += 1
            print(f"No improvement. Early stopping counter: {epochs_no_improve}/{args.patience}")

        if epochs_no_improve >= args.patience and epoch >= args.min_epochs:
            print("Early stopping triggered.")
            break

    return best_val_f1

if __name__ == '__main__':
    main()
