import os
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
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--vocab_size', type=int, default=20000)
    parser.add_argument('--max_text_len', type=int, default=96)
    parser.add_argument('--use_spec_augment', action='store_true')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(',')]

    for seed in seeds:
        print(f"\\n{'='*40}\\nStarting training for seed {seed}\\n{'='*40}")
        set_seed(seed)
        train_seed(args, seed)

def train_seed(args, seed):
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
        use_spec_augment=args.use_spec_augment, max_text_len=args.max_text_len
    )
    val_dataset = MultimodalDataset(
        args.data_dir, tokenizer, split='val',
        use_spec_augment=False, max_text_len=args.max_text_len
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Computed class weights: {class_weights}")

    # Model
    model = MultimodalClassifier(
        vocab_size=len(tokenizer.vocab),
        modality=args.modality,
        text_encoder_type=args.text_encoder,
        num_classes=num_classes
    ).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loop(args, seed, model, train_loader, val_loader, criterion, optimizer, device)

def train_loop(args, seed, model, train_loader, val_loader, criterion, optimizer, device):
    best_val_f1 = -1
    epochs_no_improve = 0
    ckpt_path = os.path.join(args.ckpt_dir, f'best_model_seed_{seed}.pt')

    for epoch in range(1, args.epochs + 1):
        print(f"\\nEpoch {epoch}/{args.epochs}")

        # Train
        model.train()
        train_loss = 0
        train_preds, train_targets = [], []

        for batch in train_loader:
            optimizer.zero_grad()

            audio = batch['audio'].to(device)
            text = batch['text'].to(device)
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
                audio = batch['audio'].to(device)
                text = batch['text'].to(device)
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
            print(f"*** New best val macro-F1! Saved model. ***")
            print(f"Confusion Matrix:\\n{val_metrics['confusion_matrix']}")
        else:
            epochs_no_improve += 1
            print(f"No improvement. Early stopping counter: {epochs_no_improve}/{args.patience}")

        if epochs_no_improve >= args.patience:
            print("Early stopping triggered.")
            break

if __name__ == '__main__':
    main()
