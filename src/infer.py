import os
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import MultimodalDataset, collate_fn
from src.text import Tokenizer
from src.models import MultimodalClassifier

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--ckpts', type=str, required=True, help='Comma separated list of checkpoint paths')
    parser.add_argument('--output', type=str, default='test.txt', help='Output predictions file')
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()

def load_model(ckpt_path, device, vocab_size):
    print(f"Loading checkpoint {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt['args']

    # number of classes assumed to be 3 based on instructions
    model = MultimodalClassifier(
        vocab_size=vocab_size,
        modality=args.modality,
        text_encoder_type=args.text_encoder,
        num_classes=3
    )

    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model, args

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_paths = args.ckpts.split(',')

    # We need to build the vocabulary exactly as during training.
    # The simplest way is to load the training data again.
    print("Building vocabulary from training data to match tokenizer...")
    tokenizer = Tokenizer()
    try:
        train_df = pd.read_csv(os.path.join(args.data_dir, 'train', 'data.csv'))
        train_texts = train_df['Text'].fillna("").tolist()
        tokenizer.build_vocab(train_texts)
    except Exception as e:
        print(f"Failed to load training data for vocab: {e}")
        # Build dummy vocab to avoid crashing
        tokenizer.build_vocab(["dummy"])

    vocab_size = len(tokenizer.vocab)

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = MultimodalDataset(
        args.data_dir, tokenizer, split='test',
        use_spec_augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    models = []
    for ckpt_path in ckpt_paths:
        model, ckpt_args = load_model(ckpt_path, device, vocab_size)
        models.append(model)

    print("Running inference...")
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            audio = batch['audio'].to(device)
            text = batch['text'].to(device)

            # Aggregate probabilities across all models
            batch_probs = []
            for model in models:
                logits = model(audio, text)
                probs = F.softmax(logits, dim=-1)
                batch_probs.append(probs)

            # Average probabilities
            avg_probs = torch.stack(batch_probs).mean(dim=0)
            preds = torch.argmax(avg_probs, dim=-1).cpu().numpy()
            all_preds.extend(preds)

    # Write predictions to output file
    # Ensure they are in exact order
    print(f"Writing {len(all_preds)} predictions to {args.output}...")
    with open(args.output, 'w') as f:
        for p in all_preds:
            f.write(f"{p}\\n")

    print("Inference complete!")

if __name__ == '__main__':
    main()
