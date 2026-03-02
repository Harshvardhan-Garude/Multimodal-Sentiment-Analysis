import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioEncoder(nn.Module):
    def __init__(self, emb_dim=256, dropout=0.2):
        super().__init__()
        # Input shape: (B, 1, 64, T)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, emb_dim)

    def forward(self, x):
        # x: (B, 1, 64, T)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Global Average Pooling result: (B, 256, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class TextEncoderLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden_dim=256, num_layers=2, dropout=0.2, out_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # Attention over time
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x):
        # x: (B, L)
        embedded = self.embedding(x)  # (B, L, E)
        output, _ = self.lstm(embedded)  # (B, L, H*2)

        # Attention pooling
        attn_weights = self.attention(output).squeeze(-1)  # (B, L)
        # Mask out padding (assuming padding is index 0)
        mask = (x == 0)
        attn_weights = attn_weights.masked_fill(mask, -1e9)
        attn_weights = F.softmax(attn_weights, dim=-1)  # (B, L)

        # Weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), output).squeeze(1)  # (B, H*2)
        out = self.fc(context)  # (B, out_dim)
        return out


class TextEncoderTransformer(nn.Module):
    def __init__(self, vocab_size, max_len=96, d_model=256, n_layers=4, n_heads=4, ff_dim=1024, dropout=0.2, out_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, out_dim)

        self.max_len = max_len

    def forward(self, x):
        # x: (B, L)
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand_as(x)

        emb = self.embedding(x) + self.pos_embedding(positions)  # (B, L, d_model)

        # Padding mask
        src_key_padding_mask = (x == 0)

        output = self.transformer(emb, src_key_padding_mask=src_key_padding_mask)  # (B, L, d_model)

        # [CLS] pooling - assuming [CLS] is the first token (index 0 in the sequence)
        cls_output = output[:, 0, :]  # (B, d_model)
        out = self.fc(cls_output)
        return out

class MultimodalClassifier(nn.Module):
    def __init__(self, vocab_size, modality='both', text_encoder_type='lstm',
                 audio_dim=256, text_dim=512, num_classes=3, dropout=0.3, hidden_dim=256):
        super().__init__()
        self.modality = modality
        self.text_encoder_type = text_encoder_type

        # Adjust dimensions based on hidden_dim
        audio_dim = hidden_dim
        text_dim = hidden_dim * 2

        self.audio_encoder = None
        self.text_encoder = None

        if modality in ['audio', 'both']:
            self.audio_encoder = AudioEncoder(emb_dim=audio_dim, dropout=dropout)

        if modality in ['text', 'both']:
            if text_encoder_type == 'lstm':
                self.text_encoder = TextEncoderLSTM(vocab_size=vocab_size, hidden_dim=hidden_dim, dropout=dropout, out_dim=text_dim)
            elif text_encoder_type == 'transformer':
                self.text_encoder = TextEncoderTransformer(vocab_size=vocab_size, d_model=hidden_dim, dropout=dropout, out_dim=text_dim)
            else:
                raise ValueError(f"Unknown text_encoder_type: {text_encoder_type}")

        # Fusion
        if modality == 'both':
            self.gated_fusion = nn.Linear(text_dim, audio_dim)
            input_dim = text_dim + audio_dim
        elif modality == 'audio':
            input_dim = audio_dim
        elif modality == 'text':
            input_dim = text_dim

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes)
        )

    def forward(self, audio, text):
        audio_emb, text_emb = None, None

        if self.modality in ['audio', 'both']:
            # Audio input: (B, 64, T) -> add channel dimension
            if audio.dim() == 3:
                audio = audio.unsqueeze(1)
            audio_emb = self.audio_encoder(audio)

        if self.modality in ['text', 'both']:
            text_emb = self.text_encoder(text)

        # Fusion and Classification
        if self.modality == 'both':
            # Gated fusion
            g = torch.sigmoid(self.gated_fusion(text_emb))  # (B, audio_dim)
            audio_gated = g * audio_emb                     # (B, audio_dim)
            fused = torch.cat([text_emb, audio_gated], dim=-1) # (B, text_dim + audio_dim)
            logits = self.classifier(fused)
        elif self.modality == 'audio':
            logits = self.classifier(audio_emb)
        elif self.modality == 'text':
            logits = self.classifier(text_emb)

        return logits
