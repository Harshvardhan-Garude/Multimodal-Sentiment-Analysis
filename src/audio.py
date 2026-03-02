import torch
import torchaudio
import torchaudio.transforms as T

class AudioPreprocessor:
    def __init__(self, sample_rate=16000, duration=4.0, n_mels=64, win_length=400, hop_length=160, use_spec_augment=False):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(self.sample_rate * self.duration)
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop_length = hop_length
        self.use_spec_augment = use_spec_augment

        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_fft=self.win_length,
            power=2.0
        )
        self.amplitude_to_db = T.AmplitudeToDB()

        if self.use_spec_augment:
            # Typical SpecAugment values
            self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
            self.time_mask = T.TimeMasking(time_mask_param=35)

    def process_file(self, filepath):
        try:
            waveform, sr = torchaudio.load(filepath)
        except Exception as e:
            # Handle empty/corrupt audio: return silence
            waveform = torch.zeros(1, self.target_length)
            sr = self.sample_rate

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Pad or truncate
        num_frames = waveform.shape[1]
        if num_frames > self.target_length:
            waveform = waveform[:, :self.target_length]
        elif num_frames < self.target_length:
            pad_amount = self.target_length - num_frames
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        # Spectrogram
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)

        # SpecAugment
        if self.use_spec_augment:
            log_mel_spec = self.freq_mask(log_mel_spec)
            log_mel_spec = self.time_mask(log_mel_spec)

        return log_mel_spec
