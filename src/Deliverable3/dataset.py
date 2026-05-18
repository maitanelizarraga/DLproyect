import os
import random
import torch
import torchaudio
import numpy as np
import glob  # <--- Add this import
from torch.utils.data import Dataset

from load_data import load_and_format, TARGET_SR

class AudioNoiseDataset(Dataset):
    # Changed from taking directories to taking explicit lists of files
    def __init__(self, clean_files, noise_files, snr_db=5.0): 
        self.clean_files = clean_files
        self.noise_files = noise_files
        self.snr_db = snr_db
        
        # Check if lists are empty
        if len(self.clean_files) == 0 or len(self.noise_files) == 0:
            print("WARNING: Empty file list passed to Dataset!")
            
        # Audio Transformations (From Slides: STFT -> Mel Scale -> Log Scale)
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SR,
            n_fft=1024,          
            hop_length=512,      
            n_mels=64            
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()


    def __len__(self):
        return len(self.clean_files)

    def mix_audio(self, clean_audio, noise_audio):
        """Applies Additive Noise at a specific SNR (Slide 15)."""
        # Calculate power
        p_clean = np.mean(clean_audio ** 2)
        p_noise = np.mean(noise_audio ** 2)
        
        # Avoid division by zero
        if p_noise == 0: 
            return clean_audio
            
        # Calculate scalar to achieve target SNR
        scalar = np.sqrt(p_clean / (10**(self.snr_db/10) * p_noise))
        noisy_audio = clean_audio + (scalar * noise_audio)
        
        # Prevent clipping (normalize between -1 and 1)
        max_val = np.max(np.abs(noisy_audio))
        if max_val > 1.0:
            noisy_audio = noisy_audio / max_val
            
        return noisy_audio

    def get_features(self, waveform):
        """Transforms 1D wave to 2D Log-Mel Spectrogram Sequence (Slide 24)."""
        # Convert numpy array back to torch tensor for torchaudio
        waveform_tensor = torch.FloatTensor(waveform).unsqueeze(0) 
        
        mel_spec = self.mel_spectrogram(waveform_tensor)
        log_mel_spec = self.amplitude_to_db(mel_spec)
        return log_mel_spec

    def __getitem__(self, idx):
        # 1. Load clean speech
        clean_path = self.clean_files[idx]
        clean_audio = load_and_format(clean_path)
        
        # 2. Load random noise
        noise_path = random.choice(self.noise_files)
        noise_audio = load_and_format(noise_path)
        
        # 3. Apply Additive Noise 
        noisy_audio = self.mix_audio(clean_audio, noise_audio)
        
        # 4. Extract Sequence Features (Log-Mel Spectrograms)
        clean_features = self.get_features(clean_audio)
        noisy_features = self.get_features(noisy_audio)
        
        # Returns: Input (Noisy Sequence), Target (Clean Sequence)
        return noisy_features, clean_features