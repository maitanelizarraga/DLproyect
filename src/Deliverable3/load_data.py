import kagglehub
import librosa
import soundfile as sf
import os

# 1. Download Datasets
librispeech_path = kagglehub.dataset_download("pypiahmad/librispeech-asr-corpus")
esc50_path = kagglehub.dataset_download("mmoreaux/environmental-sound-classification-50")

# 2. Setup Parameters
TARGET_SR = 16000  # Standard for speech processing
DURATION = 3.0     # 3 seconds for sequence consistency

def load_and_format(file_path):
    # Load using Librosa
    # sr=TARGET_SR ensures resampling 
    # mono=True ensures 1 channel
    audio, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
    
    # Ensure consistent sequence length for the DL model
    target_length = TARGET_SR * DURATION
    if len(audio) > target_length:
        audio = audio[:int(target_length)]
    else:
        audio = librosa.util.fix_length(audio, size=int(target_length))
    return audio

# Aclaration:
# We will iterate through the folders 'librispeech_path' and 'esc50_path' in order to create the clean/noisy pairs.