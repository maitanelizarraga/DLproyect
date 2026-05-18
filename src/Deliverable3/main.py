import os
import glob
import random
import kagglehub
from torch.utils.data import DataLoader
from dataset import AudioNoiseDataset

def get_splits(directory, split_ratios=(0.8, 0.1, 0.1)):
    """Finds all audio files and strictly splits them to prevent data leakage."""
    # Search for both .wav and .flac files
    wav_files = glob.glob(os.path.join(directory, '**', '*.wav'), recursive=True)
    flac_files = glob.glob(os.path.join(directory, '**', '*.flac'), recursive=True)
    
    # Combine both lists
    all_files = wav_files + flac_files
    
    # Sort them first before shuffling to ensure reproducibility across different OS
    all_files.sort()
    
    random.seed(42) # Set seed for reproducible splits
    random.shuffle(all_files)
    
    total = len(all_files)
    train_end = int(total * split_ratios[0])
    val_end = train_end + int(total * split_ratios[1])
    
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]
    
    return train_files, val_files, test_files


def main():
    print("1. Locating Datasets...")
    clean_dir = kagglehub.dataset_download("pypiahmad/librispeech-asr-corpus")
    noise_dir = kagglehub.dataset_download("mmoreaux/environmental-sound-classification-50")
    
    print("\n2. Performing Strict Train/Val/Test Splits...")
    # Split the clean speech
    clean_train, clean_val, clean_test = get_splits(clean_dir)
    # Split the background noise
    noise_train, noise_val, noise_test = get_splits(noise_dir)
    
    print(f"Train set: {len(clean_train)} speech files")
    print(f"Val set:   {len(clean_val)} speech files")
    print(f"Test set:  {len(clean_test)} speech files")

    print("\n3. Initializing Isolated Datasets...")
    train_dataset = AudioNoiseDataset(clean_train, noise_train)
    val_dataset   = AudioNoiseDataset(clean_val, noise_val)
    test_dataset  = AudioNoiseDataset(clean_test, noise_test)
    
    # Dataloaders ready for the training loop!
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False) # Don't shuffle Val/Test!
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print("SUCCESS! Data is strictly isolated without leakage.")

if __name__ == "__main__":
    main()