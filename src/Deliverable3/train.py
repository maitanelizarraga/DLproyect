import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import kagglehub
from torch.utils.data import DataLoader
import glob
import random
from dataset import AudioNoiseDataset
from model import AudioTransformer

# We reuse the split logic we built earlier
def get_splits(directory, split_ratios=(0.8, 0.1, 0.1)):
    wav_files = glob.glob(os.path.join(directory, '**', '*.wav'), recursive=True)
    flac_files = glob.glob(os.path.join(directory, '**', '*.flac'), recursive=True)
    all_files = wav_files + flac_files
    all_files.sort()
    random.shuffle(all_files)
    total = len(all_files)
    return all_files[:int(total * split_ratios[0])], all_files[int(total * split_ratios[0]):int(total * split_ratios[0]) + int(total * split_ratios[1])], all_files[int(total * split_ratios[0]) + int(total * split_ratios[1]):]

def save_spectrogram_image(noisy, clean, predicted, epoch, save_dir="src/Deliverable3/images"):
    """Saves a side-by-side image of the spectrograms for your report."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Take the first item in the batch and remove the channel dimension
    noisy_img = noisy[0, 0].cpu().detach().numpy()
    clean_img = clean[0, 0].cpu().detach().numpy()
    pred_img = predicted[0, 0].cpu().detach().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(noisy_img, aspect='auto', origin='lower', cmap='magma')
    axes[0].set_title("Input: Noisy Audio")
    
    axes[1].imshow(clean_img, aspect='auto', origin='lower', cmap='magma')
    axes[1].set_title("Target: Clean Audio")
    
    axes[2].imshow(pred_img, aspect='auto', origin='lower', cmap='magma')
    axes[2].set_title(f"Model Prediction (Epoch {epoch})")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"spectrogram_epoch_{epoch}.png"))
    plt.close()

def main():
    random.seed(42)
    # 1. Setup Directories
    MODELS_DIR = "src/Deliverable3/models"
    IMAGES_DIR = "src/Deliverable3/images"
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data (Simplified for the training script)
    print("Loading data paths...")
    clean_dir = kagglehub.dataset_download("pypiahmad/librispeech-asr-corpus")
    noise_dir = kagglehub.dataset_download("mmoreaux/environmental-sound-classification-50")
    
    clean_train, clean_val, _ = get_splits(clean_dir)
    noise_train, noise_val, _ = get_splits(noise_dir)
    
    # For the real full training we remove the [:1000] for making it smaller and faster
    train_dataset = AudioNoiseDataset(clean_train[:1000], noise_train[:1000])
    val_dataset = AudioNoiseDataset(clean_val[:200], noise_val[:200])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 3. Initialize Model, Loss, and Optimizer
    model = AudioTransformer(num_mels=64).to(device)
    criterion = nn.MSELoss() # Mean Squared Error is standard for comparing spectrograms
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 4. Training Loop
    NUM_EPOCHS = 5
    
    print("\nStarting Training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (noisy, clean) in enumerate(train_loader):
            noisy, clean = noisy.to(device), clean.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predicted_clean = model(noisy)
            
            # Calculate loss and backpropagate
            loss = criterion(predicted_clean, clean)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 5. Validation Step (No gradients needed)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_idx, (val_noisy, val_clean) in enumerate(val_loader):
                val_noisy, val_clean = val_noisy.to(device), val_clean.to(device)
                val_predicted = model(val_noisy)
                loss = criterion(val_predicted, val_clean)
                val_loss += loss.item()
                
                # Save ONE image comparison per epoch (using the first batch of validation)
                if val_idx == 0:
                    save_spectrogram_image(val_noisy, val_clean, val_predicted, epoch, IMAGES_DIR)
                    
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # 6. Save the Model Checkpoint
        model_save_path = os.path.join(MODELS_DIR, f"transformer_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved model to {model_save_path} and image to {IMAGES_DIR}\n")

if __name__ == "__main__":
    main()