import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import kagglehub
from torch.utils.data import DataLoader
import glob
import random

# Import your custom modules
from dataset import AudioNoiseDataset
from model import AudioTransformer

def get_splits(directory, split_ratios=(0.8, 0.1, 0.1)):
    """Find all audio files and strictly splits them to prevent data leakage."""
    wav_files = glob.glob(os.path.join(directory, '**', '*.wav'), recursive=True)
    flac_files = glob.glob(os.path.join(directory, '**', '*.flac'), recursive=True)
    all_files = wav_files + flac_files
    all_files.sort()
    random.seed(1) 
    random.shuffle(all_files)
    total = len(all_files)
    return (all_files[:int(total * split_ratios[0])], 
            all_files[int(total * split_ratios[0]):int(total * split_ratios[0]) + int(total * split_ratios[1])], 
            all_files[int(total * split_ratios[0]) + int(total * split_ratios[1]):])

def save_spectrogram_image(noisy, clean, predicted, epoch, save_dir="src/Deliverable3/images"):
    """Saves a side-by-side image of the spectrograms."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Take the first item in the batch for visualization
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
    random.seed(1)
    # 1. Setup Directories
    MODELS_DIR = "src/Deliverable3/models"
    IMAGES_DIR = "src/Deliverable3/images"
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data Paths (Simplified for basic training)
    print("Loading data paths...")
    clean_dir = kagglehub.dataset_download("pypiahmad/librispeech-asr-corpus")
    noise_dir = kagglehub.dataset_download("mmoreaux/environmental-sound-classification-50")
    
    clean_train, clean_val, _ = get_splits(clean_dir)
    noise_train, noise_val, _ = get_splits(noise_dir)
    
    # Dataset size for the initial 100-epoch baseline
    train_dataset = AudioNoiseDataset(clean_train[:1000], noise_train[:1000])
    val_dataset = AudioNoiseDataset(clean_val[:200], noise_val[:200])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 3. Initialize Model, Loss, and Optimizer
    model = AudioTransformer(num_mels=64).to(device)
    criterion = nn.MSELoss() # Standard for signal reconstruction
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 4. Training Parameters and History Tracking
    NUM_EPOCHS = 100
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')
    best_model_path = os.path.join(MODELS_DIR, "best_basic_transformer.pth")

    print(f"\nStarting 100-Epoch Baseline Training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            
            optimizer.zero_grad()
            predicted_clean = model(noisy)
            loss = criterion(predicted_clean, clean)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 5. Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_idx, (val_noisy, val_clean) in enumerate(val_loader):
                val_noisy, val_clean = val_noisy.to(device), val_clean.to(device)
                val_predicted = model(val_noisy)
                loss = criterion(val_predicted, val_clean)
                val_loss += loss.item()
                
                # Save one visual comparison per epoch for the report
                if val_idx == 0:
                    save_spectrogram_image(val_noisy, val_clean, val_predicted, epoch, IMAGES_DIR)
                    
        avg_val_loss = val_loss / len(val_loader)
        
        # Track history for plotting the Learning Curve
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # 6. Save ONLY the Best Model (Model Selection)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Best model saved with Validation Loss: {best_val_loss:.4f}")

    # 7. Generate and Save Final Learning Curve Graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), history["train_loss"], label='Train Loss', color='blue')
    plt.plot(range(1, NUM_EPOCHS + 1), history["val_loss"], label='Val Loss', color='red')
    plt.title('Baseline Learning Curve: MSE Loss over 100 Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(IMAGES_DIR, "basic_learning_curve.png"))
    plt.close()
    print(f"\nTraining complete. Learning curve saved to {IMAGES_DIR}/basic_learning_curve.png")

if __name__ == "__main__":
    main()