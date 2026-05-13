import os
import torch
import torch.nn as nn
import torch.optim as optim
import kagglehub
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import AudioNoiseDataset
from model import AudioTransformer
from train import save_spectrogram_image, get_splits

def main():
    print("=== INITIALIZING FINAL OPTIMIZED TRAINING ===")
    
    # ---------------------------------------------------------
    # OPTUNA WINNING HYPERPARAMETERS:
    # ---------------------------------------------------------

    BEST_LR = 0.003039903734101878
    BEST_D_MODEL = 128
    BEST_NUM_LAYERS = 3
    BEST_WEIGHT_DECAY = 5.002333980531405e-05
    NUM_EPOCHS = 50 
  
    # ---------------------------------------------------------

    MODELS_DIR = "src/Deliverable3/models_optimized"
    IMAGES_DIR = "src/Deliverable3/images_optimized"
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

   # 1. Load Data (Lighter version for CPU)
    print("Loading lightweight dataset for CPU training...")
    clean_dir = kagglehub.dataset_download("pypiahmad/librispeech-asr-corpus")
    noise_dir = kagglehub.dataset_download("mmoreaux/environmental-sound-classification-50")
    
    clean_train, clean_val, _ = get_splits(clean_dir)
    noise_train, noise_val, _ = get_splits(noise_dir)
    
    # REDUCED DATA: 8000 for training, 1000 for validation
    # This is still a "large" dataset for a CPU, but manageable
    train_dataset = AudioNoiseDataset(clean_train[:8000], noise_train[:8000])
    val_dataset = AudioNoiseDataset(clean_val[:1000], noise_val[:1000])
    
    # We use a batch size of 16 for CPU to prevent memory overflow
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 2. Initialize Model with Best Architecture
    model = AudioTransformer(num_mels=64, d_model=BEST_D_MODEL, num_layers=BEST_NUM_LAYERS).to(device)
    criterion = nn.MSELoss()
    
    # Initialize Optimizer with Best LR and Weight Decay
    optimizer = optim.AdamW(model.parameters(), lr=BEST_LR, weight_decay=BEST_WEIGHT_DECAY)
    
    # Add the plateau scheduler we discussed earlier
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 3. Final Training Loop
    print(f"\nStarting Optimized CPU Training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        # Wrap the loader in tqdm for a progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        for batch_idx, (noisy, clean) in enumerate(pbar):
            noisy, clean = noisy.to(device), clean.to(device)
            
            optimizer.zero_grad()
            predicted_clean = model(noisy)
            loss = criterion(predicted_clean, clean)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            # Update the progress bar with the current loss
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_idx, (val_noisy, val_clean) in enumerate(val_loader):
                val_noisy, val_clean = val_noisy.to(device), val_clean.to(device)
                val_predicted = model(val_noisy)
                loss = criterion(val_predicted, val_clean)
                val_loss += loss.item()
                
                if val_idx == 0:
                    save_spectrogram_image(val_noisy, val_clean, val_predicted, epoch, IMAGES_DIR)
                    
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)
        
        model_save_path = os.path.join(MODELS_DIR, f"best_transformer_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    main()