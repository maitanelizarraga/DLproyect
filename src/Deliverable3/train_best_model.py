import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import kagglehub
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

# Import custom modules
from dataset import AudioNoiseDataset
from model import AudioTransformer
from train import save_spectrogram_image, get_splits

def main():

    random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
    print("INITIALIZING FINAL OPTIMIZED TRAINING")
    

    # OPTUNA WINNING HYPERPARAMETERS:
    BEST_LR = 0.00314288089084011
    BEST_D_MODEL = 128
    BEST_NUM_LAYERS = 3
    BEST_WEIGHT_DECAY = 3.752055855124284e-05
    NUM_EPOCHS = 50 

    #Best Validation Loss: 171.8909454345703
    #Best Hyperparameters:
    #lr: 0.00314288089084011
    #d_model: 128
    #num_layers: 3
    #weight_decay: 3.752055855124284e-05


    # 1. Setup Directories
    MODELS_DIR = "src/Deliverable3/models_optimized"
    IMAGES_DIR = "src/Deliverable3/images_optimized"
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data
    print("Loading lightweight dataset for optimized training...")
    clean_dir = kagglehub.dataset_download("pypiahmad/librispeech-asr-corpus")
    noise_dir = kagglehub.dataset_download("mmoreaux/environmental-sound-classification-50")
    
    clean_train, clean_val, _ = get_splits(clean_dir)
    noise_train, noise_val, _ = get_splits(noise_dir)
    
    # Using 8000 samples for robust training
    train_dataset = AudioNoiseDataset(clean_train[:8000], noise_train[:8000])
    val_dataset = AudioNoiseDataset(clean_val[:1000], noise_val[:1000])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 3. Initialize Optimized Model & Optimizer
    model = AudioTransformer(num_mels=64, d_model=BEST_D_MODEL, num_layers=BEST_NUM_LAYERS).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=BEST_LR, weight_decay=BEST_WEIGHT_DECAY)
    
    # Plateau scheduler to refine learning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 4. History Tracking for Learning Curve
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf') 
    best_model_path = os.path.join(MODELS_DIR, "best_transformer.pth")

    # 5. Final Training Loop
    print(f"\nStarting Optimized Training Sequence...")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        # Progress bar for terminal monitoring
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        for batch_idx, (noisy, clean) in enumerate(pbar):
            noisy, clean = noisy.to(device), clean.to(device)
            
            optimizer.zero_grad()
            predicted_clean = model(noisy)
            loss = criterion(predicted_clean, clean)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_idx, (val_noisy, val_clean) in enumerate(val_loader):
                val_noisy, val_clean = val_noisy.to(device), val_clean.to(device)
                val_predicted = model(val_noisy)
                loss = criterion(val_predicted, val_clean)
                val_loss += loss.item()
                
                # Visual sequence verification (Slide 24)
                if val_idx == 0:
                    save_spectrogram_image(val_noisy, val_clean, val_predicted, epoch, IMAGES_DIR)
                    
        avg_val_loss = val_loss / len(val_loader)
        
        # Record history for the final report graph
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Model Selection: Save only the version that generalizes best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Improved Model Saved (Val Loss: {best_val_loss:.4f})")
        
        scheduler.step(avg_val_loss)

    # 6. Generate and Save Optimized Learning Curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), history["train_loss"], label='Train Loss', color='blue', linewidth=2)
    plt.plot(range(1, NUM_EPOCHS + 1), history["val_loss"], label='Val Loss', color='red', linestyle='--', linewidth=2)
    plt.title('Optimized Model Learning Curve (Audio Transformer)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    curve_path = os.path.join(IMAGES_DIR, "learning_curve_optimized.png")
    plt.savefig(curve_path)
    plt.close()
    
    print(f"\nTraining Complete. Optimized learning curve saved to {curve_path}")

if __name__ == "__main__":
    main()