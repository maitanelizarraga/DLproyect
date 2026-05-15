import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import kagglehub
from dataset import AudioNoiseDataset
from model import AudioTransformer
from main import get_splits 
from optuna.samplers import TPESampler

def objective(trial):
    """
    Optuna will run this function multiple times.
    Every time, 'trial.suggest_...' will pick new random/optimized values!
    """
    # 1. We let Optuna choose the Hyperparameters
    # Searches for learning rates between 1e-5 and 1e-2 on a log scale
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    
    # It will try different Transformer sizes
    d_model = trial.suggest_categorical("d_model", [128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 2, 6)
    
    # It will try different weight decay values for the optimizer
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # 2. Setup Device and Data 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    clean_dir = kagglehub.dataset_download("pypiahmad/librispeech-asr-corpus")
    noise_dir = kagglehub.dataset_download("mmoreaux/environmental-sound-classification-50")
    
    clean_train, clean_val, _ = get_splits(clean_dir)
    noise_train, noise_val, _ = get_splits(noise_dir)
    
    # We use a small subset of data for tuning so it runs fast
    train_dataset = AudioNoiseDataset(clean_train[:50], noise_train[:50])
    val_dataset = AudioNoiseDataset(clean_val[:10], noise_val[:10])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 3. Initialize Model with Optuna's chosen parameters
    model = AudioTransformer(num_mels=64, d_model=d_model, num_layers=num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 4. Training Loop for Tuning
    NUM_TUNE_EPOCHS = 50
    
    for epoch in range(NUM_TUNE_EPOCHS):
        model.train()
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            predicted = model(noisy)
            loss = criterion(predicted, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_noisy, val_clean in val_loader:
                val_noisy, val_clean = val_noisy.to(device), val_clean.to(device)
                val_predicted = model(val_noisy)
                loss = criterion(val_predicted, val_clean)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Report the loss to Optuna so it knows if these parameters were good
        trial.report(avg_val_loss, epoch)
        
        # Optuna feature: "Pruning" (Stop early if the parameters are clearly terrible)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_val_loss # Optuna will try to MINIMIZE this returned value

if __name__ == "__main__":
    print("Starting Automated Hyperparameter Optimization...")
    
    # Create an Optuna 'Study'
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    
    # Run 30 different combinations
    study.optimize(objective, n_trials=30)
    
    print("\nTUNING COMPLETE")
    print(f"Best Validation Loss: {study.best_value}")
    print("Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

# Optuna hyperparameters results:
#TUNING COMPLETE
#Best Validation Loss: 171.8909454345703
#Best Hyperparameters:
  #lr: 0.00314288089084011
  #d_model: 128
  #num_layers: 3
  #weight_decay: 3.752055855124284e-05