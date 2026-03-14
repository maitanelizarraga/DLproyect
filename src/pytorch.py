import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score, mean_absolute_error
import optuna  # Hyperparameter optimization framework

# 1. LOAD PREPROCESSED DATA
data_folder = "data/"

try:
    # Loading features (X) and targets (y)
    X_train_np = pd.read_csv(os.path.join(data_folder, "X_train_scaled.csv")).values
    X_val_np   = pd.read_csv(os.path.join(data_folder, "X_val_scaled.csv")).values
    X_test_np  = pd.read_csv(os.path.join(data_folder, "X_test_scaled.csv")).values
    
    y_train_np = pd.read_csv(os.path.join(data_folder, "y_train.csv")).values
    y_val_np   = pd.read_csv(os.path.join(data_folder, "y_val.csv")).values
    y_test_np  = pd.read_csv(os.path.join(data_folder, "y_test.csv")).values
    
    print("Preprocessed files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure preprocessing.py was executed.")
    exit()

# 2. CONVERT TO PYTORCH TENSORS
# Using float32 for compatibility with neural network weights
X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
y_train_t = torch.tensor(y_train_np, dtype=torch.float32)
X_val_t   = torch.tensor(X_val_np, dtype=torch.float32)
y_val_t   = torch.tensor(y_val_np, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_np, dtype=torch.float32)
y_test_t  = torch.tensor(y_test_np, dtype=torch.float32)

# 3. DATALOADERS
# Implementing Mini-batch Gradient Descent 
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. HYPERPARAMETER OPTIMIZATION (OPTUNA)
# Defining the objective function to minimize Validation Loss
def objective(trial):
    # Suggesting hyperparameters: Learning Rate and Layer Capacity
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    n_neurons = trial.suggest_int("n_neurons", 32, 128)
    
    # Temporary model for the trial
    model = nn.Sequential(
        nn.Linear(X_train_t.shape[1], n_neurons),
        nn.ReLU(),
        nn.Linear(n_neurons, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Quick training for optimization phase
    for _ in range(50): 
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
    
    # Validation check
    model.eval()
    with torch.no_grad():
        v_preds = model(X_val_t.to(device))
        v_loss = criterion(v_preds, y_val_t.to(device))
    
    return v_loss.item()

print("Running Optuna trials...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30) # Executing 30 different combinations
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")

# 5. FINAL MODEL ARCHITECTURE
class InsurancePriceModel(nn.Module):
    def __init__(self, input_dim, n_neurons):
        super(InsurancePriceModel, self).__init__()
        # Using Function Composition to capture non-linearity (Chapter 4)
        self.network = nn.Sequential(
            nn.Linear(input_dim, n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

# Initialize model with Optuna's best findings
model = InsurancePriceModel(X_train_t.shape[1], best_params['n_neurons']).to(device)
criterion = nn.MSELoss() # Standard for Regression (Maximum Likelihood)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])

# 6. FINAL TRAINING LOOP
epochs = 150
train_losses, val_losses = [], []
best_val_loss = float('inf') 
model_save_path = "models/model_insurance.pth"

if not os.path.exists("models"): os.makedirs("models")

print(f"Training final model on {device}...")

for epoch in range(epochs):
    model.train()
    batch_losses = []
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Forward and Backward passes
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    
    avg_train_loss = np.mean(batch_losses)
    train_losses.append(avg_train_loss)
    
    # Validation Phase & Implicit Early Stopping
    model.eval()
    with torch.no_grad():
        v_preds = model(X_val_t.to(device))
        v_loss = criterion(v_preds, y_val_t.to(device)).item()
        val_losses.append(v_loss)

        # Saving only the best performing state
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), model_save_path)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {v_loss:.4f}")

# 7. PERFORMANCE EVALUATION
# Loading weights from the peak performance epoch
model.load_state_dict(torch.load(model_save_path))
model.eval()

with torch.no_grad():
    # Predicting on Test Set (Unseen data)
    log_predictions = model(X_test_t.to(device)).cpu().numpy()
    
    # Reversing Log Transformation for real-world USD interpretation
    real_targets = np.expm1(y_test_np)
    real_predictions = np.expm1(log_predictions)
    
    r2 = r2_score(real_targets, real_predictions)
    mae = mean_absolute_error(real_targets, real_predictions)

print("\n" + "="*40)
print("FINAL TEST EVALUATION")
print(f"R2 Score: {r2:.4f} (Accuracy: {r2*100:.2f}%)")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print("="*40)

# 8. VISUALIZATION
if not os.path.exists("visualizations"): os.makedirs("visualizations")
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training curves with Optimized Hyperparameters')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.savefig('visualizations/training_history.png')

print("Pipeline finished successfully.")