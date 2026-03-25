import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score, mean_absolute_error
import optuna 

# 1. MODEL ARCHITECTURE
class InsurancePriceModel(nn.Module):
    def __init__(self, input_dim, n_neurons):
        super(InsurancePriceModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

def run_pytorch_model():
    data_folder = "data/"
    # Load data
    X_train_np = pd.read_csv(os.path.join(data_folder, "X_train_scaled.csv")).values
    X_val_np   = pd.read_csv(os.path.join(data_folder, "X_val_scaled.csv")).values
    X_test_np  = pd.read_csv(os.path.join(data_folder, "X_test_scaled.csv")).values
    y_train_np = pd.read_csv(os.path.join(data_folder, "y_train.csv")).values
    y_val_np   = pd.read_csv(os.path.join(data_folder, "y_val.csv")).values
    y_test_np  = pd.read_csv(os.path.join(data_folder, "y_test.csv")).values

    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_np, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val_np, dtype=torch.float32)
    y_val_t   = torch.tensor(y_val_np, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test_np, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test_np, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. HYPERPARAMETER OPTIMIZATION (OPTUNA)
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        n_neurons = trial.suggest_int("n_neurons", 32, 128)
        model = InsurancePriceModel(X_train_t.shape[1], n_neurons).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        for _ in range(50):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                criterion(model(X_batch), y_batch).backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            v_loss = criterion(model(X_val_t.to(device)), y_val_t.to(device))
        return v_loss.item()

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=15)
    best_params = study.best_params

    # 3. FINAL TRAINING
    model = InsurancePriceModel(X_train_t.shape[1], best_params['n_neurons']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    criterion = nn.MSELoss()
    
    epochs = 150
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    model_save_path = "models/model_insurance.pth"
    if not os.path.exists("models"): os.makedirs("models")

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        train_losses.append(np.mean(batch_losses))
        model.eval()
        with torch.no_grad():
            v_loss = criterion(model(X_val_t.to(device)), y_val_t.to(device)).item()
            val_losses.append(v_loss)
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                torch.save(model.state_dict(), model_save_path)

    # 4. FINAL EVALUATION & PLOTS
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    with torch.no_grad():
        log_predictions = model(X_test_t.to(device)).cpu().numpy()
        real_targets = np.expm1(y_test_np)
        real_predictions = np.expm1(log_predictions)
        r2 = r2_score(real_targets, real_predictions)
        mae = mean_absolute_error(real_targets, real_predictions)

    if not os.path.exists("visualizations"): os.makedirs("visualizations")
    
    # Graphic 1: Training History
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training curves with Optimized Hyperparameters')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.savefig('visualizations/training_history.png')
    plt.show()

    # Graphic 2: Predictions vs Actual
    plt.figure(figsize=(8, 8))
    plt.scatter(real_targets, real_predictions, alpha=0.5, color='blue')
    plt.plot([real_targets.min(), real_targets.max()], [real_targets.min(), real_targets.max()], 'r--', lw=2)
    plt.title('Predictions vs Actual Values (USD)')
    plt.xlabel('Actual Costs')
    plt.ylabel('Predicted Costs')
    plt.savefig('visualizations/predictions_scatter.png')
    plt.show()

    return mae, r2

if __name__ == "__main__":
    mae, r2 = run_pytorch_model()
    print(f"FINAL MAE: ${mae:.2f}")