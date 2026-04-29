import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
# ACTUALIZADO: Usamos JournalFileBackend para evitar el warning
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend 
from torchmetrics.classification import MulticlassF1Score
import os

from load_data import get_data_loaders, device

# --- 1. ARQUITECTURA DINÁMICA ---
class DynamicCNN(nn.Module):
    def __init__(self, trial):
        super(DynamicCNN, self).__init__()
        self.layers = nn.ModuleList()
        
        n_layers = trial.suggest_int("n_conv_layers", 2, 4)
        in_channels = 3
        
        for i in range(n_layers):
            out_channels = trial.suggest_categorical(f"n_filters_l{i}", [16, 32, 64])
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, 2)
        
        init_type = trial.suggest_categorical("weight_init", ["kaiming", "xavier"])
        self.apply(lambda m: self._init_weights(m, init_type))

    def _init_weights(self, m, init_type):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight)
            else:
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x

# --- 2. FUNCIÓN OBJETIVO DE OPTUNA ---
def objective(trial):
    n = trial.suggest_int("bsize_exp", 4, 6) 
    batch_size = 2 ** n
    trial.set_user_attr("bsize", batch_size) 
    
    # Redirigir la salida estándar temporalmente para no saturar la consola con la descarga de Kaggle
    loaders, _ = get_data_loaders(batch_size=batch_size)
    train_loader, val_loader, _ = loaders

    model = DynamicCNN(trial).to(device)
    
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    
    if optimizer_name == "Adam":
        beta1 = trial.suggest_float("beta1", 0.85, 0.99)
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))
    else:
        momentum = trial.suggest_float("momentum", 0.5, 0.99)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    criterion = nn.CrossEntropyLoss()
    metric_f1 = MulticlassF1Score(num_classes=2, average='macro').to(device)
    
    epochs = 5 
    
    print(f"\\n---> Iniciando Trial {trial.number} | Batch Size: {batch_size} | Opt: {optimizer_name} | LR: {lr:.5f}")
    
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        model.eval()
        metric_f1.reset()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                metric_f1.update(outputs, labels)
        
        val_f1 = metric_f1.compute().item()
        
        # CHIVATO DE PROGRESO: Ahora verás cómo avanza
        print(f"    Trial {trial.number} - Epoch [{epoch+1}/{epochs}] - Val F1: {val_f1*100:.2f}%")
        
        trial.report(val_f1, epoch)
        if trial.should_prune():
            print(f"    [!] Trial {trial.number} podado (pruned) por bajo rendimiento.")
            raise optuna.exceptions.TrialPruned()

    return val_f1

# --- 3. CONFIGURACIÓN DEL STUDY ---
if __name__ == "__main__":
    print("Iniciando optimización con Optuna...")
    
    # ACTUALIZADO: Código sin warnings
    storage = JournalStorage(JournalFileBackend("optuna_journal.log"))
    
    sampler = TPESampler(n_startup_trials=5, seed=42)
    pruner = MedianPruner(n_warmup_steps=2, n_startup_trials=5)
    
    study = optuna.create_study(
        study_name="cnn_optimization",
        direction="maximize", 
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=15)
    
    print("\\n--- ¡Optimización completada! ---")
    print("Mejores hiperparámetros encontrados:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")