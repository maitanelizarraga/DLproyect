import os
import torch
import torch.nn as nn
import torch.optim as optim
from load_data import get_data_loaders
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

# --- 1. LA ARQUITECTURA GANADORA (HARDCODED) ---
class BestCNN(nn.Module):
    def __init__(self):
        super(BestCNN, self).__init__()
        
        # Basado en los mejores parámetros de Optuna:
        # n_conv_layers: 4 | Filtros: L0=32, L1=32, L2=16, L3=64
        self.features = nn.Sequential(
            # Capa 0
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Capa 1
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Capa 2
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Capa 3
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 2)
        
        # Inicialización Xavier descubierta por Optuna
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --- 2. CONFIGURACIÓN E HIPERPARÁMETROS GANADORES ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entrenando el modelo final en: {device}")

# bsize_exp: 4 -> Batch Size = 16
loaders, classes = get_data_loaders(batch_size=16)
train_loader, val_loader, test_loader = loaders

model = BestCNN().to(device)
criterion = nn.CrossEntropyLoss()

# Optimizador Adam con el LR y Beta1 exactos de Optuna
BEST_LR = 0.00031417639113887194
BEST_BETA1 = 0.9435547457342313
optimizer = optim.Adam(model.parameters(), lr=BEST_LR, betas=(BEST_BETA1, 0.999))

# Scheduler para darle el toque profesional final
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

metric_acc = MulticlassAccuracy(num_classes=2).to(device)
metric_f1 = MulticlassF1Score(num_classes=2, average='macro').to(device)

EPOCHS = 15
PATIENCE = 4

best_val_f1 = 0.0
epochs_without_improvement = 0
BEST_MODEL_PATH = "src/Deliverable 2/models/final_best_cnn.pth"

# --- 3. BUCLE DE ENTRENAMIENTO FINAL ---
print("\nIniciando entrenamiento definitivo con la arquitectura óptima...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    # Validación
    model.eval()
    metric_acc.reset()
    metric_f1.reset()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            metric_acc.update(outputs, labels)
            metric_f1.update(outputs, labels)
            
    val_acc = metric_acc.compute().item() * 100
    val_f1 = metric_f1.compute().item() * 100
    
    scheduler.step(val_f1)
    
    print(f"Época {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.2f}%")
    
    # Early Stopping basado en F1
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        epochs_without_improvement = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"  -> Nuevo récord! Modelo guardado en {BEST_MODEL_PATH}")
    else:
        epochs_without_improvement += 1
        print(f"  -> Sin mejora. ({epochs_without_improvement}/{PATIENCE})")
        
    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly Stopping activado. Fin del entrenamiento.")
        break

# --- 4. EVALUACIÓN FINAL EN TEST SET ---
print("\n" + "="*40)
print("--- EVALUACIÓN FINAL EN TEST SET ---")
print("="*40)

model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()
metric_acc.reset()
metric_f1.reset()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        metric_acc.update(outputs, labels)
        metric_f1.update(outputs, labels)

test_acc = metric_acc.compute().item() * 100
test_f1 = metric_f1.compute().item() * 100

print(f"Precisión (Accuracy) Final: {test_acc:.2f}%")
print(f"F1-Score Final:             {test_f1:.2f}%")
print("="*40)