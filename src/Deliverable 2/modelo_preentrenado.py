import os
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from load_data import get_data_loaders

# --- CONFIGURATION ---
MODEL_NAME = 'resnet18'
EPOCHS = 20           
PATIENCE = 3          
# Ruta directa al archivo para que solo guarde uno y lo sobrescriba
SAVE_DIR = "src/Deliverable 2/models"
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(SAVE_DIR, f"best_{MODEL_NAME}.pth")

# Get the loaders
loaders, classes = get_data_loaders()
train_loader, val_loader, test_loader = loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Model creation with pretrained weights
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(classes))
model = model.to(device)

# Loss and optimizators
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Variables for early stopping
best_val_acc = 0.0
epochs_without_improvement = 0

# Training loop
print(f"\nInitializing training with: {MODEL_NAME}...")

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
    val_correct = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            val_correct += (pred == labels).sum().item()
            total_val += labels.size(0)
            
    val_acc = 100 * val_correct / total_val
    print(f"Época {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f} - Val Acc: {val_acc:.2f}%")

    # --- Lógica de Checkpoint (solo el mejor) y Early Stopping ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_without_improvement = 0
        
        # Guardamos solo el state_dict para que sea más ligero y sobrescribimos el mismo archivo
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"¡Nueva mejor precisión! Modelo actualizado en: {BEST_MODEL_PATH}")
    else:
        epochs_without_improvement += 1
        print(f"Sin mejora. Contador Early Stopping: {epochs_without_improvement}/{PATIENCE}")

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping activado tras {PATIENCE} épocas sin mejora.")
        break

# --- Evaluación Final ---
print("\nCargando la mejor versión del modelo para evaluación final...")
# Cargamos los pesos del archivo único
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.to(device)
model.eval()

test_correct = 0
total_test = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, pred = torch.max(outputs, 1)
        test_correct += (pred == labels).sum().item()
        total_test += labels.size(0)

print(f"\nFINAL RESULTS:")
print(f"Accuracy on Test with the BEST model: {100 * test_correct / total_test:.2f}%")