import os
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from load_data import get_data_loaders

# CONFIGURATION 
MODEL_NAME = 'resnet18'
EPOCHS = 20           
PATIENCE = 3          
# Direct route to save the best model (only one file, it will be overwritten if a better model is found)
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
    
    # Validation phase
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
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f} - Val Acc: {val_acc:.2f}%")

    # Checkpoint Logic (only the best) and Early Stopping 
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_without_improvement = 0
        
        # We save the model weights to the specified path (overwriting if it's better)
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"New better precision! Model updated at: {BEST_MODEL_PATH}")
    else:
        epochs_without_improvement += 1
        print(f"No improvement. Counter Early Stopping: {epochs_without_improvement}/{PATIENCE}")

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping activated after {PATIENCE} epochs without improvement.")
        break

# Final evaluation
print("\nCharging the best model for final evaluation on the Test Set...")
# Charging the weights of the best model found during training
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