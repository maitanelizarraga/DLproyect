import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
import os

# Import everything from load_data
from load_data import *

# IMPORT TORCHMETRICS
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

def get_transfer_model(num_classes=2):
    # Download the pre-trained VGG16 model
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # Freeze the pre-trained layers (Feature Extractor)
    for param in model.parameters():
        param.requires_grad = False

    # Adapt the architecture (Classifier Head)
    # we get the last/classifier layer
    num_ftrs = model.classifier[6].in_features
    
    # we adjust it to 2classes
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    return model

# INITIALIZATION & HYPERPARAMETERS
model_transfer = get_transfer_model(num_classes=2).to(device)

# Only optimize the parameters of the new classifier layer!
optimizer = optim.Adam(model_transfer.classifier[6].parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# LEARNING RATE SCHEDULER
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# TORCHMETRICS
num_classes = 2
metric_acc = MulticlassAccuracy(num_classes=num_classes).to(device)
metric_f1 = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)

epochs = 10 

# TRAINING LOOP 
history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

print(f"Starting Transfer Learning (VGG16) on: {device}...")
for epoch in range(epochs):
    model_transfer.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model_transfer(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    
    # VALIDATION PHASE 
    model_transfer.eval()
    val_loss = 0.0
    metric_acc.reset()
    metric_f1.reset()
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_transfer(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            metric_acc.update(outputs, labels)
            metric_f1.update(outputs, labels)
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = metric_acc.compute().item() * 100
    val_f1 = metric_f1.compute().item() * 100
    
    # Step the scheduler
    scheduler.step(avg_val_loss)
    
    # Track history
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)
    
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch [{epoch+1}/{epochs}] - LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.2f}%")

print("Finished Transfer Learning Training!")

# SAVE MODEL
model_save_path = "src/Deliverable 2/models/transfer_vgg16.pth"
# Create the folder if not exist 
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model_transfer.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# TEST EVALUATION
print("\n--- Running Final Test Evaluation ---")
model_transfer.eval()
test_loss = 0.0
metric_acc.reset()
metric_f1.reset()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model_transfer(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        metric_acc.update(outputs, labels)
        metric_f1.update(outputs, labels)

avg_test_loss = test_loss / len(test_loader)
test_acc = metric_acc.compute().item() * 100
test_f1 = metric_f1.compute().item() * 100

print(f"FINAL TEST METRICS -> Loss: {avg_test_loss:.4f} | Accuracy: {test_acc:.2f}% | F1-Score: {test_f1:.2f}%")

# PLOT TRAINING HISTORY (Fixed for non-interactive environments) 
plt.figure(figsize=(15, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', color='blue')
plt.plot(history['val_loss'], label='Val Loss', color='orange')
plt.title('VGG16 Transfer Learning - Loss Evolution')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Metrics
plt.subplot(1, 2, 2)
plt.plot(history['val_acc'], label='Val Accuracy', color='green')
plt.plot(history['val_f1'], label='Val F1-Score', color='purple')
plt.title('VGG16 Transfer Learning - Validation Metrics')
plt.xlabel('Epochs')
plt.ylabel('Percentage (%)')
plt.legend()

plt.tight_layout()
plt.savefig("transfer_results.png")
print("Plots saved as transfer_results.png")