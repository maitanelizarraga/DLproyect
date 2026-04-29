import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# Import everything from load_data
from load_data import *

# --- 1. IMPORT TORCHMETRICS ---
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

## Architecture (SimpleCNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Block 1: 224x224 -> 112x112
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # Block 2: 112x112 -> 56x56
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # Block 3: 56x56 -> 28x28
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 2) 

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 28 * 28) # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- 2. INITIALIZATION & HYPERPARAMETERS ---
model_scratch = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_scratch.parameters(), lr=0.001)

# LEARNING RATE SCHEDULER: Reduces LR by half (factor=0.5) if Val Loss doesn't improve for 2 epochs (patience=2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# TORCHMETRICS: Defined for 2 classes (Normal & Pneumonia)
num_classes = 2
metric_acc = MulticlassAccuracy(num_classes=num_classes).to(device)
metric_f1 = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)

epochs = 10 # Increased to 10 so the Scheduler has time to activate

# --- 3. TRAINING LOOP ---
history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

print(f"Starting Training from Scratch on: {device}...")
for epoch in range(epochs):
    model_scratch.train()
    running_loss = 0.0
    
    # Calculate gradients in batches (as requested by teacher)
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model_scratch(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    
    # --- VALIDATION PHASE ---
    model_scratch.eval()
    val_loss = 0.0
    
    # Reset metrics at the start of each validation epoch
    metric_acc.reset()
    metric_f1.reset()
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_scratch(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Update metrics batch by batch
            metric_acc.update(outputs, labels)
            metric_f1.update(outputs, labels)
    
    avg_val_loss = val_loss / len(val_loader)
    
    # Compute final metric values for the epoch
    val_acc = metric_acc.compute().item() * 100
    val_f1 = metric_f1.compute().item() * 100
    
    # UPDATE SCHEDULER based on validation loss
    scheduler.step(avg_val_loss)
    
    # Track history
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)
    
    # Get current Learning Rate to print it
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch [{epoch+1}/{epochs}] - LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.2f}%")

print("Finished Training!")

# --- 4. SAVE MODEL ---
model_save_path = "src/Deliverable 2/models/simple_cnn.pth"
# Create the foulder if not exist
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model_scratch.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# --- 5. TEST EVALUATION (ONLY ONCE AT THE END) ---
print("\n--- Running Final Test Evaluation ---")
model_scratch.eval()
test_loss = 0.0
metric_acc.reset()
metric_f1.reset()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model_scratch(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        metric_acc.update(outputs, labels)
        metric_f1.update(outputs, labels)

avg_test_loss = test_loss / len(test_loader)
test_acc = metric_acc.compute().item() * 100
test_f1 = metric_f1.compute().item() * 100

print(f"FINAL TEST METRICS -> Loss: {avg_test_loss:.4f} | Accuracy: {test_acc:.2f}% | F1-Score: {test_f1:.2f}%")

# --- 6. PLOT TRAINING HISTORY ---
plt.figure(figsize=(15, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', color='blue')
plt.plot(history['val_loss'], label='Val Loss', color='orange')
plt.title('Loss Evolution')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Metrics
plt.subplot(1, 2, 2)
plt.plot(history['val_acc'], label='Val Accuracy', color='green')
plt.plot(history['val_f1'], label='Val F1-Score', color='purple')
plt.title('Validation Metrics Evolution')
plt.xlabel('Epochs')
plt.ylabel('Percentage (%)')
plt.legend()

plt.tight_layout()
plt.show()