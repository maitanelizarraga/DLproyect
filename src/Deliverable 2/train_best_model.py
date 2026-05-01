import os
import torch
import torch.nn as nn
import torch.optim as optim
from load_data import get_data_loaders
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

# Winner Architecture based on Optuna results:
class BestCNN(nn.Module):
    def __init__(self):
        super(BestCNN, self).__init__()
        
        # Optuna's best parameters:
        # n_conv_layers: 4 | Filtros: L0=32, L1=32, L2=16, L3=64

        self.features = nn.Sequential(
            # Layer 0
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Layer 1
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Layer 2
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Layer 3
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 2)

        # Init weights with Xavier initialization for better convergence
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


# Winner Hyperparameters and configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training final model on: {device}")

# bsize_exp: 4 -> Batch Size = 16
loaders, classes = get_data_loaders(batch_size=16)
train_loader, val_loader, test_loader = loaders

model = BestCNN().to(device)
criterion = nn.CrossEntropyLoss()


# Adam optimizer with the exact LR and Beta1 from Optuna
BEST_LR = 0.00031417639113887194
BEST_BETA1 = 0.9435547457342313
optimizer = optim.Adam(model.parameters(), lr=BEST_LR, betas=(BEST_BETA1, 0.999))

# Scheduler to give it the final professional touch
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

metric_acc = MulticlassAccuracy(num_classes=2).to(device)
metric_f1 = MulticlassF1Score(num_classes=2, average='macro').to(device)

EPOCHS = 15
PATIENCE = 4

best_val_f1 = 0.0
epochs_without_improvement = 0
BEST_MODEL_PATH = "src/Deliverable 2/models/final_best_cnn.pth"

# Final training loop with early stopping based on validation F1-score, and learning rate scheduling for optimal convergence.
print("\n Initializing definite training with the best architecture...")
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
        
    # Validation
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
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.2f}%")
    
    # Early Stopping based on F1-score
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        epochs_without_improvement = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"  -> New record! Model saved to {BEST_MODEL_PATH}")
    else:
        epochs_without_improvement += 1
        print(f"  -> No improvement. ({epochs_without_improvement}/{PATIENCE})")
        
    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly Stopping activado. Fin del entrenamiento.")
        break

# Final evaluation in the Test Set
print("\n" + "="*40)
print("FINAL EVALUATION IN TEST SET")
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

print(f"Final Accuracy: {test_acc:.2f}%")
print(f"Final F1-Score: {test_f1:.2f}%")
print("="*40)