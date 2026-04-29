import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import timm
from torchvision import models
from load_data import get_data_loaders
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

# ==========================================
# 1. ARCHITECTURE DEFINITIONS
# ==========================================

# --- A. SimpleCNN (Baseline) ---
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

# --- B. BestCNN (Optuna Winner) ---
class BestCNN(nn.Module):
    def __init__(self):
        super(BestCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ==========================================
# 2. EVALUATION LOGIC
# ==========================================

def evaluate_model(model, loader, device, name):
    acc_metric = MulticlassAccuracy(num_classes=2).to(device)
    f1_metric = MulticlassF1Score(num_classes=2, average='macro').to(device)
    
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            acc_metric.update(outputs, labels)
            f1_metric.update(outputs, labels)
    
    print(f"\n[+] Results for {name}:")
    print(f"    - Accuracy: {acc_metric.compute().item()*100:.2f}%")
    print(f"    - F1-Score: {f1_metric.compute().item()*100:.2f}%")

# ==========================================
# 3. MAIN COMPARISON SCRIPT
# ==========================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading libraries and checking GPU... (This might take a few seconds, please wait)")
    print(f"Using device: {device}")

    # Load data (only Test Set)
    loaders, classes = get_data_loaders(batch_size=32)
    _, _, test_loader = loaders

    print("\n" + "="*50)
    print("FINAL MODEL COMPARISON - PNEUMONIA DETECTION")
    print("="*50)

    # --- 1. EVALUATE SIMPLE CNN ---
    path_simple = "src/Deliverable 2/models/simple_cnn.pth" 
    if os.path.exists(path_simple):
        m1 = SimpleCNN().to(device)
        m1.load_state_dict(torch.load(path_simple, map_location=device))
        evaluate_model(m1, test_loader, device, "BASELINE (SimpleCNN)")
    else:
        print(f"\n[!] Missing: {path_simple}")

    # --- 2. EVALUATE OPTUNA MODEL ---
    path_optuna = "src/Deliverable 2/models/final_best_cnn.pth"
    if os.path.exists(path_optuna):
        m2 = BestCNN().to(device)
        m2.load_state_dict(torch.load(path_optuna, map_location=device))
        evaluate_model(m2, test_loader, device, "SCRATCH OPTIMIZED (Optuna)")
    else:
        print(f"\n[!] Missing: {path_optuna}")

    # --- 3. EVALUATE RESNET18 (timm / Fine-tuning) ---
    path_resnet = "src/Deliverable 2/models/best_resnet18.pth"
    if os.path.exists(path_resnet):
        m3 = timm.create_model('resnet18', pretrained=False, num_classes=2).to(device)
        m3.load_state_dict(torch.load(path_resnet, map_location=device))
        evaluate_model(m3, test_loader, device, "TRANSFER LEARNING (ResNet18 - Fine-tuning)")
    else:
        print(f"\n[!] Missing: {path_resnet}")

    # --- 4. EVALUATE VGG16 (torchvision / Frozen Layers) ---
    path_vgg = "src/Deliverable 2/models/transfer_vgg16.pth"
    if os.path.exists(path_vgg):
        m4 = models.vgg16(weights=None)
        m4.classifier[6] = nn.Linear(m4.classifier[6].in_features, 2)
        m4.load_state_dict(torch.load(path_vgg, map_location=device))
        m4 = m4.to(device)
        evaluate_model(m4, test_loader, device, "TRANSFER LEARNING (VGG16 - Frozen Layers)")
    else:
        print(f"\n[!] Missing: {path_vgg}")