import os
import torch
import kagglehub
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=32, target_size=(224, 224)):

    # 1. Download path
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print("Path to dataset files:", path)

    # Structure of the dataset
    base_dir = os.path.join(path, 'chest_xray')
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')

    # Normalization (standar parameters)
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    # Creation of dataset
    # ImageFolder asociates automatic folders with labels (0: NORMAL, 1: PNEUMONIA)
    
    train_ds = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(root=val_dir, transform=test_transform)
    test_ds = datasets.ImageFolder(root=test_dir, transform=test_transform)

    # Dataloader creation
    # num_workers > 0 acelera la carga en sistemas Linux/Mac. En Windows puede dar error, si ocurre, cambiar a 0.
    num_workers = 2 if os.name != 'nt' else 0 
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Dataset summary
    print(f"\n[i] Dataset cargado exitosamente:")
    print(f"    - Clases detectadas: {train_ds.classes}")
    print(f"    - Imágenes entrenamiento: {len(train_ds)}")
    print(f"    - Imágenes validación:    {len(val_ds)}")
    print(f"    - Imágenes test:          {len(test_ds)}")

    # Return tuples for easier assignment
    return (train_loader, val_loader, test_loader), (train_ds, val_ds, test_ds)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")