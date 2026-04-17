import os
import torch
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, Subset

def get_data_loaders(batch_size=32, target_size=(224, 224)):
    # 1. Descarga del dataset
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print("Path to dataset files:", path)

    # Rutas internas del dataset de Kaggle
    base_dir = os.path.join(path, 'chest_xray')
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    # 2. Definición de Transformaciones
    train_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Carga de Datasets
    # Cargamos el set de entrenamiento completo
    full_train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    
    # Dividimos el set de entrenamiento para crear uno de validación (1000 imágenes)
    val_size = 1000
    train_size = len(full_train_dataset) - val_size
    
    train_subset, val_subset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Cargamos el test original
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    # 4. Creación de DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 5. Resumen y Verificación
    print(f"\n[i] Dataset cargado exitosamente:")
    print(f"    - Clases detectadas: {full_train_dataset.classes}")
    print(f"    - Imágenes entrenamiento: {len(train_subset)}")
    print(f"    - Imágenes validación:    {len(val_subset)}")
    print(f"    - Imágenes test:          {len(test_dataset)}")

    return (train_loader, val_loader, test_loader), full_train_dataset.classes

def imshow(img):
    """Función auxiliar para mostrar imágenes de un Tensor"""
    # Desnormalizar aproximadamente para visualización
    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
          torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

# --- Ejecución ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Trabajando con: {device}")

# Obtener loaders
loaders, classes = get_data_loaders()
train_loader, val_loader, test_loader = loaders

# Visualizar un ejemplo
dataiter = iter(train_loader)
images, labels = next(dataiter)
print(f"Ejemplo: Esta imagen es de clase: {classes[labels[0]]}")
imshow(images[0])