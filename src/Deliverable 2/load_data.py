import kagglehub

# Download latest version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("Path to dataset files:", path)



import torch
import os

# Configurar el dispositivo (Crucial para el trabajo)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Trabajando con: {device}")

# Rutas basadas en tu descarga de kagglehub
base_dir = os.path.join(path, 'chest_xray')
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
val_dir = os.path.join(base_dir, 'val')


## Defining of Transforms
from torchvision import transforms

# Transformaciones para entrenamiento (con aumento de datos)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),      # Tamaño estándar para CNNs
    transforms.RandomHorizontalFlip(),  # Gira la imagen (más datos artificiales)
    transforms.RandomRotation(10),      # Rotación ligera para robustez
    transforms.ToTensor(),              # Convierte a tensores [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformaciones para validación y test (sin aumento, solo ajuste)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


## Creation of Datasets and DataLoaders
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Crear los objetos Dataset
train_dataset = ImageFolder(root=train_dir, transform=train_transform)
val_dataset = ImageFolder(root=val_dir, transform=test_transform)
test_dataset = ImageFolder(root=test_dir, transform=test_transform)

# Crear los DataLoaders
batch_size = 32 # Ajustable según memoria de GPU

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Imágenes de entrenamiento: {len(train_dataset)}")
print(f"Imágenes de validación: {len(val_dataset)}")
print(f"Imágenes de test: {len(test_dataset)}")





##Visual Verification

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5  # desnormalizar opcionalmente para ver mejor
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Obtener algunas imágenes al azar
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Mostrar imágenes y etiquetas
classes = train_dataset.classes
print(f"Etiquetas del batch: {[classes[l] for l in labels[:4]]}")
imshow(images[0]) # Muestra la primera imagen del batch