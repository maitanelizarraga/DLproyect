import os
import torch
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, Subset

def get_data_loaders(batch_size=32, target_size=(224, 224)):
    # Download the dataset
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print("Path to dataset files:", path)
    # Internal paths of the Kaggle dataset
    base_dir = os.path.join(path, 'chest_xray')
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    # Transformations definitions
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


    # CHARGE THE DATASET
    # Charge the complete training set 
    full_train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    
    # We divide the training dataset to create a validation set (1000 images)
    val_size = 1000
    train_size = len(full_train_dataset) - val_size
    
    train_subset, val_subset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

  
    # We charge the original test dataset
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    # We create the DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Summary and Verification
    print(f"\n[i] Dataset charged successfully:")
    print(f"    - Classes detected: {full_train_dataset.classes}")
    print(f"    - Training images: {len(train_subset)}")
    print(f"    - Validation images:    {len(val_subset)}")
    print(f"    - Test images:          {len(test_dataset)}")

    return (train_loader, val_loader, test_loader), full_train_dataset.classes

def imshow(img):
    """Auxiliar function to show an image from a tensor."""
    # Denormalize approximately for visualization
    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
          torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

# Execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Working with: {device}")

# Obtain loaders
loaders, classes = get_data_loaders()
train_loader, val_loader, test_loader = loaders


# dataiter = iter(train_loader)
# images, labels = next(dataiter)
# classes = full_train_dataset.classes
# print(f"Batch labels: {[classes[l] for l in labels[:4]]}")
# imshow(images[0])