import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from collections import Counter
from torchvision.utils import make_grid
import random
import os

# reproducibility 1.hash op, 2.numpy, 3.torch, 4.gpu
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) #reproducibility for hash op.
    np.random.seed(seed) #numpy reproducibility
    torch.manual_seed(seed) #torch reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def perform_eda(train_ds, train_loader):
    print("\n" + "="*20)
    print("--- INITIALIZING EDA ---")
    print("="*20)

    set_seed(123) 
    
   
    # CORRECTION FOR SUBSETS
    # If train_ds is a Subset, we extract the info from the original dataset
    if isinstance(train_ds, torch.utils.data.Subset):
        full_dataset = train_ds.dataset
        # We obtain the targets only from the indices that belong to the subset
        actual_targets = [full_dataset.targets[i] for i in train_ds.indices]
        class_names = full_dataset.classes
    else:
        actual_targets = train_ds.targets
        class_names = train_ds.classes

    # 1. Class analysis and distribution
    counts = Counter(actual_targets)
    
    plt.figure(figsize=(8, 5))
    sns.set_style("whitegrid")
    # We map the names of the classes for the graph
    x_labels = [class_names[i] for i in counts.keys()]
    y_values = list(counts.values())
    
    ax = sns.barplot(x=x_labels, y=y_values, palette='magma', hue=x_labels, legend=False)
    plt.title('Class Distribution (Training Set)', fontsize=14)
    plt.ylabel('Number of Images')
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.show()

    total = sum(counts.values())
    for i, count in counts.items():
        print(f" Class {class_names[i]}: {count} images ({count/total:.2%})")
    
    # 2. Class Analysis and Batch Info
    images, labels = next(iter(train_loader))
    print(f"\n--- Batch Information ---")
    print(f"Dimensions: {images.shape} (B, C, H, W)")
    print(f"Pixel Range: [{images.min():.2f}, {images.max():.2f}]")

    # 3. Average Images
    plot_class_averages(train_ds, class_names)

    # 4. Intensity Analysis (Histograms)
    plot_intensity_analysis(images, labels, class_names)

    # 5. Grid of samples
    show_random_batch(images, labels, class_names)

def plot_class_averages(dataset, class_names):
    print("\nGenerating average images per class...")
    plt.figure(figsize=(12, 6))
    

    # Subset management to find indexes per class
    is_subset = isinstance(dataset, torch.utils.data.Subset)
    
    for i, class_name in enumerate(class_names):
        # We find the indexes that correspond to this class within the subset/dataset
        if is_subset:
            # i is the class index, we look in the original dataset using the subset indices
            idx = [j for j in range(len(dataset)) if dataset.dataset.targets[dataset.indices[j]] == i][:100]
        else:
            idx = [j for j, label in enumerate(dataset.targets) if label == i][:100]
        
        if not idx: continue
        
        imgs = [dataset[j][0].numpy() for j in idx]
        avg_img = np.mean(imgs, axis=0).transpose(1, 2, 0)

        # Denormalization
        avg_img = np.clip(np.array([0.229, 0.224, 0.225]) * avg_img + np.array([0.485, 0.456, 0.406]), 0, 1)
        
        plt.subplot(1, len(class_names), i+1)
        plt.imshow(avg_img)
        plt.title(f"Mean: {class_name}")
        plt.axis('off')
    plt.suptitle("Analysis of Global Patterns (Average)", fontsize=15)
    plt.show()

def plot_intensity_analysis(images, labels, class_names):
    # We find the indexes in the actual batch labels tensor
    idx_normal = (labels == 0).nonzero(as_tuple=True)[0].tolist()
    idx_pneumonia = (labels == 1).nonzero(as_tuple=True)[0].tolist()

    selected_indices = idx_normal[:2] + idx_pneumonia[:2]
    
    if len(selected_indices) == 0: return

    fig, axes = plt.subplots(2, len(selected_indices), figsize=(16, 8))
    # Ensure axes is 2D even with few images
    if len(selected_indices) == 1: axes = axes.reshape(2, 1)

    for i, img_idx in enumerate(selected_indices):
        img = images[img_idx].numpy().transpose((1, 2, 0))
        img = np.clip(np.array([0.229, 0.224, 0.225]) * img + np.array([0.485, 0.456, 0.406]), 0, 1)
        
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Class: {class_names[labels[img_idx]]}")
        axes[0, i].axis('off')
        
        gray_img = np.mean(img, axis=2)
        axes[1, i].hist(gray_img.ravel(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[1, i].set_xlim([0, 1])
        axes[1, i].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

def show_random_batch(images, labels, class_names):
    plt.figure(figsize=(12, 8))
    num_imgs = min(16, len(images))
    img_grid = make_grid(images[:num_imgs], nrow=4)
    img_np = img_grid.numpy().transpose((1, 2, 0))
    img_np = np.clip(np.array([0.229, 0.224, 0.225]) * img_np + np.array([0.485, 0.456, 0.406]), 0, 1)
    
    plt.imshow(img_np)
    plt.title(f"Batch Sample ({num_imgs} Images)")
    plt.axis('off')
    plt.show()