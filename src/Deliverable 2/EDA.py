import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from collections import Counter
from torchvision.utils import make_grid
import random
import os

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def perform_eda(train_ds, train_loader):
    print("\n" + "="*20)
    print("--- INICIANDO EDA ---")
    print("="*20)

    set_seed(123) 
    
    # --- CORRECCIÓN PARA SUBSETS ---
    # Si train_ds es un Subset, extraemos la info del dataset original
    if isinstance(train_ds, torch.utils.data.Subset):
        full_dataset = train_ds.dataset
        # Obtenemos los targets solo de los índices que pertenecen al subset
        actual_targets = [full_dataset.targets[i] for i in train_ds.indices]
        class_names = full_dataset.classes
    else:
        actual_targets = train_ds.targets
        class_names = train_ds.classes

    # 1. Análisis de Balance de Clases
    counts = Counter(actual_targets)
    
    plt.figure(figsize=(8, 5))
    sns.set_style("whitegrid")
    # Mapeamos los nombres de las clases para el gráfico
    x_labels = [class_names[i] for i in counts.keys()]
    y_values = list(counts.values())
    
    ax = sns.barplot(x=x_labels, y=y_values, palette='magma', hue=x_labels, legend=False)
    plt.title('Distribución de Clases (Training Set)', fontsize=14)
    plt.ylabel('Número de Imágenes')
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.show()

    total = sum(counts.values())
    for i, count in counts.items():
        print(f"✅ Clase {class_names[i]}: {count} imágenes ({count/total:.2%})")
    
    # 2. Análisis de Tensores y Batch
    images, labels = next(iter(train_loader))
    print(f"\n--- Info del Batch ---")
    print(f"Dimensiones: {images.shape} (B, C, H, W)")
    print(f"Rango píxeles: [{images.min():.2f}, {images.max():.2f}]")

    # 3. Imágenes Promedio
    plot_class_averages(train_ds, class_names)

    # 4. Análisis de Intensidad
    plot_intensity_analysis(images, labels, class_names)

    # 5. Grid de muestras
    show_random_batch(images, labels, class_names)

def plot_class_averages(dataset, class_names):
    print("\nGenerando imágenes promedio por clase...")
    plt.figure(figsize=(12, 6))
    
    # Manejo de Subset para encontrar índices por clase
    is_subset = isinstance(dataset, torch.utils.data.Subset)
    
    for i, class_name in enumerate(class_names):
        # Buscamos los índices que corresponden a esta clase dentro del subset/dataset
        if is_subset:
            # i es el índice de la clase, buscamos en el dataset original usando los índices del subset
            idx = [j for j in range(len(dataset)) if dataset.dataset.targets[dataset.indices[j]] == i][:100]
        else:
            idx = [j for j, label in enumerate(dataset.targets) if label == i][:100]
        
        if not idx: continue
        
        imgs = [dataset[j][0].numpy() for j in idx]
        avg_img = np.mean(imgs, axis=0).transpose(1, 2, 0)
        
        # Desnormalización
        avg_img = np.clip(np.array([0.229, 0.224, 0.225]) * avg_img + np.array([0.485, 0.456, 0.406]), 0, 1)
        
        plt.subplot(1, len(class_names), i+1)
        plt.imshow(avg_img)
        plt.title(f"Promedio: {class_name}")
        plt.axis('off')
    plt.suptitle("Análisis de Patrones Globales (Promedio)", fontsize=15)
    plt.show()

def plot_intensity_analysis(images, labels, class_names):
    # Buscamos índices en el tensor de labels del batch actual
    idx_normal = (labels == 0).nonzero(as_tuple=True)[0].tolist()
    idx_pneumonia = (labels == 1).nonzero(as_tuple=True)[0].tolist()

    selected_indices = idx_normal[:2] + idx_pneumonia[:2]
    
    if len(selected_indices) == 0: return

    fig, axes = plt.subplots(2, len(selected_indices), figsize=(16, 8))
    # Asegurar que axes sea 2D incluso con pocas imágenes
    if len(selected_indices) == 1: axes = axes.reshape(2, 1)

    for i, img_idx in enumerate(selected_indices):
        img = images[img_idx].numpy().transpose((1, 2, 0))
        img = np.clip(np.array([0.229, 0.224, 0.225]) * img + np.array([0.485, 0.456, 0.406]), 0, 1)
        
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Clase: {class_names[labels[img_idx]]}")
        axes[0, i].axis('off')
        
        gray_img = np.mean(img, axis=2)
        axes[1, i].hist(gray_img.ravel(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[1, i].set_xlim([0, 1])
        axes[1, i].set_ylabel("Frecuencia")

    plt.tight_layout()
    plt.show()

def show_random_batch(images, labels, class_names):
    plt.figure(figsize=(12, 8))
    num_imgs = min(16, len(images))
    img_grid = make_grid(images[:num_imgs], nrow=4)
    img_np = img_grid.numpy().transpose((1, 2, 0))
    img_np = np.clip(np.array([0.229, 0.224, 0.225]) * img_np + np.array([0.485, 0.456, 0.406]), 0, 1)
    
    plt.imshow(img_np)
    plt.title(f"Muestra de Batch ({num_imgs} imágenes)")
    plt.axis('off')
    plt.show()