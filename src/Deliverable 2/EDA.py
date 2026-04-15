import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from collections import Counter
from torchvision.utils import make_grid

def perform_eda(train_ds, train_loader):
    print("--- EDA ---")
    
    # 1. Análisis de Balance de Clases
    counts = Counter(train_ds.targets)
    class_names = train_ds.classes
    
    plt.figure(figsize=(8, 5))
    sns.set_style("whitegrid")
    ax = sns.barplot(x=[class_names[i] for i in counts.keys()], 
                     y=list(counts.values()), 
                     palette='magma', hue=[class_names[i] for i in counts.keys()], legend=False)
    
    plt.title('Distribución de Clases (Training Set)', fontsize=14)
    plt.ylabel('Número de Imágenes')
    
    # Añadir etiquetas de cantidad sobre las barras
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

    # 3. Visualización de "Imágenes Promedio" por Clase
    # Esto ayuda a ver si hay sesgos (ej. si una clase siempre tiene marcas de texto)
    plot_class_averages(train_ds, class_names)

    # 4. Análisis de Intensidad y Brillo
    plot_intensity_analysis(images, labels, class_names)

    # 5. Grid de muestras aleatorias
    show_random_batch(images, labels, class_names)

def plot_class_averages(dataset, class_names):
    """Calcula y muestra la imagen promedio para cada categoría."""
    print("\nGenerando imágenes promedio por clase...")
    plt.figure(figsize=(12, 6))
    
    for i, class_name in enumerate(class_names):
        # Filtrar imágenes de la clase actual (limitamos a 100 para velocidad)
        idx = [j for j, label in enumerate(dataset.targets) if label == i][:100]
        imgs = [dataset[j][0].numpy() for j in idx]
        
        avg_img = np.mean(imgs, axis=0).transpose(1, 2, 0)
        
        # Desnormalizar (suponiendo ImageNet stats)
        avg_img = np.clip(np.array([0.229, 0.224, 0.225]) * avg_img + np.array([0.485, 0.456, 0.406]), 0, 1)
        
        plt.subplot(1, len(class_names), i+1)
        plt.imshow(avg_img)
        plt.title(f"Promedio: {class_name}")
        plt.axis('off')
    plt.suptitle("Análisis de Patrones Globales por Clase", fontsize=15)
    plt.show()

def plot_intensity_analysis(images, labels, class_names, n_samples=4):
    """Analiza la distribución de frecuencias de color/gris."""
    fig, axes = plt.subplots(2, n_samples, figsize=(16, 8))
    
    for i in range(min(n_samples, len(images))):
        img = images[i].numpy().transpose((1, 2, 0))
        img = np.clip(np.array([0.229, 0.224, 0.225]) * img + np.array([0.485, 0.456, 0.406]), 0, 1)
        
        # Imagen Original
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Label: {class_names[labels[i]]}")
        axes[0, i].axis('off')
        
        # Histograma
        # Convertimos a escala de grises para el histograma si es RGB
        gray_img = np.mean(img, axis=2)
        axes[1, i].hist(gray_img.ravel(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[1, i].set_title("Distribución de Intensidad")
        axes[1, i].set_xlim([0, 1])

    plt.tight_layout()
    plt.show()

def show_random_batch(images, labels, class_names):
    """Muestra un grid de imágenes del loader actual."""
    plt.figure(figsize=(12, 8))
    # Desnormalizar el batch para mostrar
    img_grid = make_grid(images[:16], nrow=4)
    img_np = img_grid.numpy().transpose((1, 2, 0))
    img_np = np.clip(np.array([0.229, 0.224, 0.225]) * img_np + np.array([0.485, 0.456, 0.406]), 0, 1)
    
    plt.imshow(img_np)
    plt.title("Muestra del Data Loader (Primeras 16 imágenes)")
    plt.axis('off')
    plt.show()