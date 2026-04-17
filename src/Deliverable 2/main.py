from load_data import get_data_loaders
# Asegúrate de que perform_eda esté preparado para recibir (dataset, loader) o (clases, loader)
from EDA import perform_eda 

def main():
    # 1. Obtenemos los loaders y la lista de clases
    # Nota: Ajustamos el desempaquetado para que coincida con el return anterior
    loaders, classes = get_data_loaders(batch_size=64)
    
    train_loader, val_loader, test_loader = loaders

    print(f"\n--- Iniciando Análisis Exploratorio de Datos (EDA) ---")
    print(f"Clases a analizar: {classes}")

    # 2. Ejecutar el EDA
    # Dependiendo de cómo definiste perform_eda, podrías pasarle 'classes' o el loader
    # Si perform_eda necesita el dataset para ver atributos como .classes:
    # Pasamos el dataset interno del subset si es necesario
    train_ds = train_loader.dataset 
    
    perform_eda(train_ds, train_loader)

if __name__ == "__main__":
    main()