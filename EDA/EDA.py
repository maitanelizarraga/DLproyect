import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CONFIGURACIÓN DE ESTILO PROFESIONAL
sns.set_theme(style="whitegrid")

def importarcsv(): 
    # Aseguramos la ruta correcta según tu estructura /data
    ruta = "data/insurance.csv"
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el archivo en {ruta}")
    df = pd.read_csv(ruta) 
    return df

def initialinspection(df): 
    print("=== INSPECCIÓN INICIAL ===")
    print(f"Dimensiones: {df.shape}")
    print("\nTipos de Datos y Nulos:")
    print(df.info())
    print("\nEstadísticas Descriptivas:")
    print(df.describe())
    print("\nConteo de Fumadores (Variable Crítica):")
    print(df['smoker'].value_counts())

def datacleaning(df): 
    print("\n=== DATA CLEANING ===")
    # 1. Eliminar duplicados (Mantenemos la integridad de los datos)
    initial_count = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_count:
        print(f"Se eliminaron {initial_count - len(df)} filas duplicadas.")
    
    # 2. Análisis de Outliers en la variable objetivo (Charges)
    # Justificación: Importante para elegir la Loss Function (Capítulo 5)
    q1 = df['charges'].quantile(0.25)
    q3 = df['charges'].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df['charges'] < (q1 - 1.5 * iqr)) | (df['charges'] > (q3 + 1.5 * iqr))]
    print(f"Se detectaron {len(outliers)} posibles valores atípicos en 'charges'.")
    
    return df

def visual_eda(df): 
    print("\n=== GENERANDO Y MOSTRANDO VISUALIZACIONES ===")
    
    # 1. Crear carpeta para organizar el proyecto (Estructura profesional)
    folder_name = "visualizations"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Carpeta '{folder_name}' creada.")

    #Graph 1: Distribution of Charges 
    plt.figure(figsize=(10, 5))
    sns.histplot(df['charges'], kde=True, color='blue')
    plt.title('Distribution of Medical Charges (Target)')
    #plt.savefig(f'{folder_name}/eda_distribucion.png') # Se guarda en la carpeta
    plt.show() # Se muestra en pantalla

    #Graph 2: Correlation Matrix 
    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numeric Variables')
    #plt.savefig(f'{folder_name}/eda_correlacion.png')
    plt.show()

    #Graph 3: Age vs Charges by Smoker/Non-Smoker
    # Este es el gráfico más importante para defender tu Red Neuronal
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='age', y='charges', hue='smoker', data=df, palette='magma', alpha=0.7)
    plt.title('Age vs Charges by Smoker/Non-Smoker')
    #plt.savefig(f'{folder_name}/eda_segmentacion_fumadores.png')
    plt.show()


    #Graph 4: Charges for smokers vs non-smokers
    plt.figure(figsize=(6,4))
    sns.boxplot(x="smoker", y="charges", data=df)
    plt.title("Charges by smoker/non-smoker")
    plt.show()

    #Graph 5: Charges by sex
    plt.figure(figsize=(6,4))
    sns.boxplot(x="sex", y="charges", data=df)
    plt.title("Charges by sex")
    plt.show()

    #Graph 6: Charges by region
    plt.figure(figsize=(8,4))
    sns.boxplot(x="region", y="charges", data=df)
    plt.title("Charges by region")
    plt.show()


    #Graph 7: BMI vs Charges coloreado por fumador
    plt.figure(figsize=(6,4))
    sns.scatterplot(x="bmi", y="charges", hue="smoker", data=df)
    plt.title("BMI vs charges by smoker/non-smoker")
    plt.show()



def main():
    try:


        # Import dataset
        df = pd.read_csv("data/insurance.csv") 
        
        # Fast view of the dataset
        print(f"Dataset cargado: {df.shape[0]} filas.")
        
        # Delete duplicates
        df = df.drop_duplicates()
        


        
        df = importarcsv() 
        initialinspection(df)
        df = datacleaning(df)
        visual_eda(df) 
        
        # Save cleaned data for future use
        df.to_csv("data/insurance_cleaned.csv", index=False)
        
    except Exception as e:
        print(f"Error durante el proceso: {e}")
        
    except Exception as e:
        print(f"Error: {e}. Asegúrate de que el archivo esté en 'data/insurance.csv'")

if __name__ == "__main__": 
    main()