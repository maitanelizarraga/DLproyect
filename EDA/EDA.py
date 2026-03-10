import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CONFIGURACIÓN DE ESTILO PROFESIONAL
sns.set_theme(style="whitegrid")

def importarcsv(): 
    # Ensure the correct path to the dataset
    route = "data/insurance.csv"
    if not os.path.exists(route):
        raise FileNotFoundError(f"File not found in the next route: {route}")
    df = pd.read_csv(route) 
    return df

def initialinspection(df): 
    print("INITIAL INSPECTION")
    print(f"Dimensions: {df.shape}")
    print("\nData types and nulls:")
    print(df.info())
    print("\nDescriptive Statistics:")
    print(df.describe())
    print("\nSmoker Count:")
    print(df['smoker'].value_counts())

def datacleaning(df): 
    print("\n=== DATA CLEANING ===")
    # Delete duplicates
    initial_count = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_count:
        print(f"Duplicated rows deleted: {initial_count - len(df)}")
    
    # 2. Outliers analysis in 'charges' (Target Variable)
    # Important to choose Loss Function
    q1 = df['charges'].quantile(0.25)
    q3 = df['charges'].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df['charges'] < (q1 - 1.5 * iqr)) | (df['charges'] > (q3 + 1.5 * iqr))]
    print(f"Possible atipic values detected in 'charges': {len(outliers)} ")
    
    return df

def visual_eda(df): 
    print("\n GENERATING AND SHOWING VISUALIZATIONS")
    
    folder_name = "visualizations"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Foulder '{folder_name}' created.")

    #Graph 1: Distribution of Charges 
    plt.figure(figsize=(10, 5))
    sns.histplot(df['charges'], kde=True, color='blue')
    plt.title('Distribution of Medical Charges (Target)')
    #plt.savefig(f'{folder_name}/eda_distribution.png') 
    plt.show()

    #Graph 2: Correlation Matrix 
    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numeric Variables')
    #plt.savefig(f'{folder_name}/eda_correlation.png')
    plt.show()

    #Graph 3: Age vs Charges by Smoker/Non-Smoker
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='age', y='charges', hue='smoker', data=df, palette='magma', alpha=0.7)
    plt.title('Age vs Charges by Smoker/Non-Smoker')
    #plt.savefig(f'{folder_name}/eda_segmentation_smokers.png')
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
        print(f"Error during process: {e}")
        
    except Exception as e:
        print(f"Error: {e}. Make sure the csv file is in the following route:'data/insurance.csv'")

if __name__ == "__main__": 
    main()