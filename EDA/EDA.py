# EXPLORATORY DATA ANALYSIS

def importarcsv(): 
    import pandas as pd 
    df = pd.read_csv("data/insurance.csv") 
    return df

def initialinspection(df): 
    print("ANALYSIS:")
    print("First row:" + "\n" + str(df.iloc[0])) 
    print(" ")
    print("Data types:" + "\n" + str(df.dtypes)) 
    print(" ")
    print("Shape:" + "\n" + str(df.shape)) 
    print(" ")
    print("Missing values:" + "\n" + str(df.isnull().sum())) 
    print(" ")
    print("Unique values:" + "\n" + str(df.nunique()))
    print(" ")

def datacleaning(df): 
    #
    return df

#def eda(df): 
    



def main():
    df = importarcsv() 
    initialinspection(df)
    df = datacleaning(df)
    #eda(df)
    
if __name__ == "__main__": 
    main()
