def preprocess_data(df):
    import pandas as pd

    # Convertir variables categóricas binarias
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})

    # One-Hot Encoding para las 4 regiones (sin drop_first)
    df = pd.get_dummies(df, columns=['region'], drop_first=False)

    return df


def main():
    import pandas as pd

    df = pd.read_csv("data/insurance_cleaned.csv")
    df = preprocess_data(df)
    df.to_csv("data/insurance_preprocessed.csv", index=False)


if __name__ == "__main__":
    main()
