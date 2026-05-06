import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


def preprocess_data(df):
    import pandas as pd

    # from categorlical to numerical
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})

    # one-hot Encoding for regions
    df = pd.get_dummies(df, columns=['region'], drop_first=False, dtype=int)

    return df

def division(df):
    X = df.drop('charges', axis=1)
    # y = df['charges']
    # we use logarithms to normalize as is often skewed
    y = np.log1p(df['charges'])

    #70% for training and 30% for validation + test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    # Now we divide half for validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def estandarizacion(X_train, X_val, X_test):
    scaler = StandardScaler()
    
    # fit_transform learns and scales training data, while transform applies that learned scaling to new data (val/test).
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled

def main():
    import pandas as pd

    df = pd.read_csv("data/insurance_cleaned.csv")
    df = preprocess_data(df)
    df.to_csv("data/insurance_preprocessed.csv", index=False)
    X_train, X_val, X_test, y_train, y_val, y_test = division(df)
    X_train_scaled, X_val_scaled, X_test_scaled = estandarizacion(X_train, X_val, X_test)
    
    # Save preprocessed data for future use
    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv("data/X_train_scaled.csv", index=False)
    pd.DataFrame(X_val_scaled, columns=X_val.columns).to_csv("data/X_val_scaled.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv("data/X_test_scaled.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_val.to_csv("data/y_val.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)


if __name__ == "__main__":
    main()
