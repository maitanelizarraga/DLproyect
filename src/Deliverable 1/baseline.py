from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np

def run_baseline():
    # Charging data
    X_train = pd.read_csv("data/X_train_scaled.csv")
    X_test = pd.read_csv("data/X_test_scaled.csv")
    y_train = pd.read_csv("data/y_train.csv")
    y_test = pd.read_csv("data/y_test.csv")

    # Training
    baseline = LinearRegression()
    baseline.fit(X_train, y_train)

    # Prediction and log reversal
    b_preds = baseline.predict(X_test)
    real_preds = np.expm1(b_preds)
    real_y = np.expm1(y_test)

    mae = mean_absolute_error(real_y, real_preds)
    r2 = r2_score(real_y, real_preds)
    
    return mae, r2

if __name__ == "__main__":
    mae, r2 = run_baseline()
    print(f"BASELINE MAE: ${mae:.2f}")
    print(f"BASELINE R2: {r2:.4f}")