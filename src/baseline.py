from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np

# Load preprocessed data (the data created by your preprocessing.py)
X_train = pd.read_csv("data/X_train_scaled.csv")
X_test = pd.read_csv("data/X_test_scaled.csv")
y_train = pd.read_csv("data/y_train.csv")
y_test = pd.read_csv("data/y_test.csv")

# Train a linear regression model as a baseline
baseline = LinearRegression()
baseline.fit(X_train, y_train)

# Predict and evaluate
b_preds = baseline.predict(X_test)
real_preds = np.expm1(b_preds)
real_y = np.expm1(y_test)

print(f"BASELINE MAE: ${mean_absolute_error(real_y, real_preds):.2f}")
print(f"BASELINE R2: {r2_score(real_y, real_preds):.4f}")