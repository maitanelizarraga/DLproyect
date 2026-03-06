import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# 1. Reading and preprocessing data
df = pd.read_csv("insurance.csv")

print(df.dtypes)

# separate the target value
X = df.drop("charges", axis=1)
y = df["charges"]

print("Columnas reales de X:")
print(X.columns.tolist())


# encoding for categorical variables (NO get_dummies)
cat_cols = ["sex", "smoker", "region"]

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])


# Asegurar que todo es numérico
X = X.astype(float)
y = y.astype(float)

# train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# 2. Transform into tensors (REGRESIÓN → float32)
X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
X_test_t  = torch.tensor(X_test.values, dtype=torch.float32)

y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_t  = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


# 3. DataLoaders
train_data = TensorDataset(X_train_t, y_train_t)
test_data  = TensorDataset(X_test_t, y_test_t)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader  = DataLoader(test_data, batch_size=64)


# 4. Model (REGRESIÓN → salida 1)
input_dim = X_train.shape[1]

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)   # salida de regresión
        )

    def forward(self, x):
        return self.model(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork().to(device)

print(f"Using device: {device}")
print(model)


# 5. Loss and optimizer (REGRESIÓN)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 6. Train and test functions
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            print(f"Batch {batch}, Loss: {loss.item():.4f}")


def test(dataloader, model, loss_fn):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= len(dataloader)
    print(f"Test Loss: {test_loss:.4f}")


# 7. Training
epochs = 10
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Training finished")


# 8. Saving model
torch.save(model.state_dict(), "model.pth")
print("Model saved in model.pth")
