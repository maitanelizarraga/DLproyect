# DLproyect
# Insurance Price Prediction with Deep Learning

## Project Structure
- `src/`: Python scripts for EDA, Preprocessing, and Model Training.
- `data/`: Raw and processed datasets.
- `visualizations/`: Plots generated during data analysis and training.
- `models/`: Saved weights of the PyTorch model (`.pth`).

## How to Run
To replicate the results, execute the scripts in the following order:
1. Exploratory Data Analysis: `python src/EDA.py`
    Analyzes the dataset and generates initial visualizations.

2. Data Preprocessing: `python src/preprocesing.py`
    Cleans the data, applies One-Hot Encoding, and performs Log-Transformation/Scaling.

3. Model Training & Optimization: `python src/pytorch.py`
    Runs the Optuna hyperparameter search and trains the Neural Network.

4. Sequential Workflow: `python src/main.py`
    Executes the entire pipeline from start to finish automatically.
