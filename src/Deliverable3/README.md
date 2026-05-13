# DLproyect 
# ASSIGMENT 3 - Audio Noise Suppression with Transformers

## Project Structure
- `images/`: Spectrogram comparison visualizations generated during baseline training.
- `images_optimized/`: Spectrogram comparison visualizations generated during optimized model training.
- `models/`: Saved model weights from baseline training (`.pth`).
- `models_optimized/`: Saved model weights from hyperparameter-optimized training (`.pth`).

## How to Run
To replicate the results, execute the scripts in the following order:
1.  Data Pipeline Setup: `main.py`
    Downloads datasets, performs strict train/validation/test splits to prevent data leakage, and verifies the data pipeline.

2. Exploratory Data Analysis `eda.py`
    Analyzes the dataset and generates initial visualizations.

3. Initial Baseline Training: `train.py`
    Trains a basic Audio Transformer with default hyperparameters (5 epochs) to establish a starting point for performance comparison.

4. Hyperparameter Optimization: `tune.py`
    Uses Optuna with TPE (Tree-structured Parzen Estimator) sampling to search for optimal learning rate, model dimensions (d_model), number of transformer layers, and weight decay.

5. Optimized Model Training: `train_best_model.py`
    Trains the definitive model using the winning hyperparameter configuration from Optuna for 50 epochs with learning rate scheduling and gradient clipping.

6. Final Evaluation: `eval_test.py`
    Evaluates the best trained model on completely unseen test data, calculates final MSE loss, and generates spectrogram comparison visualizations for the report.
