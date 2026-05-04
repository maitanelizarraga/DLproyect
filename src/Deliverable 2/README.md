# DLproyect 
# ASSIGMENT 2 - Pneumonia Detection with Deep Learning

## Project Structure
- `src/`: Python scripts for EDA, Preprocessing, and Model Training.
- `data/`: Raw and processed datasets (Chest X-Ray images).
- `visualizations/`: Plots generated during data analysis and training.
- `models/`: Saved weights of the PyTorch model (`.pth`).

## How to Run
To replicate the results, execute the scripts in the following order:
1. Exploratory Data Analysis: `main.py`
    Analyzes the dataset, checks for class balance, and generates initial visualizations (EDA).

2. Initial Baseline: `SimpleCNN.py`
    Trains a basic Convolutional Neural Network from scratch to establish a starting point in performance without any optimization.

3. Hyperparameter Optimization: `optuna_search.py`
    Looks for the best architecture and optimal hyperparameters for our model.

4. Optimization Dashboard: `lanzar_dashboard_Deliverable_2.py`
    Used to visually check the best combination of hyperparameters and architecture resulting from the search.

5. Model Training (Custom): `train_best_model.py`
    We proceed to train the definitive model using the winning configuration from Optuna.

6. Model Training (ResNet18): `modelo_preentrenado.py`
    We train a ResNet18 to have a solid baseline for comparison.
    
7. Model Training (VGG16): `model_transfer.py`
    We train a VGG16 to add another performance benchmark.

8. Final Evaluation: `final_comparison.py`
    Run this script to see the final results and determine which one is the best model (accuracy, F1-score, etc.).

9. Explainable AI (XAI): `xai_gradcam_comparison.py`
    Finally, we generate heatmaps to see the explainability of the best model and understand what the AI is actually looking at.