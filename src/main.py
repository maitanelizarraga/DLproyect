import os
import sys

# Add src directory to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from EDA import main as run_eda
from preprocesing import main as run_preprocessing
from pytorch import run_pytorch_model
from baseline import run_baseline

def main():
    print("\n" + "="*50)
    print("      STARTING INSURANCE PRICE PREDICTION PIPELINE")
    print("="*50)

    # 1. EDA and preprocessing
    run_eda()
    run_preprocessing()

    # 2. Training and metric capture
    p_mae, p_r2 = run_pytorch_model()
    b_mae, b_r2 = run_baseline()

    # 3. Calculate improvement
    improvement = ((b_mae - p_mae) / b_mae) * 100

    # 4. Final Comparative Summary
    print("---------------------------------------")
    print("             FINAL COMPARATIVE SUMMARY")
    print("---------------------------------------")
    print(f"{'Metric':<15} | {'Baseline (Linear)':<20} | {'PyTorch (Deep)':<15}")
    print("---------------------------------------")
    print(f"{'MAE (Error)':<15} | ${b_mae:<19.2f} | ${p_mae:<14.2f}")
    print(f"{'R2 (Precision)':<15} | {b_r2:<20.4f} | {p_r2:<15.4f}")
    print("---------------------------------------")
    print(f"RESULT: Deep Learning is {improvement:.2f}% better.")
    print("---------------------------------------")

if __name__ == "__main__":
    main()