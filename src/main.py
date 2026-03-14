
import os

def main():
    print("--- INITIALIZING PIPELINE OF DEEP LEARNING ---")
    os.system("python src/EDA.py")
    os.system("python src/preprocesing.py")
    os.system("python src/pytorch.py")
    print("--- PIPELINE FINISHED SUCCESSFULLY ---")

if __name__ == "__main__":
    main()