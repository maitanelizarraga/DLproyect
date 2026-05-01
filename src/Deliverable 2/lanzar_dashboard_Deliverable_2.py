import optuna
from optuna_dashboard import run_server

# 1. Need to change and put the path to your actual database file (Este es el mío (Ziortza))
storage = optuna.storages.RDBStorage("sqlite:///src/Deliverable 2/optuna_study.db")

# 2. Launch the web server
if __name__ == "__main__":
    print("Starting Optuna Dashboard...")
    print("Open your browser at: http://127.0.0.1:8080")
    run_server(storage, host="127.0.0.1", port=8080)