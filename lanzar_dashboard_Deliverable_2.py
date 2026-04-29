import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from optuna_dashboard import run_server

# 1. Le decimos a Optuna dónde está nuestro archivo .log (con el nombre nuevo)
storage = JournalStorage(JournalFileStorage("optuna_journal.log"))

# 2. Lanzamos el servidor web
if __name__ == "__main__":
    print("🚀 Arrancando Optuna Dashboard...")
    print("👉 Abre tu navegador en: http://127.0.0.1:8080")
    run_server(storage, host="127.0.0.1", port=8080)