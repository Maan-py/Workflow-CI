import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- BAGIAN INI DIHAPUS/DIKOMENTARI ---
# Jangan hardcode URI atau Experiment saat menggunakan 'mlflow run' di CI/CD
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# mlflow.set_tracking_uri(f"file:{BASE_DIR}/mlruns")
# mlflow.set_experiment("workflow-ci")
# --------------------------------------

df = pd.read_csv("UNSW_NB15_preprocessing/UNSW_NB15_preprocessed.csv")

X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.sklearn.autolog()

# Gunakan context manager start_run()
# Jika dijalankan via 'mlflow run', dia akan otomatis mengambil ID yang sudah dibuat sebelumnya.
with mlflow.start_run() as run:
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    # Ambil Run ID dari object 'run' yang sedang aktif
    run_id = run.info.run_id

    # Simpan Run ID ke file
    with open("run_id.txt", "w") as f:
        f.write(run_id)

    # Tidak perlu log_model manual jika autolog sudah aktif,
    # tapi jika ingin memastikan path-nya:
    mlflow.sklearn.log_model(
        sk_model=model, artifact_path="model", registered_model_name=None
    )
