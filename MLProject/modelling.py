import mlflow
import mlflow.sklearn
import pandas as pd
import dagshub
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DAGSHUB_TOKEN = os.environ.get('DAGSHUB_TOKEN', None)

if DAGSHUB_TOKEN:
    # Authenticate to DagsHub
    dagshub.auth.add_app_token(DAGSHUB_TOKEN)
    print("Successfully authenticated with DagsHub.")
else:
    print("DAGSHUB_TOKEN not found in environment variables.")

# MLflow tracking
mlflow.set_tracking_uri("https://dagshub.com/Maan-py/SMSML_Muhammad-Luqmaan.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = "Maan-py"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ.get("DAGSHUB_TOKEN")

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
dagshub.init(
    repo_owner="Maan-py", 
    repo_name="SMSML_Muhammad-Luqmaan",
    mlflow=True,
)

# Pastikan experiment ada
experiment_name = "UNSW_NB15_Basic"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

df = pd.read_csv("UNSW_NB15_preprocessing/UNSW_NB15_preprocessed.csv")

X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run(run_name="UNSW_NB15_Basic_Run"):
    mlflow.sklearn.autolog()

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
