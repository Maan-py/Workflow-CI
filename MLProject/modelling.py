import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow.set_tracking_uri(f"file:{BASE_DIR}/mlruns")
mlflow.set_experiment("workflow-ci")  # pastikan experiment

df = pd.read_csv("UNSW_NB15_preprocessing/UNSW_NB15_preprocessed.csv")
X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.sklearn.autolog()  # autologging tanpa start_run manual

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

mlflow.sklearn.log_model(model, name="model")

active_run = mlflow.active_run()
if active_run:
    print("Run ID:", active_run.info.run_id)
