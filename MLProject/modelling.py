import mlflow
import mlflow.sklearn
import pandas as pd
import dagshub
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
dagshub.init(
    repo_owner="Maan-py", 
    repo_name="SMSML_Muhammad-Luqmaan",
    mlflow=True
)
mlflow.set_experiment("UNSW_NB15_Basic")

df = pd.read_csv("UNSW_NB15_preprocessing/UNSW_NB15_preprocessed.csv")

X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run():
    mlflow.sklearn.autolog()

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
