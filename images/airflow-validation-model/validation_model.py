import os
import json

import click
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


@click.command()
@click.option("--input-dir-data")
@click.option("--input-dir-model")
@click.option("--output-dir")
def validation_model(input_dir_data: str, input_dir_model: str, output_dir: str):
    X = pd.read_csv(os.path.join(input_dir_data, "X_test.csv"))
    y = pd.read_csv(os.path.join(input_dir_data, "y_test.csv"))

    os.makedirs(output_dir, exist_ok=True)

    model = joblib.load(os.path.join(input_dir_model, "random_forest.joblib"))

    predicts = model.predict(X)

    metrics = {}

    metrics["accuracy_score"] = accuracy_score(y, predicts)
    metrics["f1_score"] = f1_score(y, predicts)
    metrics["roc_auc_score"] = roc_auc_score(y, predicts)

    with open(os.path.join(output_dir, "metrics.json"), "w") as metrics_file:
        json.dump(metrics, metrics_file)


if __name__ == "__main__":
    validation_model()
