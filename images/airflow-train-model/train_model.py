import os

import click
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


@click.command()
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--n-estimators", default=1000, type=int)
def train_model(input_dir: str, output_dir: str, n_estimators: int):

    X = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
    y = pd.read_csv(os.path.join(input_dir, "y_train.csv"))

    os.makedirs(output_dir, exist_ok=True)

    model = RandomForestClassifier(n_estimators=n_estimators)

    model.fit(X, y.values.ravel())

    joblib.dump(model, os.path.join(output_dir, "random_forest.joblib"))


if __name__ == "__main__":
    train_model()
