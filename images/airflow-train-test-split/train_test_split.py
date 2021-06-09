import os
import pandas as pd
import click

from sklearn.model_selection import train_test_split


@click.command()
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--test-size", default=0.3, type=float)
def split(input_dir: str, output_dir: str, test_size: float):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))
    os.makedirs(output_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=test_size
    )
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)


if __name__ == "__main__":
    split()
