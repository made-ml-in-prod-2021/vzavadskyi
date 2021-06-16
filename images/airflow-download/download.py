import os

import click
import pandas as pd
from sklearn.datasets import make_classification


@click.command()
@click.argument("--output_dir")
def download(output_dir: str):
    X, y = make_classification(n_samples=1000)
    os.makedirs(output_dir, exist_ok=True)
    df_X = pd.DataFrame(X)
    df_y = pd.DataFrame(y)
    df_y.columns = ["target"]
    df_X.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    df_y.to_csv(os.path.join(output_dir, "target.csv"), index=False)


if __name__ == "__main__":
    download()
