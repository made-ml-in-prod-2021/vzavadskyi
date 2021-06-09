import os
import joblib

import click
import pandas as pd


@click.command()
@click.option("--input-dir-data")
@click.option("--input-dir-model")
@click.option("--output-dir")
def get_predict(input_dir_data: str, input_dir_model: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir_data, "data.csv"))
    model = joblib.load(os.path.join(input_dir_model, "random_forest.joblib"))

    os.makedirs(output_dir, exist_ok=True)

    predicts = model.predict(data)
    df_predicts = pd.DataFrame(predicts)
    df_predicts.columns = ["predictions"]

    df_predicts.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


if __name__ == "__main__":
    get_predict()
