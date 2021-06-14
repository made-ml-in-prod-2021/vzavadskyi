import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.sensors.filesystem import FileSensor

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    "prepare_model",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=days_ago(0),
) as dag:

    check_data_ready = FileSensor(
        task_id="wait-raw-data",
        filepath="/home/vadim/MADE/vzavadskyi/data/raw/data.csv",
        poke_interval=10,
        retries=2
    )

    check_target_data = FileSensor(
        task_id="wait-target-data",
        filepath="/home/vadim/MADE/vzavadskyi/data/raw/target.csv",
        poke_interval=10,
        retries=2
    )


    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        volumes=["/home/vadim/MADE/vzavadskyi/data:/data"],
    )
    train_test_split = DockerOperator(
        image="airflow-train-test-split",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/processed/{{ ds }}/splitted",
        task_id="docker-airflow-train-test-split",
        do_xcom_push=False,
        volumes=["/home/vadim/MADE/vzavadskyi/data:/data"],
    )
    train_model = DockerOperator(
        image="airflow-train-model",
        command="--input-dir /data/processed/{{ ds }}/splitted --output-dir /data/models/{{ ds }} --n-estimators=40",
        task_id="docker-airflow-train-model",
        do_xcom_push=False,
        volumes=["/home/vadim/MADE/vzavadskyi/data:/data"],
    )
    validation_model = DockerOperator(
        image="airflow-validation-model",
        command="--input-dir-data /data/processed/{{ ds }}/splitted --input-dir-model /data/models/{{ ds }} --output-dir /data/models/{{ ds }}/metrics",
        task_id="docker-airflow-validation-model",
        do_xcom_push=False,
        volumes=["/home/vadim/MADE/vzavadskyi/data:/data"],
    )

    [check_data_ready, check_target_data] >> preprocess >> train_test_split >> train_model >> validation_model
