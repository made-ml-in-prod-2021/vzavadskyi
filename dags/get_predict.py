import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "get_predict",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(0),
) as dag:
    get_predict = DockerOperator(
        image="airflow-get-predict",
        command="--input-dir-data /data/processed/{{ ds }} --input-dir-model /data/models/{{ ds }} --output-dir /data/predictions/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-get-predict",
        do_xcom_push=False,
        volumes=["/home/vadim/MADE/vzavadskyi/data:/data"],
    )
