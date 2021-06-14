import pytest

@pytest.fixture()
def project_tree():
    return {
        "docker-airflow-preprocess": ["docker-airflow-train-test-split"],
        "docker-airflow-train-test-split": ["docker-airflow-train-model"],
    }


def test_can_load_predict_dag(dag_bag):
    dag = dag_bag.get_dag(dag_id="get_predict")
    assert len(dag.tasks) == 3


def correct_tree(dag_bag, project_tree):
    dag = dag_bag.get_dag(dag_id="get_predict")
    for task_id, downstream_list in project_tree.items():
        assert dag.has_task(task_id), f"Uncorrect chain of dags. Cannot find task whith id {task_id}"
        task = dag.get_task(task_id)
        assert task.downstream_task_ids == set(downstream_list)
