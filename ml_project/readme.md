Основой проекта явяется код и архитектура проекта:
https://github.com/made-ml-in-prod-2021/ml_project_example авторства Михаила Марюфича

Установка 

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python setup.py install

Запуск:

python.exe ml_example/main.py --config-dir YOUR_PATH\ml_project\configs hydra.run.dir=.

Тестирование:

pytest -v
