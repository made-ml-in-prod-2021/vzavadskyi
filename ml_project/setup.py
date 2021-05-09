from setuptools import find_packages, setup

setup(
    name="ml_example",
    packages=find_packages(),
    version="0.1.0",
    description="Pipeline for Heart Disease prediction",
    author="Vadim Zavadskiy",
    install_requires=[
        "click==7.1.2",
        "python-dotenv>=0.5.1",
        "scikit-learn==0.24.1",
        "dataclasses==0.8",
        "pyyaml==5.4.1",
        "marshmallow-dataclass==8.3.0",
        "pandas==1.1.5",
    ],
    license="MIT",
)
