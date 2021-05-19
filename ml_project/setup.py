from setuptools import find_packages, setup

setup(
    name="ml_example",
    packages=find_packages(),
    version="0.1.0",
    description="Pipeline for Heart Disease prediction",
    author="Vadim Zavadskiy",
    install_requires=[
        "pandas==1.1.5",
        "pytest==6.1.1",
        "numpy==1.18.5",
        "faker==4.1.7",
        "matplotlib==3.3.1",
        "yaml==0.2.5",
        "omegaconf==2.0.6",
        "setuptools==49.6.0",
        "hydra-core==1.0.6",
        "click==7.1.2",
        "python-dotenv>=0.5.1",
        "scikit-learn==0.24.1",
        "dataclasses==0.8",
        "pyyaml==5.4.1",
        "marshmallow-dataclass==8.3.0",
    ],
    license="MIT",
)
