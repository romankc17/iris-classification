
from setuptools import setup, find_packages

setup(
    name='iris_classification',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'matplotlib',
        'jupyter',
    ],
    entry_points={
        'console_scripts': [
            'train_model=src.train_model:main',
            'make_prediction=src.predict:predict',
        ],
    },
)
