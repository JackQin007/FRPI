from setuptools import setup, find_packages

setup(
    name='frpi',
    version='0.0.1',
    description='Constrained policy optimization algorithms',
    packages=find_packages(),
        install_requires=[
        'gymnasium',
        'dm-haiku',
        'optax',
        'numpyro',
    ],
)
