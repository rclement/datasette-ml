from setuptools import setup

VERSION = "0.0.0"

setup(
    name="datasette-ml",
    description="A Datasette plugin providing an MLOps platform to train, eval and predict ML models",
    author="Romain Clement",
    url="https://github.com/rclement/datasette-ml",
    license="Apache License, Version 2.0",
    version=VERSION,
    packages=["datasette_ml"],
    entry_points={"datasette": ["ml = datasette_ml"]},
    install_requires=["datasette"],
)
