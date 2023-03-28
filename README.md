# datasette-ml

> A Datasette plugin providing an MLOps platform to train, evaluate and make predictions from machine learning models

[![PyPI](https://img.shields.io/pypi/v/datasette-ml.svg)](https://pypi.org/project/datasette-ml/)
[![CI/CD](https://github.com/rclement/datasette-ml/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/rclement/datasette-ml/actions/workflows/ci-cd.yml)
[![Coverage Status](https://img.shields.io/codecov/c/github/rclement/datasette-ml)](https://codecov.io/gh/rclement/datasette-ml)
[![License](https://img.shields.io/github/license/rclement/datasette-ml)](https://github.com/rclement/datasette-ml/blob/master/LICENSE)

Try out a live demo at [https://datasette-ml-demo.vercel.app](https://datasette-ml-demo.vercel.app/-/dashboards)

**WARNING**: this plugin is still experimental and not ready for production.
Some breaking changes might happen between releases before reaching a stable version.
Use it at your own risks!

![Datasette Dashboards Demo](https://raw.githubusercontent.com/rclement/datasette-ml/master/demo/datasette-ml-demo.png)

## Installation

Install this plugin in the same environment as Datasette:

```bash
$ datasette install datasette-ml
```

## Usage

Define dashboards within `metadata.yml` / `metadata.json`:

```yaml
plugins:
  datasette-ml:
    db: sqml
```

A new menu entry is now available, pointing at `/-/ml` to access the MLOps dashboard.

### Configuration properties

| Property | Type     | Description                                     |
| -------- | -------- | ----------------------------------------------- |
| `db`     | `string` | Database to store ML models (default is `sqml`) |

### SQL functions

#### Loading sample datasets

`sqml_load_dataset`

#### Training models

`sqml_train`

#### Running predictions

`sqml_predict`

## Development

To set up this plugin locally, first checkout the code.
Then create a new virtual environment and the required dependencies:

```bash
poetry shell
poetry install
```

To run the QA suite:

```bash
poetry run qa
```

## Demo

With the developmnent environment setup, you can run the demo locally:

```bash
datasette sqml.db --create
```

## License

Licensed under Apache License, Version 2.0

Copyright (c) 2023 - present Romain Clement
