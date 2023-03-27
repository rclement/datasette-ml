import typing as t
import numpy as np
import pandas as pd
import pytest
import sqlite_utils

from pathlib import Path
from datasette.app import Datasette
from sklearn.datasets import make_classification, make_regression


@pytest.fixture(scope="session")
def regression_dataset() -> pd.DataFrame:
    X, y = make_regression(n_samples=1000, n_features=3, random_state=0)
    return pd.DataFrame(
        np.concatenate([X, np.vstack(y)], axis=1),
        columns=[f"feature{i + 1}" for i in range(X.shape[1])] + ["target"],
    )


@pytest.fixture(scope="session")
def classification_dataset() -> pd.DataFrame:
    X, y = make_classification(n_samples=1000, random_state=0)
    return pd.DataFrame(
        np.concatenate([X, np.vstack(y)], axis=1),
        columns=[f"feature{i + 1}" for i in range(X.shape[1])] + ["target"],
    )


@pytest.fixture(scope="function")
def sqml_db(
    tmp_path_factory: pytest.TempPathFactory,
    regression_dataset: pd.DataFrame,
    classification_dataset: pd.DataFrame,
) -> Path:
    db_directory = tmp_path_factory.mktemp("dbs")
    db_path = db_directory / "sqml.db"
    db = sqlite_utils.Database(db_path)

    data_regression = sqlite_utils.db.Table(db, "data_regression")
    data_regression.insert_all(regression_dataset.to_dict("records"))

    data_classification = sqlite_utils.db.Table(db, "data_classification")
    data_classification.insert_all(classification_dataset.to_dict("records"))
    return db_path


@pytest.fixture(scope="session")
def datasette_metadata() -> t.Mapping[str, t.Any]:
    return {"plugins": {"datasette-ml": {"db": "sqml"}}}


@pytest.fixture(scope="function")
def datasette(sqml_db: Path, datasette_metadata: t.Mapping[str, t.Any]) -> Datasette:
    return Datasette([str(sqml_db)], metadata=datasette_metadata)
