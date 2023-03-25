import typing as t
import pytest
import sqlite_utils

from pathlib import Path
from datasette.app import Datasette
from faker import Faker


@pytest.fixture(scope="function")
def sqml_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    db_directory = tmp_path_factory.mktemp("dbs")
    db_path = db_directory / "sqml.db"
    faker = Faker()
    db = sqlite_utils.Database(db_path)
    data_regression = sqlite_utils.db.Table(db, "data_regression")
    data_regression.insert_all(
        [
            dict(
                feature1=faker.pyfloat(),
                feature2=faker.pyfloat(),
                target=faker.pyfloat(),
            )
            for _ in range(10)
        ]
    )
    data_classification = sqlite_utils.db.Table(db, "data_classification")
    data_classification.insert_all(
        [
            dict(
                feature1=faker.pyfloat(),
                feature2=faker.pyfloat(),
                target=faker.random_int(min=0, max=1),
            )
            for _ in range(10)
        ]
    )
    return db_path


@pytest.fixture(scope="session")
def datasette_metadata() -> dict[str, t.Any]:
    return {"plugins": {"datasette-ml": {"db": "sqml"}}}


@pytest.fixture(scope="function")
def datasette(sqml_db: Path, datasette_metadata: dict[str, t.Any]) -> Datasette:
    return Datasette([str(sqml_db)], metadata=datasette_metadata)
