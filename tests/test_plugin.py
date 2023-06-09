import copy
import json
import sys
import typing as t
import sklearn
import pytest

from urllib.parse import urlencode
from pathlib import Path
from datasette.app import Datasette
from datasette.database import Database
from faker import Faker


@pytest.mark.asyncio
async def test_plugin_is_installed(datasette: Datasette) -> None:
    response = await datasette.client.get("/-/plugins.json")
    assert response.status_code == 200

    installed_plugins = {p["name"] for p in response.json()}
    assert "datasette-ml" in installed_plugins


# ------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sql_table",
    [
        "sqml_experiments",
        "sqml_runs",
        "sqml_models",
        "sqml_metrics",
        "sqml_deployments",
    ],
)
async def test_plugin_created_sql_table(datasette: Datasette, sql_table: str) -> None:
    response = await datasette.client.get("/sqml.json")
    assert response.status_code == 200

    available_sql_tables = {f["name"] for f in response.json()["tables"]}
    assert sql_table in available_sql_tables


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sql_view",
    [
        "sqml_runs_overview",
    ],
)
async def test_plugin_created_sql_view(datasette: Datasette, sql_view: str) -> None:
    response = await datasette.client.get("/sqml.json")
    assert response.status_code == 200

    available_sql_views = {f["name"] for f in response.json()["views"]}
    assert sql_view in available_sql_views


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sql_function",
    [
        "sqml_python_version",
        "sqml_sklearn_version",
        "sqml_load_dataset",
        "sqml_train",
        "sqml_predict",
    ],
)
async def test_plugin_registered_sql_function(
    datasette: Datasette, sql_function: str
) -> None:
    response = await datasette.client.get(
        "/sqml.json?sql=select * from pragma_function_list()&_shape=array"
    )
    assert response.status_code == 200

    available_sql_functions = {f["name"] for f in response.json()}
    assert sql_function in available_sql_functions


@pytest.mark.asyncio
async def test_plugin_default_database(
    sqml_db: Path, datasette_metadata: t.Mapping[str, t.Any]
) -> None:
    metadata = copy.deepcopy(datasette_metadata)
    metadata["plugins"].pop("datasette-ml")
    datasette = Datasette([str(sqml_db)], metadata=metadata)

    response = await datasette.client.get("/sqml.json")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_plugin_unknown_database(
    sqml_db: Path, datasette_metadata: t.Mapping[str, t.Any]
) -> None:
    metadata = copy.deepcopy(datasette_metadata)
    metadata["plugins"]["datasette-ml"]["db"] = "unknown"
    datasette = Datasette([str(sqml_db)], metadata=metadata)

    with pytest.raises(KeyError):
        await datasette.invoke_startup()


# ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sqml_python_version(datasette: Datasette) -> None:
    query = """
        SELECT sqml_python_version() AS version;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    rows = response.json()
    assert len(rows) == 1
    assert rows[0]["version"] == sys.version


@pytest.mark.asyncio
async def test_sqml_sklearn_version(datasette: Datasette) -> None:
    query = """
        SELECT sqml_sklearn_version() AS version;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    rows = response.json()
    assert len(rows) == 1
    assert rows[0]["version"] == sklearn.__version__


# ------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dataset",
    ["iris", "digits", "wine", "breast_cancer", "diabetes"],
)
async def test_sqml_load_dataset(datasette: Datasette, dataset: str) -> None:
    query = f"""
        SELECT sqml_load_dataset('{dataset}') AS info;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    rows = response.json()
    assert len(rows) == 1

    info = json.loads(rows[0]["info"])
    db: Database = datasette.get_database("sqml")
    columns = await db.table_columns(info["table"])
    count_res = await db.execute(
        f'select count(*) from [{info["table"]}]',
    )
    count = count_res.rows[0][0]

    assert info["table"] == f"dataset_{dataset}"
    assert columns == info["feature_names"] + ["target"]
    assert count > 0 and count == info["size"]


@pytest.mark.asyncio
async def test_sqml_load_dataset_unknown(datasette: Datasette) -> None:
    query = """
        SELECT sqml_load_dataset('unknown') AS info;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    rows = response.json()
    assert len(rows) == 1

    info = json.loads(rows[0]["info"])
    assert "error" in info.keys()


# ------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prediction_type", "algorithm"),
    [
        ("regression", "dummy"),
        ("regression", "linear_regression"),
        ("regression", "sgd"),
        ("regression", "ridge"),
        ("regression", "ridge_cv"),
        ("regression", "elastic_net"),
        ("regression", "elastic_net_cv"),
        ("regression", "lasso"),
        ("regression", "lasso_cv"),
        ("regression", "decision_tree"),
        ("regression", "ada_boost"),
        ("regression", "bagging"),
        ("regression", "gradient_boosting"),
        ("regression", "random_forest"),
        ("regression", "knn"),
        ("regression", "mlp"),
        ("regression", "svr"),
        ("classification", "dummy"),
        ("classification", "logistic_regression"),
        ("classification", "sgd"),
        ("classification", "ridge"),
        ("classification", "ridge_cv"),
        ("classification", "decision_tree"),
        ("classification", "ada_boost"),
        ("classification", "bagging"),
        ("classification", "gradient_boosting"),
        ("classification", "random_forest"),
        ("classification", "knn"),
        ("classification", "mlp"),
        ("classification", "svc"),
    ],
)
async def test_sqml_train(
    datasette: Datasette, faker: Faker, prediction_type: str, algorithm: str
) -> None:
    experiment_name = faker.bs()
    dataset = f"data_{prediction_type}"
    target = "target"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    rows = response.json()
    assert len(rows) == 1

    training = json.loads(rows[0]["training"])
    db: Database = datasette.get_database("sqml")

    assert training["experiment_name"] == experiment_name
    assert training["prediction_type"] == prediction_type
    assert training["algorithm"] == algorithm
    assert training["deployed"]
    assert isinstance(training["score"], float)

    experiment = (
        await db.execute(
            """
            SELECT *
            FROM sqml_experiments
            WHERE id = 1
            """,
        )
    ).rows[0]
    assert experiment["name"] == experiment_name
    assert experiment["prediction_type"] == prediction_type

    run = (
        await db.execute(
            """
            SELECT *
            FROM sqml_runs
            WHERE id = 1
            """,
        )
    ).rows[0]
    assert run["status"] == "success"
    assert run["algorithm"] == algorithm
    assert run["dataset"] == dataset
    assert run["target"] == target
    assert run["test_size"] == 0.25
    assert run["split_strategy"] == "shuffle"
    assert run["experiment_id"] == 1

    model = (
        await db.execute(
            """
            SELECT *
            FROM sqml_models
            WHERE id = 1
            """,
        )
    ).rows[0]
    assert model["run_id"] == 1
    assert model["library"] == "scikit-learn"
    assert isinstance(model["data"], bytes) and len(model["data"]) > 0

    metrics = {
        m["name"]: m["value"]
        for m in (
            await db.execute(
                """
                SELECT *
                FROM sqml_metrics
                WHERE model_id = 1
                """,
            )
        ).rows
    }

    assert isinstance(metrics["score"], float)
    if prediction_type == "regression":
        assert len(metrics.keys()) == 4
        assert isinstance(metrics["r2"], float)
        assert isinstance(metrics["mae"], float)
        assert isinstance(metrics["rmse"], float)
        assert metrics["score"] == metrics["r2"]
    else:
        assert len(metrics.keys()) == 5
        assert isinstance(metrics["accuracy"], float)
        assert isinstance(metrics["f1"], float)
        assert isinstance(metrics["precision"], float)
        assert isinstance(metrics["recall"], float)
        assert metrics["score"] == metrics["accuracy"]

    deployment = (
        await db.execute(
            """
            SELECT *
            FROM sqml_deployments
            WHERE id = 1
            """,
        )
    ).rows[0]
    assert deployment["experiment_id"] == 1
    assert deployment["model_id"] == 1
    assert deployment["active"]


@pytest.mark.asyncio
async def test_sqml_train_better_model(datasette: Datasette, faker: Faker) -> None:
    db: Database = datasette.get_database("sqml")

    experiment_name = faker.bs()
    prediction_type = "regression"
    algorithm = "linear_regression"
    dataset = f"data_{prediction_type}"
    target = "target"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    await db.execute_write(
        """
        UPDATE sqml_metrics
        SET value = 0.5
        WHERE id = 1 AND name = 'score'
        """
    )

    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    rows = response.json()
    assert len(rows) == 1

    training = json.loads(rows[0]["training"])
    assert training["deployed"]

    runs = (await db.execute("SELECT * FROM sqml_runs ORDER BY id")).rows
    assert len(runs) == 2

    deployments = (await db.execute("SELECT * FROM sqml_deployments ORDER BY id")).rows
    assert len(deployments) == 2
    assert not deployments[0]["active"]
    assert deployments[0]["model_id"] == 1
    assert deployments[1]["active"]
    assert deployments[1]["model_id"] == 2


@pytest.mark.asyncio
async def test_sqml_train_worse_model(datasette: Datasette, faker: Faker) -> None:
    db: Database = datasette.get_database("sqml")

    experiment_name = faker.bs()
    prediction_type = "regression"
    algorithm = "linear_regression"
    dataset = f"data_{prediction_type}"
    target = "target"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    await db.execute_write(
        """
        UPDATE sqml_metrics
        SET value = 1.0
        WHERE id = 1 AND name = 'score'
        """
    )

    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    rows = response.json()
    assert len(rows) == 1

    training = json.loads(rows[0]["training"])
    assert not training["deployed"]

    runs = (await db.execute("SELECT * FROM sqml_runs ORDER BY id")).rows
    assert len(runs) == 2

    deployments = (await db.execute("SELECT * FROM sqml_deployments ORDER BY id")).rows
    assert len(deployments) == 1
    assert deployments[0]["active"]
    assert deployments[0]["model_id"] == 1


@pytest.mark.asyncio
async def test_sqml_train_existing_experiment(
    datasette: Datasette, faker: Faker
) -> None:
    experiment_name = faker.bs()
    prediction_type = "regression"
    algorithm = "linear_regression"
    dataset = f"data_{prediction_type}"
    target = "target"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    db: Database = datasette.get_database("sqml")

    experiment = (
        await db.execute(
            """
            SELECT count(*) AS count
            FROM sqml_experiments
            """,
        )
    ).rows[0]
    assert experiment["count"] == 1

    run = (
        await db.execute(
            """
            SELECT count(*) AS count
            FROM sqml_runs
            """,
        )
    ).rows[0]
    assert run["count"] == 2


@pytest.mark.asyncio
async def test_sqml_train_existing_experiment_wrong_prediction_type(
    datasette: Datasette, faker: Faker
) -> None:
    experiment_name = faker.bs()
    prediction_type = "regression"
    algorithm = "linear_regression"
    dataset = f"data_{prediction_type}"
    target = "target"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    prediction_type = "classification"
    algorithm = "logistic_regression"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    rows = response.json()
    assert len(rows) == 1

    training = json.loads(rows[0]["training"])
    assert "error" in training.keys()


@pytest.mark.asyncio
async def test_sqml_train_wrong_prediction_type_algorithm(
    datasette: Datasette, faker: Faker
) -> None:
    experiment_name = faker.bs()
    prediction_type = "regression"
    algorithm = "logistic_regression"
    dataset = f"data_{prediction_type}"
    target = "target"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    rows = response.json()
    assert len(rows) == 1

    training = json.loads(rows[0]["training"])
    assert "error" in training.keys()


@pytest.mark.asyncio
async def test_sqml_train_unknown_prediction_type(
    datasette: Datasette, faker: Faker
) -> None:
    experiment_name = faker.bs()
    prediction_type = "unknown"
    algorithm = "linear_regression"
    dataset = f"data_{prediction_type}"
    target = "target"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    rows = response.json()
    assert len(rows) == 1

    training = json.loads(rows[0]["training"])
    assert "error" in training.keys()


@pytest.mark.asyncio
async def test_sqml_train_unknown_algorithm(datasette: Datasette, faker: Faker) -> None:
    experiment_name = faker.bs()
    prediction_type = "regression"
    algorithm = "unknown"
    dataset = f"data_{prediction_type}"
    target = "target"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    rows = response.json()
    assert len(rows) == 1

    training = json.loads(rows[0]["training"])
    assert "error" in training.keys()


@pytest.mark.asyncio
async def test_sqml_train_unknown_dataset(datasette: Datasette, faker: Faker) -> None:
    experiment_name = faker.bs()
    prediction_type = "regression"
    algorithm = "linear_regression"
    dataset = "unknown"
    target = "target"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    rows = response.json()
    assert len(rows) == 1

    training = json.loads(rows[0]["training"])
    assert "error" in training.keys()


@pytest.mark.asyncio
async def test_sqml_train_unknown_target(datasette: Datasette, faker: Faker) -> None:
    experiment_name = faker.bs()
    prediction_type = "regression"
    algorithm = "linear_regression"
    dataset = f"data_{prediction_type}"
    target = "unknown"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    rows = response.json()
    assert len(rows) == 1

    training = json.loads(rows[0]["training"])
    assert "error" in training.keys()


@pytest.mark.asyncio
async def test_sqml_train_unknown_split_strategy(
    datasette: Datasette, faker: Faker
) -> None:
    experiment_name = faker.bs()
    prediction_type = "regression"
    algorithm = "linear_regression"
    dataset = f"data_{prediction_type}"
    target = "target"
    test_size = 0.25
    split_strategy = "unknown"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}',
            {test_size},
            '{split_strategy}'
        ) AS training;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    rows = response.json()
    assert len(rows) == 1

    training = json.loads(rows[0]["training"])
    assert "error" in training.keys()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_size",
    [-0.25, 1.1],
)
async def test_sqml_train_out_of_range_test_size(
    datasette: Datasette, faker: Faker, test_size: float
) -> None:
    experiment_name = faker.bs()
    prediction_type = "regression"
    algorithm = "linear_regression"
    dataset = f"data_{prediction_type}"
    target = "target"
    split_strategy = "shuffle"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}',
            {test_size},
            '{split_strategy}'
        ) AS training;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    rows = response.json()
    assert len(rows) == 1

    training = json.loads(rows[0]["training"])
    assert "error" in training.keys()


# ------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prediction_type", "algorithm"),
    [
        ("regression", "linear_regression"),
        ("classification", "logistic_regression"),
    ],
)
async def test_sqml_predict(
    datasette: Datasette, faker: Faker, prediction_type: str, algorithm: str
) -> None:
    experiment_name = faker.bs()
    dataset = f"data_{prediction_type}"
    target = "target"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    db: Database = datasette.get_database("sqml")
    data_row = (
        await db.execute(
            f"""
            SELECT *
            FROM {dataset}
            LIMIT 1
            """,
        )
    ).rows[0]

    features = json.dumps({k: v for k, v in dict(data_row).items() if k != target})
    query = f"""
        SELECT sqml_predict(
            '{experiment_name}',
            '{features}'
        ) AS prediction;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    rows = response.json()
    assert len(rows) == 1

    prediction = rows[0]["prediction"]
    assert isinstance(prediction, float)


@pytest.mark.asyncio
async def test_sqml_predict_unknown_experiment(
    datasette: Datasette, faker: Faker
) -> None:
    experiment_name = faker.bs()
    query = f"""
        SELECT sqml_predict(
            '{experiment_name}',
            '{{}}'
        ) AS prediction;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    rows = response.json()
    assert len(rows) == 1

    prediction = json.loads(rows[0]["prediction"])
    assert "error" in prediction.keys()


@pytest.mark.asyncio
async def test_sqml_predict_no_deployment(datasette: Datasette, faker: Faker) -> None:
    experiment_name = faker.bs()

    db: Database = datasette.get_database("sqml")
    await datasette.invoke_startup()
    await db.execute_write(
        """
        INSERT INTO sqml_experiments(name, prediction_type)
        VALUES (?, ?)
        """,
        (experiment_name, "classification"),
    )

    query = f"""
        SELECT sqml_predict(
            '{experiment_name}',
            '{{}}'
        ) AS prediction;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    rows = response.json()
    assert len(rows) == 1

    prediction = json.loads(rows[0]["prediction"])
    assert "error" in prediction.keys()


# ------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prediction_type", "algorithm"),
    [
        ("regression", "linear_regression"),
        ("classification", "logistic_regression"),
    ],
)
async def test_sqml_predict_batch(
    datasette: Datasette, faker: Faker, prediction_type: str, algorithm: str
) -> None:
    experiment_name = faker.bs()
    dataset = f"data_{prediction_type}"
    target = "target"
    query = f"""
        SELECT sqml_train(
            '{experiment_name}',
            '{prediction_type}',
            '{algorithm}',
            '{dataset}',
            '{target}'
        ) AS training;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    db: Database = datasette.get_database("sqml")
    data_rows = (await db.execute(f"SELECT * FROM {dataset}")).rows
    count_rows = (await db.execute(f"SELECT count(*) AS count FROM {dataset}")).rows[0][
        "count"
    ]

    features = ", ".join(
        f"'{k}', [{k}]" for k in dict(data_rows[0]).keys() if k != target
    )
    query = f"""
        SELECT sqml_predict_batch(
            '{experiment_name}',
            json_group_array(
                json_object({features})
            )
        ) AS predictions
        FROM {dataset};
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    rows = response.json()
    assert len(rows) == 1

    predictions = json.loads(rows[0]["predictions"])
    assert len(predictions) == count_rows
    for pred in predictions:
        assert isinstance(pred, float)


@pytest.mark.asyncio
async def test_sqml_predict_batch_unknown_experiment(
    datasette: Datasette, faker: Faker
) -> None:
    experiment_name = faker.bs()
    query = f"""
        SELECT sqml_predict_batch(
            '{experiment_name}',
            '[]'
        ) AS prediction;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    rows = response.json()
    assert len(rows) == 1

    prediction = json.loads(rows[0]["prediction"])
    assert "error" in prediction.keys()


@pytest.mark.asyncio
async def test_sqml_predict_batch_no_deployment(
    datasette: Datasette, faker: Faker
) -> None:
    experiment_name = faker.bs()

    db: Database = datasette.get_database("sqml")
    await datasette.invoke_startup()
    await db.execute_write(
        """
        INSERT INTO sqml_experiments(name, prediction_type)
        VALUES (?, ?)
        """,
        (experiment_name, "classification"),
    )

    query = f"""
        SELECT sqml_predict_batch(
            '{experiment_name}',
            '[]'
        ) AS prediction;
        """
    qs = urlencode({"sql": query, "_shape": "array"})
    response = await datasette.client.get(f"/sqml.json?{qs}")
    assert response.status_code == 200

    rows = response.json()
    assert len(rows) == 1

    prediction = json.loads(rows[0]["prediction"])
    assert "error" in prediction.keys()
