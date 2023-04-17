import sqlite3
import typing as t

from datasette import hookimpl
from datasette.database import Database

from .sqml import SQML


if t.TYPE_CHECKING:  # pragma: no cover
    from datasette.app import Datasette


sqml = SQML()


@hookimpl
def startup(datasette: "Datasette") -> None:
    config = datasette.plugin_config("datasette-ml") or {}
    db_name = config.get("db", "sqml")
    db: Database = datasette.get_database(db_name)
    sqml.setup_schema(db.connect(True))


@hookimpl
def prepare_connection(
    conn: sqlite3.Connection, database: str, datasette: "Datasette"
) -> None:
    sqml.register_functions(conn)
