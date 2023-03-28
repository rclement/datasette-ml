import sqlite3
import typing as t

from pathlib import Path
from datasette import hookimpl
from datasette.database import Database

from .sqml import SQML


if t.TYPE_CHECKING:  # pragma: no cover
    from datasette.app import Datasette


@hookimpl
def startup(datasette: "Datasette") -> t.Callable[[], t.Coroutine[t.Any, t.Any, None]]:
    async def init() -> None:
        config = datasette.plugin_config("datasette-ml") or {}
        db_name = config.get("db", "sqml")
        db: Database = datasette.get_database(db_name)
        schema_sql = (Path(__file__).parent / "sql" / "schema.sql").read_text()
        await db.execute_write_script(schema_sql)

    return init


@hookimpl
def prepare_connection(
    conn: sqlite3.Connection, database: str, datasette: "Datasette"
) -> None:
    db: Database = datasette.get_database("sqml")

    sqml = SQML()
    sqml.register(conn)
    sqml.set_internal_connection(db.connect(True))
