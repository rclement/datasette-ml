import sqlite_utils

from pathlib import Path
from datasette_ml.sqml import SQML


def main() -> None:
    db_path = Path(__file__).parent / "sqml.db"
    db_path.unlink(missing_ok=True)
    db = sqlite_utils.Database(db_path)

    sqml = SQML()
    sqml.setup_schema(db.conn)
    sqml.register_functions(db.conn)

    samples_sql = (Path(__file__).parent / "samples.sql").read_text()
    db.executescript(samples_sql)


if __name__ == "__main__":
    main()
