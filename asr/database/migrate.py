"""Database migration module.

This module implements tools for collapsing, converting and migrating
databases.

Significant functions in this module:

  - :func:`collapse_database`
  - :func:`convert_database`
  - :func:`migrate_database`
"""


from pathlib import Path
from typing import Any, Dict

from asr.core.cache import Cache, MemoryBackend
from asr.core.migrate import records_to_migration_report
from asr.core.resultfile import get_resultfile_records_from_database_row
from asr.database import ASEDatabaseInterface, Row, connect
from asr.utils import timed_print


def collapse_database(databasein: str, databaseout: str) -> None:
    """Keep first class materials in databasein, add child data writing to databaseout.

    Take a database with a subset of materials that have been marked as
    "first_class_material"=True and use the children information on these rows
    to include children data. Write only these new first class rows to the
    "collapsed" database.

    Parameters
    ----------
    databasein : str
        Path to input database.
    databaseout : str
        Path to output database.
    """
    assert_databases_not_identical(databasein, databaseout)
    dbin = connect(databasein)
    assert_database_doesnt_exist(databaseout)
    with connect(databaseout) as dbout:
        write_collapsed_database(dbin, dbout)
    copy_database_metadata(dbin, dbout)


def write_collapsed_database(
    dbin: ASEDatabaseInterface, dbout: ASEDatabaseInterface
) -> None:
    for ir, row in enumerate(dbin.select("first_class_material=True")):
        timed_print(f"Treating row #{ir}")
        assert "records" not in row.data
        data = get_data_including_child_data(dbin, row)
        assert "records" not in data
        write_row_with_new_data(dbout, row, data)


def write_row_with_new_data(
    dbout: ASEDatabaseInterface, row: Row, data=None, records=None
) -> None:
    """Write row, data and records to dbout."""
    if data is None:
        data = {}
    dbout.write(
        atoms=row.toatoms(),
        key_value_pairs=row.key_value_pairs,
        records=records,
        data=data,
    )


def get_data_including_child_data(
    dbin: ASEDatabaseInterface, row: Row
) -> Dict[str, Any]:
    """Analyze children data in row and add children data to row.

    Parameters
    ----------
    dbin : ASEDatabaseInterface
        Path to input database.
    row : Row
        Row from which children information is extracted.

    Returns
    -------
    Dict[str, Any]
        Data object that now also contains a key=__children_data__ which
        itself is a dictionary with key = children_material_UID and values
        looking like dict(directory=child_directory, data=child_row.data).

    """
    children = get_children_from_row(row)
    children_data = get_children_data_from_database(dbin, children)
    data = add_children_data(row.data, children_data)
    return data


def copy_database_metadata(
    dbin: ASEDatabaseInterface, dbout: ASEDatabaseInterface
) -> None:
    """Copy metadata from dbin to dbout."""
    dbout.metadata = dbin.metadata


def add_children_data(data, children_data):
    """Add children_data to data under key=__children_data__."""
    data = dict(__children_data__=children_data, **data)
    return data


def get_children_data_from_database(
    dbin: ASEDatabaseInterface, children: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:
    """Make dict that contain data objects from children.

    Parameters
    ----------
    dbin : ASEDatabaseInterface
        Database to retrieve children from
    children : Dict[str, str]
        Dictionary with key=child_directory and value=child_uid.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        A dict where key=child_uid and value=
        dict(directory=child_directory, data=child_row.data).
    """
    children_data = {}
    for child_directory, child_uid in children.items():
        try:
            child_row = dbin.get(uid=child_uid)
        except AssertionError:
            # If there are multiple matching child rows
            # we need to use the one that matches child_directory
            child_rows = list(dbin.select(uid=child_uid))
            child_rows = list(
                filter(lambda x: x.folder.endswith(child_directory), child_rows)
            )
            assert len(child_rows) == 1, "Cannot find matching child rows."
            child_row = child_rows[0]
        children_data[child_uid] = dict(
            directory=child_directory,
            data=child_row.data,
        )

    return children_data


def get_children_from_row(row: Row) -> Dict[str, str]:
    """Get children from data object on row.

    On each first class material row look for __children__ which is a dict that
    maps a folder to a material UID:
    """
    children = row.data.get("__children__", {})
    return children


def convert_database(databasein: str, databaseout: str):
    """Convert resultfiles in database to records and write to new database.

    Takes a database where data is represented by resultfiles and writes a new
    database where the resultfiles has been converted to records.

    Parameters
    ----------
    databasein : str
        Path to input database.
    databaseout : str
        Path to output database.
    """
    assert_databases_not_identical(databasein, databaseout)
    assert_database_doesnt_exist(databaseout)
    dbin = connect(databasein)
    with connect(databaseout) as dbout:
        write_converted_database(dbin, dbout)
    copy_database_metadata(dbin, dbout)


def get_other_data_files_from_row(row: Row) -> Dict[str, Any]:
    """Get files from row data that does are not result-files."""
    data = row.data
    other_data = {}
    for name, value in data.items():
        if not name.startswith("results-"):
            other_data[name] = value
    return other_data


def write_converted_database(
    dbin: ASEDatabaseInterface, dbout: ASEDatabaseInterface
) -> None:
    for row in dbin.select():
        timed_print(f"Treating row.id={row.id}")
        records = get_resultfile_records_from_database_row(row)
        data = get_other_data_files_from_row(row)
        data = {
            key: value
            for key, value in data.items()
            if not key.startswith("__children")
        }
        write_row_with_new_data(dbout, row, data=data, records=records)


def migrate_database(databasein: str, databaseout: str) -> None:
    """Migrate records in database.

    Apply all migrations to the records in all rows of the input database and
    write a new database where rows has been migrated.

    Parameters
    ----------
    databasein : str
        Path to input database.
    databaseout : str
        Path to output database.
    """
    assert_databases_not_identical(databasein, databaseout)
    assert_database_doesnt_exist(databaseout)
    dbin = connect(databasein)
    with connect(databaseout) as dbout:
        write_migrated_database(dbin, dbout)


def assert_databases_not_identical(databasein: str, databaseout: str) -> None:
    assert (
        not databasein == databaseout
    ), "Input and output databases cannot be identical."


def assert_database_doesnt_exist(database: str) -> None:
    assert not Path(database).exists(), f"{database} already exists."


def write_migrated_database(
    dbin: ASEDatabaseInterface, dbout: ASEDatabaseInterface
) -> None:
    for row in dbin.select():
        timed_print(f"Treating row.id={row.id}")
        records = row.records
        cache = Cache(backend=MemoryBackend())
        for record in records:
            cache.add(record)

        report = records_to_migration_report(records)
        if report.n_erroneous_migrations > 0:
            report.print_errors()

        for migration in report.applicable_migrations:
            migration.apply(cache)
        records = cache.select()
        write_row_with_new_data(dbout, row, data=row.data, records=records)
