"""Define an object that represents a database project."""
import multiprocessing.pool
import runpy
import typing
from dataclasses import dataclass, field
from pathlib import Path

from ase.db import connect
from ase.db.core import Database

from asr.database.browser import layout

if typing.TYPE_CHECKING:
    from asr.database.key_descriptions import KeyDescriptions


def args2query(args):
    return args["query"]


def row_to_dict(row, project):
    from asr.database.app import Summary

    def create_layout(*args, **kwargs):
        return project.layout_function(*args, pool=project.pool, **kwargs)

    project_name = project["name"]
    uid = row.get(project["uid_key"])
    s = Summary(
        row,
        create_layout=create_layout,
        key_descriptions=project["key_descriptions"],
        prefix=str(project.tmpdir / f"{project_name}/{uid}-"),
    )
    return s




def make_default_key_descriptions(db=None):
    from asr.database.app import create_default_key_descriptions

    return create_default_key_descriptions(db=db)


@dataclass
class DatabaseProject:
    """Class that represents a database project.

    Parameters
    ----------
    name
        The name of the database project.
    title
        The title of the database object.
    database
        A database connection
    key_descriptions
        Key descriptions used by the web application
    uid_key
        Key to be used as unique identifier
    row_to_dict_function
        A function that takes (row, project) as input and produces an
        object (normally a dict) that is handed to the row template
        also specified in this project.
    handle_query_function
        A function that takes a query tuple and returns a query tuple.
        Useful for doing translations when the query uses aliases
        for values, for example to convert stability=low to stability=1.
    default_columns
        Default columns that the application should show on the search page.
    table_template
        Path to the table jinja-template.
        The table template shows the rows of the database.
    row_template
        Path to the row jinja-template.
        The row template is responsible for showing a detailed description
        of a particular row.
    search_template
        Path to the search jinja-template. The search template embeds the table
        template and is responsible for formatting the search field.
    """

    name: str
    title: str
    database: Database
    uid_key: str = "uid"
    key_descriptions: "KeyDescriptions" = field(
        default_factory=make_default_key_descriptions
    )
    tmpdir: typing.Optional[Path] = None
    row_to_dict_function: typing.Callable = row_to_dict
    handle_query_function: typing.Callable = args2query
    default_columns: typing.List[str] = field(
        default_factory=lambda: list(["formula", "id"])
    )
    table_template: str = "asr/database/templates/table.html"
    row_template: str = "asr/database/templates/row.html"
    search_template: str = "asr/database/templates/search.html"
    layout_function: typing.Callable = layout
    pool: typing.Optional[multiprocessing.pool.Pool] = None
    template_search_path: typing.Optional[str] = None

    def __getitem__(self, item):
        return self.__dict__[item]

    @classmethod
    def from_pyfile(cls, path: str) -> "DatabaseProject":
        """Make a database project from a Python file.

        Parameters
        ----------
        path : str
            Path to a Python file that defines some or all of
            the attributes that defines a database project, e.g.
            name=, title=. At a minimum name and database needs
            to be defined.

        Returns
        -------
        DatabaseProject
            A database project constructed from the attributes defined
            in the python file.
        """
        dct = runpy.run_path(str(path))

        values = {}

        KEYS_ALLOWED_FOR_PY_FILE_PROJECT_SPEC = set(
            (
                "name",
                "title",
                "database",
                "key_descriptions",
                "uid_key",
                "handle_query_function",
                "row_to_dict_function",
                "default_columns",
                "table_template",
                "search_template",
                "row_template",
            )
        )
        for key in KEYS_ALLOWED_FOR_PY_FILE_PROJECT_SPEC:
            if key in dct:
                values[key] = dct[key]
        return cls(**dct)

    @classmethod
    def from_database(cls, path: str, pool=None) -> "DatabaseProject":
        db = connect(path, serial=True)
        metadata = db.metadata
        name = metadata.get("name", Path(path).name)

        key_descriptions = make_default_key_descriptions(db)
        title = metadata.get("title", name)
        uid_key = metadata.get("uid", "uid")
        default_columns = metadata.get("default_columns", cls.default_columns)
        table_template = str(metadata.get("table_template", cls.table_template))
        search_template = str(metadata.get("search_template", cls.search_template))
        row_template = str(metadata.get("row_template", cls.row_template))

        return cls(
            name=name,
            title=title,
            key_descriptions=key_descriptions,
            uid_key=uid_key,
            database=db,
            default_columns=default_columns,
            table_template=table_template,
            search_template=search_template,
            row_template=row_template,
            pool=pool,
        )
