"""Database web application."""
from typing import List
import multiprocessing
import tempfile
from pathlib import Path
import warnings

from flask import render_template, send_file, Response, jsonify, redirect
import flask.json
from jinja2 import UndefinedError
from ase.db import connect
from ase import Atoms
from ase.calculators.calculator import kptdensity2monkhorstpack
from ase.geometry import cell_to_cellpar
from ase.formula import Formula
from ase.db.app import new_app

import asr
from asr.core import (command, option, argument, ASRResult,
                      decode_object, UnknownDataFormat)


def create_key_descriptions(db=None, extra_kvp_descriptions=None):
    from asr.database.key_descriptions import key_descriptions
    from asr.database.fromtree import parse_key_descriptions
    from asr.core import read_json
    from ase.db.web import create_key_descriptions

    flatten = {key: value
               for recipe, dct in key_descriptions.items()
               for key, value in dct.items()}

    if extra_kvp_descriptions is not None and Path(extra_kvp_descriptions).is_file():
        extras = read_json(extra_kvp_descriptions)
        flatten.update(extras)

    if db is not None:
        metadata = db.metadata
        if 'keys' not in metadata:
            raise KeyError('Missing list of keys for database. '
                           'To fix this either: run database.fromtree again. '
                           'or python -m asr.database.set_metadata DATABASEFILE.')
        keys = metadata.get('keys')
    else:
        keys = list(flatten.keys())

    kd = {}
    for key in keys:
        description = flatten.get(key)
        if description is None:
            warnings.warn(f'Missing key description for {key}')
            continue
        kd[key] = description

    kd = {key: (desc['shortdesc'], desc['longdesc'], desc['units']) for
          key, desc in parse_key_descriptions(kd).items()}

    return create_key_descriptions(kd)


class Summary:
    def __init__(self, row, key_descriptions, create_layout, prefix=''):
        self.row = row

        atoms = Atoms(cell=row.cell, pbc=row.pbc)
        self.size = kptdensity2monkhorstpack(atoms,
                                             kptdensity=1.8,
                                             even=False)

        self.cell = [['{:.3f}'.format(a) for a in axis] for axis in row.cell]
        par = ['{:.3f}'.format(x) for x in cell_to_cellpar(row.cell)]
        self.lengths = par[:3]
        self.angles = par[3:]

        self.stress = row.get('stress')
        if self.stress is not None:
            self.stress = ', '.join('{0:.3f}'.format(s) for s in self.stress)

        self.formula = Formula(
            Formula(row.formula).format('metal')).format('html')

        kd = key_descriptions
        self.layout = create_layout(row, kd, prefix)

        self.dipole = row.get('dipole')
        if self.dipole is not None:
            self.dipole = ', '.join('{0:.3f}'.format(d) for d in self.dipole)

        self.data = row.get('data')
        if self.data:
            self.data = ', '.join(self.data.keys())

        self.constraints = row.get('constraints')
        if self.constraints:
            self.constraints = ', '.join(c.__class__.__name__
                                         for c in self.constraints)


class WebApp:
    def __init__(self, app, projects, tmpdir):
        self.app = app
        self.tmpdir = tmpdir
        self.projects = projects

    def initialize_project(self, database, extra_kvp_descriptions=None,
                           pool=None):
        from asr.database import browser
        from functools import partial

        db = connect(database, serial=True)
        metadata = db.metadata
        name = metadata.get("name", Path(database).name)

        tmpdir = self.tmpdir
        # Make temporary directory
        (tmpdir / name).mkdir()

        def layout(*args, **kwargs):
            return browser.layout(*args, pool=pool, **kwargs)

        metadata = db.metadata

        project = {
            "name": name,
            "title": metadata.get("title", name),
            "key_descriptions": create_key_descriptions(db,
                                                        extra_kvp_descriptions),
            "uid_key": metadata.get("uid", "uid"),
            "database": db,
            "handle_query_function": handle_query,
            "row_to_dict_function": partial(
                row_to_dict, layout_function=layout, tmpdir=tmpdir,
            ),
            "default_columns": metadata.get("default_columns", ["formula", "uid"]),
            "table_template": str(
                metadata.get(
                    "table_template", "asr/database/templates/table.html",
                )
            ),
            "search_template": str(
                metadata.get(
                    "search_template", "asr/database/templates/search.html"
                )
            ),
            "row_template": str(
                metadata.get("row_template", "asr/database/templates/row.html")
            ),
        }

        self.projects[name] = project


def setup_app(route_slash=True):
    # used to cache png-files:
    tmpdir = Path(tempfile.mkdtemp(prefix="asr-app-"))

    path = Path(asr.__file__).parent.parent
    projects = {}
    app = new_app(projects)
    app.jinja_loader.searchpath.append(str(path))

    if route_slash:
        @app.route("/")
        def index():
            return render_template(
                "asr/database/templates/projects.html",
                projects=sorted([
                    (name, proj["title"], proj["database"].count())
                    for name, proj in projects.items()
                ]))

    @app.route("/<project>/file/<uid>/<name>")
    def file(project, uid, name):
        assert project in projects
        path = tmpdir / f"{project}/{uid}-{name}"  # XXXXXXXXXXX
        return send_file(str(path))

    webapp = WebApp(app, projects, tmpdir)
    setup_data_endpoints(webapp)
    return webapp


def setup_data_endpoints(webapp):
    """Set endpoints for downloading data."""
    from ase.io.jsonio import MyEncoder

    projects = webapp.projects
    app = webapp.app
    app.json_provider_class = MyEncoder

    @app.route('/<project_name>/row/<uid>/all_data')
    def get_all_data(project_name: str, uid: str):
        """Show details for one database row."""
        project = projects[project_name]
        uid_key = project['uid_key']
        row = project['database'].get('{uid_key}={uid}'
                                      .format(uid_key=uid_key, uid=uid))
        content = flask.json.dumps(row.data)
        return Response(
            content,
            mimetype='application/json',
            headers={'Content-Disposition':
                     f'attachment;filename={uid}_data.json'})

    @app.route('/<project_name>/row/<uid>/data')
    def show_row_data(project_name: str, uid: str):
        """Show details for one database row."""
        project = projects[project_name]
        uid_key = project['uid_key']
        row = project['database'].get('{uid_key}={uid}'
                                      .format(uid_key=uid_key, uid=uid))
        sorted_data = {key: value for key, value
                       in sorted(row.data.items(), key=lambda x: x[0])}
        return render_template(
            'asr/database/templates/data.html',
            data=sorted_data, uid=uid, project_name=project_name)

    @app.route('/<project_name>/row/<uid>/data/<filename>')
    def get_row_data_file(project_name: str, uid: str, filename: str):
        """Show details for one database row."""
        project = projects[project_name]
        uid_key = project['uid_key']
        row = project['database'].get('{uid_key}={uid}'
                                      .format(uid_key=uid_key, uid=uid))
        try:
            result = decode_object(row.data[filename])
            return render_template(
                'asr/database/templates/result_object.html',
                result=result,
                filename=filename,
                project_name=project_name,
                uid=uid,
            )
        except (UnknownDataFormat, UndefinedError):
            return redirect(f'{filename}/json')

    @app.route('/<project_name>/row/<uid>/data/<filename>/json')
    def get_row_data_file_json(project_name: str, uid: str, filename: str):
        """Show details for one database row."""
        project = projects[project_name]
        uid_key = project['uid_key']
        row = project['database'].get('{uid_key}={uid}'
                                      .format(uid_key=uid_key, uid=uid))
        return jsonify(row.data.get(filename))

    @app.template_filter()
    def asr_sort_key_descriptions(value):
        """Sort column drop down menu."""
        def sort_func(item):
            return item[1][1]

        return sorted(value.items(), key=sort_func)


def handle_query(args):
    return args["query"]


def row_to_dict(row, project, layout_function, tmpdir):
    project_name = project['name']
    uid = row.get(project['uid_key'])
    s = Summary(row,
                create_layout=layout_function,
                key_descriptions=project['key_descriptions'],
                prefix=str(tmpdir / f'{project_name}/{uid}-'))
    return s


@command()
@argument("databases", nargs=-1, type=str)
@option("--host", help="Host address.", type=str)
@option("--test", is_flag=True, help="Test the app.")
@option("--extra_kvp_descriptions", type=str,
        help='File containing extra kvp descriptions for info.json')
def main(databases: List[str], host: str = "0.0.0.0",
         test: bool = False,
         extra_kvp_descriptions: str = 'key_descriptions.json') -> ASRResult:

    # The app uses threads, and we cannot call matplotlib multithreadedly.
    # Therefore we use a multiprocessing pool for the plotting.
    # We could use more cores, but they tend to fail to close
    # correctly on KeyboardInterrupt.
    pool = multiprocessing.Pool(1)
    try:
        _main(databases, host, test, extra_kvp_descriptions, pool)
    finally:
        pool.close()
        pool.join()


def _main(databases, host, test, extra_kvp_descriptions, pool):
    webapp = setup_app()
    projects = webapp.projects
    app = webapp.app

    for database in databases:
        webapp.initialize_project(database, extra_kvp_descriptions, pool)

    if test:
        import traceback
        app.testing = True
        with app.test_client() as c:
            for name in projects:
                print(f'Testing {name}')
                c.get(f'/{name}/').data.decode()
                project = projects[name]
                db = project['database']
                uid_key = project['uid_key']
                n = len(db)
                uids = []
                for row in db.select(include_data=False):
                    uids.append(row.get(uid_key))
                    if len(uids) == n:
                        break
                print(len(uids))

                for i, uid in enumerate(uids):
                    url = f'/{name}/row/{uid}'
                    print(f'\rRows: {i + 1}/{len(uids)} {url}',
                          end='', flush=True)
                    try:
                        c.get(url).data.decode()
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        print()
                        row = db.get(uid=uid)
                        exc = traceback.format_exc()
                        exc += (f'Problem with {uid}: '
                                f'Formula={row.formula} '
                                f'Crystal type={row.crystal_type}\n'
                                + '-' * 20 + '\n')
                        with Path('errors.txt').open(mode='a') as fid:
                            fid.write(exc)
                            print(exc)
    else:
        webapp.app.run(host=host, debug=True)


if __name__ == "__main__":
    main.cli()
