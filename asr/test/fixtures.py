"""Module containing the implementations of all ASR pytest fixtures."""

from ase.parallel import world, broadcast
from asr.core import write_json
from .materials import std_test_materials
import os
import pytest
from _pytest.tmpdir import _mk_tmp
from pathlib import Path


@pytest.fixture()
def mockgpaw(monkeypatch):
    """Fixture that mocks up GPAW."""
    import sys
    monkeypatch.syspath_prepend(Path(__file__).parent.resolve() / "mocks")
    for module in list(sys.modules):
        if "gpaw" in module:
            sys.modules.pop(module)

    yield sys.path

    for module in list(sys.modules):
        if "gpaw" in module:
            sys.modules.pop(module)


@pytest.fixture(params=std_test_materials)
def test_material(request):
    """Fixture that returns an ase.Atoms object representing a std test material."""
    return request.param.copy()


@pytest.fixture()
def asr_tmpdir(request, tmp_path_factory):
    """Create temp folder and change directory to that folder.

    A context manager that creates a temporary folder and changes
    the current working directory to it for isolated filesystem tests.
    """
    if world.rank == 0:
        path = _mk_tmp(request, tmp_path_factory)
    else:
        path = None
    path = broadcast(path)
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(cwd)


def _get_webcontent(name='database.db'):
    from asr.database.fromtree import main as fromtree
    from asr.database.material_fingerprint import main as mf
    from asr.database.app import setup_app

    mf()
    fromtree(recursive=True)
    content = ""

    if world.rank == 0:
        webapp = setup_app()
        webapp.initialize_project(name)

        app = webapp.app

        app.testing = True
        with app.test_client() as c:
            project = webapp.projects["database.db"]
            db = project["database"]
            uid_key = project["uid_key"]
            row = db.get(id=1)
            uid = row.get(uid_key)
            url = f"/database.db/row/{uid}"
            content = c.get(url).data.decode()
            content = (
                content
                .replace("\n", "")
                .replace(" ", "")
            )
    else:
        content = None
    content = broadcast(content)
    return content


@pytest.fixture(autouse=True)
def set_asr_test_environ_variable(monkeypatch):
    monkeypatch.setenv("ASRTESTENV", "true")


@pytest.fixture()
def get_webcontent():
    """Return a utility function that can create and return webcontent."""
    return _get_webcontent


@pytest.fixture()
def asr_tmpdir_w_params(asr_tmpdir):
    """Make temp dir and create a params.json with settings for fast evaluation."""
    params = {
        'asr.gs@calculate': {
            'calculator': {
                "name": "gpaw",
                "kpts": {"density": 2, "gamma": True},
                "xc": "PBE",
            },
        },
        'asr.bandstructure@calculate': {
            'npoints': 10,
            'emptybands': 5,
        },
        'asr.hse@calculate': {
            'kptdensity': 2,
            'emptybands': 5,
        },
        'asr.gw@gs': {
            'kptdensity': 2,
        },
        'asr.bse@calculate': {
            'kptdensity': 2,
        },
        'asr.pdos@calculate': {
            'kptdensity': 2,
            'emptybands': 5,
        },
        'asr.piezoelectrictensor': {
            'calculator': {
                "name": "gpaw",
                "kpts": {"density": 2},
            },
        },
        'asr.formalpolarization': {
            'calculator': {
                "name": "gpaw",
                "kpts": {"density": 2},
            },
        },
    }

    write_json('params.json', params)


@pytest.fixture(params=std_test_materials)
def duplicates_test_db(request, asr_tmpdir):
    """Set up a database containing only duplicates of a material."""
    import numpy as np
    import ase.db

    db = ase.db.connect("duplicates.db")
    atoms = request.param.copy()

    db.write(atoms=atoms)

    rotated_atoms = atoms.copy()
    rotated_atoms.rotate(23, v='z', rotate_cell=True)
    db.write(atoms=rotated_atoms, magstate='FM')

    pbc_c = atoms.get_pbc()
    repeat = np.array([2, 2, 2], int)
    repeat[~pbc_c] = 1
    supercell_ref = atoms.repeat(repeat)
    db.write(supercell_ref)

    translated_atoms = atoms.copy()
    translated_atoms.translate(0.5)
    db.write(translated_atoms)

    rattled_atoms = atoms.copy()
    rattled_atoms.rattle(0.001)
    db.write(rattled_atoms)

    stretch_nonpbc_atoms = atoms.copy()
    cell = stretch_nonpbc_atoms.get_cell()
    pbc_c = atoms.get_pbc()
    cell[~pbc_c][:, ~pbc_c] *= 2
    stretch_nonpbc_atoms.set_cell(cell)
    db.write(stretch_nonpbc_atoms)

    return (atoms, db)


@pytest.fixture()
def crosslinks_test_dbs(asr_tmpdir):
    """Set up database for testing the crosslinks recipe."""
    from pathlib import Path
    from ase.io import write
    from ase.db import connect
    from asr.core import chdir
    from asr.database.material_fingerprint import main as material_fingerprint
    from asr.database.treelinks import main as treelinks
    from asr.database.fromtree import main as fromtree

    write('structure.json', std_test_materials[0])
    p = Path('.')

    for i in range(1, len(std_test_materials)):
        dirpath = Path(p / f'folder_{i - 1}')
        dirpath.mkdir()
        write(dirpath / 'structure.json', std_test_materials[i])

    pathlist = list(p.glob('folder_*'))
    for path in pathlist:
        with chdir(path):
            material_fingerprint()
    material_fingerprint()

    # run asr.database.treelinks to create results and links.json files
    treelinks(include=['folder_*'], exclude=[''])

    # first, create one database based on the tree structure
    fromtree(recursive=True, dbname='db.db')
    # second, create one database with only the parent structure present
    fromtree(recursive=True, dbname='dbref.db')

    # set metadata in such a way that asr.database.crosslinks can work correctly
    db = connect('db.db')
    db.delete([1])
    dbref = connect('dbref.db')
    dbref.delete([2, 3, 4])
    db.metadata = {'title': 'Example DB',
                   'link_name': '{row.formula}-{row.uid}',
                   'link_url': 'test/test/{row.uid}'}
    dbref.metadata = {'title': 'Example Reference DB',
                      'link_name': '{row.uid}-{row.formula}',
                      'link_url': 'testref/testref/{row.uid}'}

    return None
