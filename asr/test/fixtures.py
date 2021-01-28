"""Module containing the implementations of all ASR pytest fixtures."""
import numpy as np
from ase.parallel import world, broadcast
from asr.core import write_json
from .materials import std_test_materials, BN
import os
import pytest
import datetime
from _pytest.tmpdir import _mk_tmp
from pathlib import Path
from asr.core import get_cache
from asr.core.specification import construct_run_spec
from asr.core.record import Record


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


VARIOUS_OBJECT_TYPES = [
    1,
    1.02,
    1 + 1e-15,
    1 + 1j,
    1e20,
    1e-17,
    'a',
    (1, 'a'),
    [1, 'a'],
    [1, (1, 'abc', [1.0, ('a', )])],
    np.array([1.1, 2.0], float),
    BN,
    set(['a', 1, '2']),
    Path('directory1/directory2/file.txt'),
    datetime.datetime.now(),
]


@pytest.fixture
def external_file(asr_tmpdir):
    from asr.core import ExternalFile
    filename = 'somefile.txt'
    Path(filename).write_text('sometext')
    return ExternalFile.fromstr(filename)


@pytest.fixture(params=VARIOUS_OBJECT_TYPES)
def various_object_types(request):
    """Fixture that yield object of different relevant types."""
    return request.param


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
    # from asr.database.material_fingerprint import main as mf

    # mf()
    fromtree(recursive=True)
    content = ""
    from asr.database import app as appmodule
    from pathlib import Path
    if world.rank == 0:
        from asr.database.app import app, initialize_project, projects

        tmpdir = Path("tmp/")
        tmpdir.mkdir()
        appmodule.tmpdir = tmpdir
        initialize_project(name)

        app.testing = True
        with app.test_client() as c:
            project = projects["database.db"]
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
def fast_calc():
    fast_calc = {
        "name": "gpaw",
        "kpts": {"density": 1, "gamma": True},
        "xc": "PBE",
    }
    return fast_calc


@pytest.fixture()
def asr_tmpdir_w_params(asr_tmpdir):
    """Make temp dir and create a params.json with settings for fast evaluation."""
    fast_calc = {
        "name": "gpaw",
        "kpts": {"density": 1, "gamma": True},
        "xc": "PBE",
    }
    params = {
        'asr.gs@calculate': {
            'calculator': fast_calc,
        },
        'asr.gs@main': {
            'calculator': fast_calc,
        },
        'asr.bandstructure@main': {
            'npoints': 10,
            'calculator': fast_calc,
        },
        'asr.hse@main': {
            'calculator': fast_calc,
            'kptdensity': 2,
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
        'asr.piezoelectrictensor@main': {
            'calculator': fast_calc,
            'relaxcalculator': fast_calc
        },
        'asr.formalpolarization@main': {
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


@pytest.fixture
def record(various_object_types):
    run_spec = construct_run_spec(
        name='asr.test',
        parameters={'a': 1},
        version=0,
    )
    run_record = Record(
        run_specification=run_spec,
        result=various_object_types,
    )
    return run_record


@pytest.fixture
def fscache(asr_tmpdir):
    cache = get_cache('filesystem')
    return cache
