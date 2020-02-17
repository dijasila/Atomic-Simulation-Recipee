import pytest
import os
import numpy as np
import contextlib
from pathlib import Path

from ase import Atoms
from ase.build import bulk


def get_webcontent(name='database.db'):
    from asr.database.fromtree import main as fromtree
    fromtree()

    from asr.database import app as appmodule
    from pathlib import Path
    from asr.database.app import app, initialize_project, projects

    tmpdir = Path("tmp/")
    tmpdir.mkdir()
    appmodule.tmpdir = tmpdir
    initialize_project(name)

    app.testing = True
    with app.test_client() as c:
        content = c.get(f"/database.db/").data.decode()
        assert "Fermi level" in content
        assert "Band gap" in content
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
    return content


@pytest.fixture()
def usemocks(monkeypatch):
    from pathlib import Path
    import sys
    monkeypatch.syspath_prepend(Path(__file__).parent.resolve() / "mocks")
    if 'gpaw' in sys.modules:
        sys.modules.pop('gpaw')
    yield
    if 'gpaw' in sys.modules:
        sys.modules.pop('gpaw')


# Make some 1D, 2D and 3D test materials
Si = bulk("Si")
Ag = bulk("Ag")
Ag2 = bulk("Ag").repeat((2, 1, 1))
Fe = bulk("Fe")
Fe.set_initial_magnetic_moments([1])
abn = 2.51
BN = Atoms(
    "BN",
    scaled_positions=[[0, 0, 0], [1 / 3, 2 / 3, 0]],
    cell=[
        [abn, 0.0, 0.0],
        [-0.5 * abn, np.sqrt(3) / 2 * abn, 0],
        [0.0, 0.0, 15.0],
    ],
    pbc=[True, True, False],
)
Agchain = Atoms(
    "Ag",
    scaled_positions=[[0.5, 0.5, 0]],
    cell=[
        [15.0, 0.0, 0.0],
        [0.0, 15.0, 0.0],
        [0.0, 0.0, 2],
    ],
    pbc=[False, False, True],
)
test_materials = [Si, BN, Agchain]


@contextlib.contextmanager
def create_new_working_directory(path='workdir', unique=False):
    """Changes working directory and returns to previous on exit."""
    i = 0
    if unique:
        while Path(f'{path}-{i}').is_dir():
            i += 1
        path = f'{path}-{i}'

    Path(path).mkdir()
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


@pytest.fixture()
def separate_folder(tmpdir):
    """A context manager that creates a temporary folder and changes
    the current working directory to it for isolated filesystem tests.
    """
    cwd = os.getcwd()
    os.chdir(str(tmpdir))

    try:
        yield create_new_working_directory
    finally:
        os.chdir(cwd)


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers",
        """integration_test: Marks an integration test""",
    )
    config.addinivalue_line(
        "markers",
        """integration_test_gpaw: Marks an integration
        test specifically using gpaw""",
    )
    config.addinivalue_line(
        "markers",
        """ci: Mark a test for running in continuous integration""",
    )
