from asr.relax import BrokenSymmetryError
from pathlib import Path
import pytest


@pytest.fixture
def mock_GPAW(monkeypatch):
    import numpy as np
    from gpaw import GPAW

    def get_forces(*args, **kwargs):
        return np.zeros((2, 3), float)

    def get_stress(*args, **kwargs):
        return np.zeros((3, 3), float)

    def get_potential_energy(*args, **kwargs):
        return 0

    monkeypatch.setattr(GPAW, 'get_stress', get_stress)
    monkeypatch.setattr(GPAW, 'get_forces', get_forces)
    monkeypatch.setattr(GPAW, 'get_potential_energy', get_potential_energy)


@pytest.mark.parametrize('name', ['Al', 'Cu', 'Ag', 'Au', 'Ni',
                                  'Pd', 'Pt', 'C'])
def test_relax_emt(isolated_filesystem, name):
    from asr.relax import main as relax
    from ase.build import bulk

    unrelaxed = bulk(name)
    unrelaxed.write('unrelaxed.json')
    relax(calculator={'name': 'emt'})


@pytest.mark.parametrize('name', ['Al', 'Cu', 'Ag', 'Au', 'Ni',
                                  'Pd', 'Pt', 'C'])
@pytest.mark.xfail(raises=BrokenSymmetryError)
def test_relax_emt_fail_broken_symmetry(isolated_filesystem, name,
                                        monkeypatch):
    from asr.relax import main as relax
    from ase.build import bulk
    import numpy as np
    from ase.calculators.emt import EMT
    
    unrelaxed = bulk(name)

    def get_stress(*args, **kwargs):
        return np.random.rand(3, 3)

    monkeypatch.setattr(EMT, 'get_stress', get_stress)
    unrelaxed.write('unrelaxed.json')
    relax(calculator={'name': 'emt'}, enforce_symmetry=False)


def test_relax_gpaw_mock(isolated_filesystem, mock_GPAW):
    from asr.setup.materials import main as setupmaterial
    from asr.relax import main as relax
    setupmaterial.cli(["-s", "BN,natoms=2"])
    Path("materials.json").rename("unrelaxed.json")
    relax(calculator={'name': 'gpaw'})


@pytest.mark.slow
def test_relax_si_gpaw(isolated_filesystem):
    from asr.setup.materials import main as setupmaterial
    from asr.relax import main as relax
    from asr.core import read_json

    setupmaterial.cli(["-s", "Si2"])
    Path("materials.json").rename("unrelaxed.json")
    relaxargs = (
        "{'mode':{'ecut':200,'dedecut':'estimate',...},"
        "'kpts':{'density':1,'gamma':True},...}"
    )
    results = relax.cli(["--calculator", relaxargs])
    assert abs(results["c"] - 3.978) < 0.001

    diskresults = read_json("results-asr.relax.json")
    assert results == diskresults


@pytest.mark.slow
def test_relax_bn_gpaw(isolated_filesystem):
    from asr.setup.materials import main as setupmaterial
    from asr.relax import main as relax
    from asr.core import read_json

    setupmaterial.cli(["-s", "BN,natoms=2"])
    Path("materials.json").rename("unrelaxed.json")
    relaxargs = (
        "{'mode':{'ecut':300,'dedecut':'estimate',...},"
        "'kpts':{'density':2,'gamma':True},...}"
    )
    relax.cli(["--calculator", relaxargs])

    results = read_json("results-asr.relax.json")
    assert results["c"] > 5
