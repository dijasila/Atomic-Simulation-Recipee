import pytest
from pytest import approx
import numpy as np


@pytest.mark.ci
def test_stiffness_gpaw(asr_tmpdir_w_params, mockgpaw, mocker, test_material,
                        fast_calc,
                        get_webcontent):
    from asr.c2db.stiffness import main as stiffness

    strain_percent = 1
    results = stiffness(
        atoms=test_material,
        strain_percent=strain_percent,
        calculator=fast_calc,
    )
    nd = np.sum(test_material.pbc)

    # check that all keys are in results-asr.c2db.stiffness.json:
    keys = ['stiffness_tensor', 'eigenvalues']
    if nd == 2:
        keys.extend(['speed_of_sound_x', 'speed_of_sound_y',
                     'c_11', 'c_22', 'c_33', 'c_23', 'c_13', 'c_12'])
    for key in keys:
        assert key in results

    if nd == 1:
        stiffness_tensor = 0.
        eigenvalues = 0.
    elif nd == 2:
        stiffness_tensor = np.zeros((3, 3))
        eigenvalues = np.zeros(3)
    else:
        stiffness_tensor = np.zeros((6, 6))
        eigenvalues = np.zeros(6)

    assert results['stiffness_tensor'] == approx(stiffness_tensor)
    assert results['eigenvalues'] == approx(eigenvalues)

    test_material.write('structure.json')
    content = get_webcontent()
    assert 'Dynamical(stiffness)' in content, content


@pytest.mark.ci
# @pytest.mark.parametrize('name', ['Al', 'Cu', 'Ag', 'Au', 'Ni',
#                                   'Pd', 'Pt', 'C'])
@pytest.mark.parametrize('name', ['Al'])
def test_stiffness_emt(asr_tmpdir_w_params, name, mockgpaw, get_webcontent):
    from ase.build import bulk
    from asr.c2db.stiffness import main as stiffness
    atoms = bulk(name)
    atoms.set_initial_magnetic_moments([0.0] * len(atoms))
    atoms.write('structure.json')

    result = stiffness(atoms=atoms, calculator=dict(name='emt'))
    stiffness_tensor = result['stiffness_tensor']
    assert stiffness_tensor == approx(stiffness_tensor.T, abs=1)

    content = get_webcontent()
    assert 'Dynamical(stiffness)' in content, content
