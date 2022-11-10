import pytest
from .materials import BN, Ag
from pathlib import Path
from asr.defectinfo import (get_nearest_neighbor_distance,
                            get_primitive_pristine_folderpaths,
                            get_defectinfo_from_path)


@pytest.mark.ci
def test_get_nearest_neighbor_distance():
    materials = [BN.copy(), Ag.copy()]
    for atoms in materials:
        cell = atoms.get_cell()
        lengths = cell.lengths()
        distance = get_nearest_neighbor_distance(atoms)

        assert distance == pytest.approx(min(lengths))


@pytest.mark.parametrize('pristine', [True, False])
@pytest.mark.ci
def test_get_primitive_pristine_folderpaths(pristine):

    try:
        primitivepath, pristinepath = get_primitive_pristine_folderpaths(
            Path('.'), pristine)
        refprim = '..'
        refpris = ''
        assert refprim == primitivepath.name
        assert refpris == pristinepath.name
    except IndexError:
        assert not pristine


@pytest.mark.parametrize('defect', ['v_Mo', 'v_S',
                                    'B_N.Se_N.0-3',
                                    'v_Mo.v_S.5-6'])
@pytest.mark.parametrize('charge', [-1, 0, 1, 4])
@pytest.mark.ci
def test_get_defectinfo_from_path(defect, charge):
    path = Path(f'defects.MoS2_331.{defect}/charge_{charge}')
    defectname, chargestate = get_defectinfo_from_path(path)

    assert defectname == defect
    assert chargestate == f'(charge {charge})'
