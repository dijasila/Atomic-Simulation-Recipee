import pytest
from ase.build import mx2
from ase.io import write
from asr.defectlinks import (get_list_of_links,
                             get_uid_from_fingerprint,
                             get_defectstring_from_defectinfo,
                             get_hostformula_from_defectpath,
                             get_charge_from_folder)
from asr.defect_symmetry import DefectInfo
from pathlib import Path

basepath = Path('MoS2-XXX-X-X-X/')
extension = 'defects.MoS2_X.'


@pytest.mark.parametrize('defect', ['v_S', 'v_Mo', 'S_Mo', 'Mo_S'])
@pytest.mark.parametrize('charge', [-1, 0, 1])
@pytest.mark.ci
def test_get_list_of_links(asr_tmpdir, defect, charge):
    atoms = mx2('MoS2')
    basepath.mkdir()
    defectpath = Path(basepath / f'{extension}{defect}')
    defectpath.mkdir()
    for refcharge in [-2, -1, 0, 1, 2]:
        refpath = Path(defectpath / f'charge_{refcharge}')
        refpath.mkdir()
        write(refpath / 'structure.json', atoms)
    systempath = Path(defectpath / f'charge_{charge}')
    write(systempath / 'structure.json', atoms)
    linklist = get_list_of_links(systempath, charge)
    for link in linklist:
        if defect.startswith('v'):
            refstring = 'V'
        else:
            refstring = defect
        assert link[1].startswith(refstring.split('_')[0])
        assert f'charge {charge}' in link[1]


@pytest.mark.ci
def test_get_uid_from_fingerprint(asr_tmpdir):
    atoms = mx2('MoS2')
    path = Path('.')
    write('structure.json', atoms)
    uid = get_uid_from_fingerprint(path)
    assert uid.startswith(atoms.get_chemical_formula())


@pytest.mark.parametrize('defect', ['v_S', 'v_Mo', 'S_Mo', 'Mo_S'])
@pytest.mark.parametrize('charge', [-1, 0, 1])
@pytest.mark.ci
def test_get_defectstring_from_defectinfo(defect, charge):
    defecttuple = defect.split('_')
    defectinfo = DefectInfo(defecttype=defecttuple[0],
                            defectkind=defecttuple[1])

    defectstring = get_defectstring_from_defectinfo(defectinfo, charge)
    if defectstring.startswith('v'):
        refstring = 'V'
    else:
        refstring = defectstring
    assert defectstring.startswith(refstring.split('_')[0])
    assert f'charge {charge}' in defectstring


@pytest.mark.parametrize('defect', ['v_S', 'v_Mo', 'S_Mo', 'Mo_S'])
@pytest.mark.ci
def test_get_hostformula_from_defectpath(defect):
    defectpath = Path(basepath / f'{extension}{defect}' / 'charge_0')
    get_hostformula_from_defectpath(defectpath)


@pytest.mark.parametrize('charge', [-2, -1, 0, 1, 2])
@pytest.mark.ci
def test_get_charge_from_folder(charge):
    defectpath = Path(basepath / '{extension}X_X' / f'charge_{charge}')
    refcharge = get_charge_from_folder(defectpath)
    assert refcharge == pytest.approx(charge)
