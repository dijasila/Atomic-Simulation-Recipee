import pytest
from ase.build import mx2
from ase.io import write
from asr.defectlinks import (get_list_of_links,
                             get_defectstring_from_defectinfo,
                             get_hostformula_from_defectpath,
                             get_charge_from_folder)
from asr.defect_symmetry import DefectInfo
from pathlib import Path


tokenlist = ['v_S', 'S_Mo', 'S_Mo.Mo_S.1-2', 'v_B.v_X.3-6']


@pytest.mark.parametrize('defect', tokenlist)
@pytest.mark.parametrize('charge', [-1, 0, 1])
@pytest.mark.ci
def test_get_list_of_links(asr_tmpdir, defect, charge):
    basepath = Path('MoS2-XXX-X-X-X/')
    extension = 'defects.MoS2_X.'

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
    linklist = get_list_of_links(systempath)
    for link in linklist:
        assert link[1].startswith(defect.split('_')[0])
        assert f'charge {charge}' in link[1]


@pytest.mark.parametrize('defect', tokenlist)
@pytest.mark.parametrize('charge', [-1, 0, 1])
@pytest.mark.ci
def test_get_defectstring_from_defectinfo(defect, charge):
    defectinfo = DefectInfo(defecttoken=defect)

    defectstring = get_defectstring_from_defectinfo(defectinfo, charge)
    refstring = defectstring
    assert defectstring.startswith(refstring.split('_')[0])
    assert f'charge {charge}' in defectstring


@pytest.mark.parametrize('defect', tokenlist)
@pytest.mark.ci
def test_get_hostformula_from_defectpath(defect):
    basepath = Path('MoS2-XXX-X-X-X/')
    extension = 'defects.MoS2_X.'
    defectpath = Path(basepath / f'{extension}{defect}' / 'charge_0')
    hostformula = get_hostformula_from_defectpath(defectpath)
    assert hostformula.format('reduce') == 'MoS2'


@pytest.mark.parametrize('charge', [-2, -1, 0, 1, 2])
@pytest.mark.ci
def test_get_charge_from_folder(charge):
    basepath = Path('MoS2-XXX-X-X-X/')
    defectpath = Path(basepath / '{extension}X_X' / f'charge_{charge}')
    refcharge = get_charge_from_folder(defectpath)
    assert refcharge == pytest.approx(charge)
