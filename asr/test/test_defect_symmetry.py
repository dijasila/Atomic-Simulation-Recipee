import pytest
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.io import write, read
from .materials import BN, Ag, std_test_materials
from asr.core import chdir
from asr.defect_symmetry import (DefectInfo,
                                 get_supercell_shape,
                                 conserved_atoms,
                                 compare_structures,
                                 return_defect_coordinates,
                                 get_spg_symmetry,
                                 get_mapped_structure,
                                 indexlist_cut_atoms,
                                 WFCubeFile,
                                 check_and_return_input)


@pytest.mark.parametrize('defecttype', ['v', 'i', 'S'])
@pytest.mark.parametrize('defectkind', ['Mo', 'Te'])
@pytest.mark.ci
def test_get_defect_info(asr_tmpdir, defecttype, defectkind):
    def get_defect_path(defecttype, defectkind):
        return Path(f'defects.XXX_000.{defecttype}_{defectkind}/charge_0')
    path = get_defect_path(defecttype, defectkind)
    defectinfo = DefectInfo(defectpath=path)
    assert [f'{defecttype}_{defectkind}'] == defectinfo.names
    assert [0] == defectinfo.specs
    for name in defectinfo.names:
        if defecttype == 'v':
            assert defectinfo.is_vacancy(name)
        else:
            assert not defectinfo.is_vacancy(name)


@pytest.mark.ci
def test_get_supercell_shape(asr_tmpdir):
    atoms = BN.copy()
    for i in range(1, 10):
        for j in range(1, 10):
            pristine = atoms.repeat((i, j, 1))
            N = get_supercell_shape(atoms, pristine)
            assert N == min(i, j)


@pytest.mark.parametrize('tokens',
                         ['v_N.Se_B.1-2',
                          'v_N.v_B.v_X.0-3-4',
                          'Se_B', 'v_S'])
@pytest.mark.ci
def test_number_of_vacancies(tokens):
    tokenlist = tokens.split('.')
    nvacancies = sum(token.startswith('v') for token in tokenlist)

    defectinfo = DefectInfo(defecttoken=tokens)
    assert nvacancies == defectinfo.number_of_vacancies


@pytest.mark.parametrize('Nvac', range(5))
@pytest.mark.ci
def test_conserved_atoms(Nvac):
    atoms = BN.copy()
    for i in range(2, 10):
        for j in range(len(atoms) * i):
            supercell = atoms.repeat((i, i, 1))
            for k in range(Nvac):
                supercell.pop(0)
            assert conserved_atoms(supercell,
                                   atoms,
                                   i,
                                   Nvac)


@pytest.mark.parametrize('sc_size', [1, 2, 3, 4, 5])
@pytest.mark.ci
def test_compare_structures(sc_size):
    atoms = BN.copy()

    indices = compare_structures(atoms, atoms, 0.1)
    assert indices == []

    reference = atoms.repeat((sc_size, sc_size, 1))
    indices = compare_structures(atoms, reference, 0.1)

    assert len(indices) == sc_size * sc_size * len(atoms) - 2


@pytest.mark.parametrize('defecttoken',
                         ['v_Mo',
                          'v_N.Se_S.0-1',
                          'v_Se.v_Se.v_X.2-3-4',
                          'v_B.S_Mo.Te_Se.v_N.5-0-1-2'])
@pytest.mark.ci
def test_return_defect_coordinates(defecttoken):
    atoms = BN.copy()
    supercell = atoms.repeat((3, 3, 1))
    defectinfo = DefectInfo(defecttoken=defecttoken)
    reference = defecttoken.split('.')
    if len(reference) == 1:
        index = 0
    else:
        index = int(reference[-1].split('-')[0])
    position = return_defect_coordinates(
        supercell, defectinfo)
    ref_position = supercell.get_positions()[index]

    assert position == pytest.approx(ref_position)


@pytest.mark.ci
def test_get_spg_symmetry():
    results = ['D3h', 'Oh']
    for i, atoms in enumerate([BN.copy(), Ag.copy()]):
        sym = get_spg_symmetry(atoms)
        assert sym == results[i]


@pytest.mark.xfail
@pytest.mark.parametrize('defect', ['v_N', 'N_B'])
@pytest.mark.parametrize('size', [10])
@pytest.mark.ci
def test_get_mapped_structure(asr_tmpdir, size, defect):
    # XXX port from oldmaster
    from asr.setup.defects import main as setup

    atoms = BN.copy()
    write('unrelaxed.json', atoms)
    setup(general_algorithm=size)
    p = Path('.')
    pristine = read('defects.pristine_sc.000/structure.json')
    pathlist = list(p.glob(f'defects.BN*{defect}/charge_0'))
    for path in pathlist:
        defectinfo = DefectInfo(defectpath=path)
        unrelaxed = read(path / 'unrelaxed.json')
        structure = unrelaxed.copy()
        structure.rattle()
        _ = get_mapped_structure(
            structure, unrelaxed, atoms, pristine, defectinfo)


@pytest.mark.ci
def test_indexlist_cut_atoms():

    threshold = 1.01
    for atoms in std_test_materials:
        struc = atoms.copy()
        indices = indexlist_cut_atoms(atoms, threshold)
        del struc[indices]
        assert len(struc) == len(atoms) - len(indices)

    res = [3, 0]
    for atoms in std_test_materials:
        for i, delta in enumerate([-0.05, 0.05]):
            positions = atoms.get_scaled_positions()
            symbols = atoms.get_chemical_symbols()
            newpos = np.array([[1 + delta, 0.5, 0.5],
                               [0.5, 1 + delta, 0.5],
                               [0.5, 0.5, 1 + delta]])
            symbols.append('X')
            symbols.append('X')
            symbols.append('X')
            positions = np.append(positions, newpos, axis=0)
            newatoms = Atoms(symbols,
                             scaled_positions=positions,
                             cell=atoms.get_cell(),
                             pbc=True)
            indices = indexlist_cut_atoms(newatoms, threshold)
            print(indices)
            del newatoms[indices]
            assert len(newatoms) - res[i] == len(atoms)


@pytest.mark.parametrize('band', [0, 1, 12])
@pytest.mark.parametrize('spin', [0, 1])
@pytest.mark.ci
def test_WFCubeFile(band, spin):

    wfcubefiles = [
        WFCubeFile(spin=spin, band=band),
        WFCubeFile.fromfilename(f'wf.{band}_{spin}.cube')]

    for wfcubefile in wfcubefiles:
        assert wfcubefile.filename == f'wf.{band}_{spin}.cube'
        assert wfcubefile.spin == spin
        assert wfcubefile.band == band


@pytest.mark.ci
def test_check_and_return_input(asr_tmpdir):

    for atoms in std_test_materials:
        folder = f'{atoms.get_chemical_formula()}'
        Path(folder).mkdir()
        with chdir(folder):
            write('primitive.json', atoms)
            supercell = atoms.copy()
            supercell = supercell.repeat((2, 2, 2))
            supercell.pop(0)
            write('supercell_unrelaxed.json', supercell)
            relaxed = supercell.copy()
            relaxed.rattle()
            write('supercell_relaxed.json', relaxed)
            pristine = atoms.repeat((2, 2, 2))
            write('pristine.json', pristine)
            struc, un, prim, pris = check_and_return_input(
                'supercell_relaxed.json',
                'supercell_unrelaxed.json',
                'primitive.json',
                'pristine.json')
            assert un is not None
            struc, un, prim, pris = check_and_return_input(
                'supercell_relaxed.json',
                'NO',
                'primitive.json',
                'pristine.json')
            assert un is None
