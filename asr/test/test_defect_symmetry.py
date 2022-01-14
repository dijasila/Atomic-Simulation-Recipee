import pytest
import numpy as np


@pytest.mark.parametrize('extrinsic', ['NO', 'C', 'Se,Te'])
@pytest.mark.parametrize('intrinsic', [True, False])
@pytest.mark.parametrize('vacancies', [True, False])
@pytest.mark.ci
def test_get_defect_info(asr_tmpdir, extrinsic, intrinsic, vacancies):
    from .materials import BN
    from pathlib import Path
    from ase.io import write
    from asr.defect_symmetry import get_defect_info, is_vacancy
    from asr.setup.defects import main

    atoms = BN.copy()
    write('unrelaxed.json', atoms)
    main(extrinsic=extrinsic,
         intrinsic=intrinsic,
         vacancies=vacancies)
    p = Path('.')
    pathlist = list(p.glob('defects.BN*/charge_0'))
    for path in pathlist:
        defecttype, defectpos = get_defect_info(path)
        string = str(path.absolute()).split('/')[-2].split('.')[-1]
        assert defecttype == string.split('_')[0]
        assert defectpos == string.split('_')[1]
        if string.split('_')[0] == 'v':
            assert is_vacancy(path)


@pytest.mark.ci
def test_get_supercell_shape(asr_tmpdir):
    from .materials import BN
    from asr.defect_symmetry import get_supercell_shape

    atoms = BN.copy()
    for i in range(1, 10):
        for j in range(1, 10):
            pristine = atoms.repeat((i, j, 1))
            N = get_supercell_shape(atoms, pristine)
            assert N == min(i, j)


@pytest.mark.parametrize('is_vacancy', [True, False])
@pytest.mark.ci
def test_conserved_atoms(is_vacancy):
    from asr.defect_symmetry import conserved_atoms
    from .materials import BN

    atoms = BN.copy()
    for i in range(2, 10):
        for j in range(len(atoms) * i):
            supercell = atoms.repeat((i, i, 1))
            if is_vacancy:
                supercell.pop(j)
                print(len(supercell), len(atoms), i)
                assert conserved_atoms(supercell,
                                       atoms,
                                       i,
                                       is_vacancy)
            else:
                supercell.symbols[j] = 'X'
                assert conserved_atoms(supercell,
                                       atoms,
                                       i,
                                       is_vacancy)


@pytest.mark.parametrize('sc_size', [1, 2, 3, 4, 5])
@pytest.mark.ci
def test_compare_structures(sc_size):
    from asr.defect_symmetry import compare_structures
    from .materials import BN

    atoms = BN.copy()

    indices = compare_structures(atoms, atoms, 0.1)
    assert indices == []

    reference = atoms.repeat((sc_size, sc_size, 1))
    indices = compare_structures(atoms, reference, 0.1)

    assert len(indices) == sc_size * sc_size * len(atoms) - 2


@pytest.mark.parametrize('defect', ['v_N', 'B_N',
                                    'v_B', 'N_B'])
@pytest.mark.ci
def test_return_defect_coordinates(defect):
    from .materials import BN
    from pathlib import Path
    from asr.defect_symmetry import return_defect_coordinates

    atoms = BN.copy()
    supercell = atoms.repeat((3, 3, 1))

    for i in range(len(supercell)):
        path = Path(f'defects.BN_331.{defect}/charge_0/')
        system = supercell.copy()
        ref_position = supercell.get_positions()[i]
        if defect.startswith('v'):
            system.pop(i)
            is_vacancy = True
        else:
            system.symbols[i] = 'X'
            is_vacancy = False
        position = return_defect_coordinates(
            system, atoms, supercell, is_vacancy,
            defectpath=path)

        assert pytest.approx(position, ref_position)


@pytest.mark.ci
def test_get_spg_symmetry():
    from .materials import BN, Ag
    from asr.defect_symmetry import get_spg_symmetry

    results = ['D3h', 'Oh']
    for i, atoms in enumerate([BN, Ag]):
        sym = get_spg_symmetry(atoms)
        assert sym == results[i]


@pytest.mark.parametrize('defect', ['v_N', 'N_B'])
@pytest.mark.parametrize('size', [10])
@pytest.mark.ci
def test_get_mapped_structure(asr_tmpdir, size, defect):
    from .materials import BN
    from pathlib import Path
    from ase.io import write, read
    from asr.setup.defects import main as setup
    from asr.defect_symmetry import get_mapped_structure

    atoms = BN.copy()
    write('unrelaxed.json', atoms)
    setup(general_algorithm=size)
    p = Path('.')
    pristine = read('defects.pristine_sc.000/structure.json')
    pathlist = list(p.glob(f'defects.BN*{defect}/charge_0'))
    for path in pathlist:
        unrelaxed = read(path / 'unrelaxed.json')
        structure = unrelaxed.copy()
        structure.rattle()
        _ = get_mapped_structure(
            structure, unrelaxed, atoms, pristine, path)


@pytest.mark.ci
def test_indexlist_cut_atoms():
    from .materials import std_test_materials
    from ase import Atoms
    from asr.defect_symmetry import indexlist_cut_atoms

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


@pytest.mark.ci
def test_check_and_return_input(asr_tmpdir):
    from .materials import std_test_materials
    from pathlib import Path
    from asr.core import chdir
    from ase.io import write
    from asr.defect_symmetry import check_and_return_input

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
