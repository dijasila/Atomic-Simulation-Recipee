from asr.core import command, option

tests = [{'cli': ['ase build -x diamond Si structure.json',
                  'asr run setup.strains']}]


def get_relevant_strains(pbc):
    import numpy as np
    if np.sum(pbc) == 3:
        ij = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    elif np.sum(pbc) == 2:
        ij = ((0, 0), (1, 1), (0, 1))
    elif np.sum(pbc) == 1:
        ij = ((2, 2))
    return ij


def get_strained_folder_name(strain_percent, i, j):
    from pathlib import Path
    itov_i = ['x', 'y', 'z']
    name = itov_i[i] + itov_i[j]
    folder = Path(f'strains-{strain_percent}%-{name}/')
    return folder


@command('asr.setup.strains',
         tests=tests)
@option('--strain-percent', help='Strain percentage')
@option('--kptdensity', help='Setup up relax and gs calc with fixed density')
def main(strain_percent=1, kptdensity=6.0):
    from ase.io import read
    import numpy as np
    from asr.setup.params import main as setup_params
    from asr.core import chdir
    from ase.calculators.calculator import kpts2sizeandoffsets

    atoms = read('structure.json')
    ij = get_relevant_strains(atoms.pbc)
    cell_cv = atoms.get_cell()
    size, _ = kpts2sizeandoffsets(density=kptdensity, atoms=atoms)

    nk1, nk2, nk3 = size
    for i, j in ij:
        strain_vv = np.eye(3)
        strain_vv[i, j] = strain_percent / 100.0
        strain_vv = (strain_vv + strain_vv.T) / 2
        strained_cell_cv = np.dot(cell_cv, strain_vv)
        atoms.set_cell(strained_cell_cv, scale_atoms=True)
        folder = get_strained_folder_name(strain_percent, i, j)
        folder.mkdir()
        atoms.write(str(folder / 'unrelaxed.json'))

        with chdir(folder):
            params = ("asr.relax:kpts {'size':[" +
                      '{},{},{}'.format(*size)
                      + ']}').split()
            params.extend(['asr.relax:fixcell', 'True'])
            setup_params(params=params)
