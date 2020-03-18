import json
from pathlib import Path
from ase.io import read
from asr.core import command, option
from math import isclose
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

@command('asr.setup.stacking',
         resources='1:1h')
@option('--distance', type=float,
        help='Initial distance between the monolayers')
def main(distance=12.):
    """
    Creates bilayer structures.
    """
    atom = read('structure.json')

    try:
        magstate = get_magstate(atom)
    except RuntimeError:
        magstate = 'nm'

    setup_rotation(atom, distance)
    print('INFO: finished!')

def get_magstate(atom):
    """
    Obtains the magnetic state of a given atomic structure
    """
    magmom = atom.get_magnetic_moment()
    magmoms = atom.get_magnetic_moments()

    if abs(magmom) > 0.02:
        state = 'fm'
    elif abs(magmom) < 0.02 and abs(magmoms).max() > 0.1:
        state = 'afm'
    else:
        state = 'nm'

    return state


def setup_rotation(atom, distance):
    """
    Analyzes both cell and basis. Checks first which cell we have and rotates
    the structures accordingly. Afterwards, check whether those rotations are
    equivalent with symmetry operations for the spacegroup. If that's not the
    case, keep the rotations and continue.
    """

    cell = atom.get_cell()
    print('INFO: atoms object: {}'.format(atom))
    bravais = cell.get_bravais_lattice()
    print('INFO: Bravais lattice for this structure: {}'.format(bravais))

    name = bravais.lattice_system
    if name == 'cubic':
        rotations = 90
    else:
        rotations = 180

    i = 0
    for rot in range(0,360,rotations):
        print('INFO: rotation {}: {}'.format(i, rot))
        i = i+1
    newstruc = atom.copy()
    newstruc.rotate(rotations, 'z', rotate_cell=False)
    newstruc.wrap()

    newpos = newstruc.get_positions()
    newpos[:,2] = newpos[:,2] + distance
    newstruc.set_positions(newpos)

    newstruc = newstruc + atom

    # cell[:,2] = cell[2,2] + distance
    # newstruc.set_cell(cell)

    from ase.io import write
    write('newstruc.json', newstruc)

    return rot


if __name__ == '__main__':
    main()
