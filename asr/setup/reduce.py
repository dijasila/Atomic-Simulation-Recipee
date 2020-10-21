from asr.core import command, option
import numpy as np
from ase.build import cut
from ase.build import niggli_reduce


def check_one_symmetry(spos_ac, ft_c, a_ij, tol=1.e-7):
    """Checks whether atoms satisfy one given symmetry operation."""

    a_a = np.zeros(len(spos_ac), int)
    for a_j in a_ij.values():
        spos_jc = spos_ac[a_j]
        for a in a_j:
            spos_c = spos_ac[a]
            sdiff_jc = spos_c - spos_jc - ft_c
            sdiff_jc -= sdiff_jc.round()
            indices = np.where(abs(sdiff_jc).max(1) < tol)[0]
            if len(indices) == 1:
                j = indices[0]
                a_a[a] = a_j[j]
            else:
                assert len(indices) == 0
                return

    return a_a


def check_if_supercell(spos_ac, Z_a):
    """Check if unit cell can be reduced"""

    a_ij = {}
    for a, Z in enumerate(Z_a):
        if Z in a_ij:
            a_ij[Z].append(a)
        else:
            a_ij[Z] = [a]

    a_j = a_ij[Z_a[0]]  # just pick the first species

    ftrans_sc = spos_ac[a_j[1:]] - spos_ac[a_j[0]]
    ftrans_sc -= np.rint(ftrans_sc)
    for ft_c in ftrans_sc:
        a_a = check_one_symmetry(spos_ac, ft_c, a_ij)
        if a_a is not None:
            return ft_c
        else:
            pass

            
@command('asr.setup.reduce')
@option('--initial', help='Initial atomic structure file')
@option('--final', help='Final atomic structure file')
def main(initial='start.json', final='unrelaxed.json'):
    """Reduce supercell and perform niggli reduction if possible"""

    from ase.io import read
    from ase.build import niggli_reduce

    atoms = read(initial)
    Z_a = atoms.get_atomic_numbers()
    spos_ac = atoms.get_scaled_positions() % 1.0
    ft_c = check_if_supercell(spos_ac, Z_a)
    if ft_c is not None:
        cell_cv = atoms.get_cell()
        cell_cv /= cell_cv.lengths()
        atoms = cut(atoms, a=ft_c, b=cell_cv[1], c=cell_cv[2])

    pbc = atoms.get_pbc()
    atoms.set_pbc(True)
    niggli_reduce(atoms)
    atoms.set_pbc(pbc)

    atoms.write(final)


if __name__ == '__main__':
    main.cli()
