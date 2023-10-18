"""Phonon band structure and dynamical stability.

Deprecated: Please use the more efficient and optimized asr.phonopy
recipe for calculating phonon properties instead.

"""
import numpy as np

from ase.io import read
from ase.phonons import Phonons
from ase.parallel import paropen

from asr.utils.symmetry import c2db_symmetry_eps
from asr.core import command, option, ASRResult
from asr.paneldata import PhononResult


@command('asr.phonons',
         requires=['structure.json', 'gs.gpw'],
         dependencies=['asr.gs@calculate'])
@option('-n', help='Supercell size', type=int)
@option('--ecut', help='Energy cutoff', type=float)
@option('--kptdensity', help='Kpoint density', type=float)
@option('--fconverge', help='Force convergence criterium', type=float)
def calculate(n: int = 2, ecut: float = 800, kptdensity: float = 6.0,
              fconverge: float = 1e-4) -> ASRResult:
    """Calculate atomic forces used for phonon spectrum."""
    from asr.calculators import get_calculator
    from gpaw.mpi import world

    atoms = read('structure.json')
    gsold = get_calculator()('gs.gpw', txt=None)

    # Set initial magnetic moments
    from asr.utils import is_magnetic
    if is_magnetic():
        magmoms_m = gsold.get_magnetic_moments()
        # Some calculators return magnetic moments resolved into their
        # cartesian components
        if len(magmoms_m.shape) == 2:
            magmoms_m = np.linalg.norm(magmoms_m, axis=1)
        atoms.set_initial_magnetic_moments(magmoms_m)

    params = gsold.parameters.copy()  # TODO: remove fix density from gs params
    if 'fixdensity' in params:
        params.pop('fixdensity')
    params.update({'mode': {'name': 'pw', 'ecut': ecut},
                   'kpts': {'density': kptdensity, 'gamma': True}})

    # Set essential parameters for phonons
    params['symmetry'] = {'point_group': False}

    # Make sure to converge forces! Can be important
    params['convergence'] = {'forces': fconverge}

    with paropen('phonons.txt', mode='a') as fd:
        params['txt'] = fd
        with get_calculator()(**params) as calc:
            supercell = [n if periodic else 1 for periodic in atoms.pbc]
            p = Phonons(atoms=atoms, calc=calc, supercell=supercell)
            if world.rank == 0:
                p.cache.strip_empties()
            world.barrier()
            p.run()


def requires():
    return ['results-asr.phonons@calculate.json']


@command('asr.phonons',
         requires=requires,
         returns=PhononResult,
         dependencies=['asr.phonons@calculate'])
@option('--mingo/--no-mingo', is_flag=True,
        help='Perform Mingo correction of force constant matrix')
def main(mingo: bool = True) -> PhononResult:
    from asr.core import read_json
    calculateresult = read_json('results-asr.phonons@calculate.json')
    atoms = read('structure.json')
    n = calculateresult.metadata.params['n']
    supercell = [n if periodic else 1 for periodic in atoms.pbc]
    p = Phonons(atoms=atoms, supercell=supercell)
    p.read(symmetrize=0)

    if mingo:
        # We correct the force constant matrix and
        # dynamical matrix
        C_N = mingocorrection(p.C_N, atoms, supercell)
        p.C_N = C_N

        # Calculate dynamical matrix
        D_N = C_N.copy()
        m_a = atoms.get_masses()
        m_inv_x = np.repeat(m_a**-0.5, 3)
        M_inv = np.outer(m_inv_x, m_inv_x)
        for D in D_N:
            D *= M_inv
            p.D_N = D_N

    # First calculate the exactly known q-points
    N_c = p.supercell
    q_qc = np.indices(N_c).reshape(3, -1).T / N_c
    out = p.band_structure(q_qc, modes=True, born=False, verbose=False)
    omega_kl, u_kl = out

    R_cN = p.compute_lattice_vectors()
    eigs = []
    for q_c in q_qc:
        phase_N = np.exp(-2j * np.pi * np.dot(q_c, R_cN))
        C_q = np.sum(phase_N[:, np.newaxis, np.newaxis] * p.C_N, axis=0)
        eigs.append(np.linalg.eigvalsh(C_q))

    eigs = np.array(eigs)
    mineig = np.min(eigs)

    if mineig < -0.01:
        dynamic_stability = 'low'
    else:
        dynamic_stability = 'high'

    results = {'omega_kl': omega_kl,
               'q_qc': q_qc,
               'modes_kl': u_kl,
               'minhessianeig': mineig,
               'dynamic_stability_phonons': dynamic_stability}

    # Next calculate an approximate phonon band structure
    path = atoms.cell.bandpath(npoints=100, pbc=atoms.pbc,
                               eps=c2db_symmetry_eps)
    freqs_kl = p.band_structure(path.kpts, modes=False, born=False,
                                verbose=False)
    results['interp_freqs_kl'] = freqs_kl
    results['path'] = path

    return results


def plot_phonons(row, fname):
    import matplotlib.pyplot as plt

    data = row.data.get('results-asr.phonons.json')
    if data is None:
        return

    omega_kl = data['omega_kl']
    gamma = omega_kl[0]
    fig = plt.figure(figsize=(6.4, 3.9))
    ax = fig.gca()

    x0 = -0.0005  # eV
    for x, color in [(gamma[gamma < x0], 'r'),
                     (gamma[gamma >= x0], 'b')]:
        if len(x) > 0:
            markerline, _, _ = ax.stem(x * 1000, np.ones_like(x), bottom=-1,
                                       markerfmt=color + 'o',
                                       linefmt=color + '-')
            plt.setp(markerline, alpha=0.4)
    ax.set_xlabel(r'phonon frequency at $\Gamma$ [meV]')
    ax.axis(ymin=0.0, ymax=1.3)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def mingocorrection(Cin_NVV, atoms, supercell):
    na = len(atoms)
    nc = np.prod(supercell)
    dimension = nc * na * 3

    Cin = (Cin_NVV.reshape(*supercell, na, 3, na, 3).
           transpose(3, 4, 0, 1, 2, 5, 6))

    C = np.empty((*supercell, na, 3, *supercell, na, 3))

    from itertools import product
    for n1, n2, n3 in product(range(supercell[0]),
                              range(supercell[1]),
                              range(supercell[2])):
        inds1 = (np.arange(supercell[0]) - n1) % supercell[0]
        inds2 = (np.arange(supercell[1]) - n2) % supercell[1]
        inds3 = (np.arange(supercell[2]) - n3) % supercell[2]
        C[n1, n2, n3] = Cin[:, :, inds1][:, :, :, inds2][:, :, :, :, inds3]

    C.shape = (dimension, dimension)
    C += C.T.copy()
    C *= 0.5

    # Mingo correction.
    #
    # See:
    #
    #    Phonon transmission through defects in carbon nanotubes
    #    from first principles
    #
    #    N. Mingo, D. A. Stewart, D. A. Broido, and D. Srivastava
    #    Phys. Rev. B 77, 033418 – Published 30 January 2008
    #    http://dx.doi.org/10.1103/PhysRevB.77.033418

    R_in = np.zeros((dimension, 3))
    for n in range(3):
        R_in[n::3, n] = 1.0
    a_in = -np.dot(C, R_in)
    B_inin = np.zeros((dimension, 3, dimension, 3))
    for i in range(dimension):
        B_inin[i, :, i] = np.dot(R_in.T, C[i, :, np.newaxis]**2 * R_in) / 4
        for j in range(dimension):
            B_inin[i, :, j] += np.outer(R_in[i], R_in[j]).T * C[i, j]**2 / 4

    L_in = np.dot(np.linalg.pinv(B_inin.reshape((dimension * 3,
                                                 dimension * 3))),
                  a_in.reshape((dimension * 3,))).reshape((dimension, 3))
    D_ii = C**2 * (np.dot(L_in, R_in.T) + np.dot(L_in, R_in.T).T) / 4
    C += D_ii

    C.shape = (*supercell, na, 3, *supercell, na, 3)
    Cout = C[0, 0, 0].transpose(2, 3, 4, 0, 1, 5, 6).reshape(nc,
                                                             na * 3,
                                                             na * 3)
    return Cout


if __name__ == '__main__':
    main.cli()
