"""Topological analysis of electronic structure."""
import numpy as np
from asr.core import command, option, ASRResult, prepare_result
from asr.database.browser import (fig, entry_parameter_description,
                                  describe_entry, WebPanel,
                                  make_panel_description, href)


olsen_title = ('T. Olsen et al. Discovering two-dimensional topological '
               'insulators from high-throughput computations. '
               'Phys. Rev. Mater. 3 024005.')
olsen_doi = 'https://doi.org/10.1103/PhysRevMaterials.3.024005'

panel_description = make_panel_description(
    """\
The spectrum was calculated by diagonalizing the Berry phase matrix
obtained by parallel transporting the occupied Bloch states along the
k₀-direction for each value of k₁. The eigenvalues can be interpreted
as the charge centers of hybrid Wannier functions localised in the
0-direction and the colours show the expectation values of spin for
the corresponding Wannier functions. A gapless spectrum is a minimal
requirement for non-trivial topological invariants.
""",
    articles=[href(olsen_title, olsen_doi)],
)


@prepare_result
class CalculateResult(ASRResult):

    phi0_km: np.ndarray
    phi1_km: np.ndarray
    phi2_km: np.ndarray
    phi0_pi_km: np.ndarray
    s0_km: np.ndarray
    s1_km: np.ndarray
    s2_km: np.ndarray
    s0_pi_km: np.ndarray

    key_descriptions = {
        'phi0_km': ('Berry phase spectrum at k_2=0, '
                    'localized along the k_0 direction'),
        'phi1_km': ('Berry phase spectrum at k_0=0, '
                    'localized along the k_1 direction'),
        'phi2_km': ('Berry phase spectrum at k_1=0, '
                    'localized along the k_2 direction'),
        'phi0_pi_km': ('Berry phase spectrum at k_2=pi, '
                       'localized along the k_0 direction'),
        's0_km': ('Expectation value of spin in the easy-axis direction '
                  'for the Berry phases at k_2=0'),
        's1_km': ('Expectation value of spin in the easy-axis direction '
                  'for the Berry phases at k_0=0'),
        's2_km': ('Expectation value of spin in the easy-axis direction '
                  'for the Berry phases at k_1=0'),
        's0_pi_km': ('Expectation value of spin in the easy-axis direction '
                     'for the Berry phases at k_2=pi'),
    }


@command(module='asr.berry',
         requires=['gs.gpw'],
         dependencies=['asr.gs'],
         resources='120:10h',
         returns=CalculateResult)
@option('--gs', help='Ground state', type=str)
@option('--kpar', help='K-points along path', type=int)
@option('--kperp', help='K-points orthogonal to path', type=int)
def calculate(gs: str = 'gs.gpw', kpar: int = 120,
              kperp: int = 7) -> CalculateResult:
    """Calculate ground state on specified k-point grid."""
    import os
    from ase.io import read
    from gpaw import GPAW
    from gpaw.berryphase import parallel_transport
    from gpaw.mpi import world
    from asr.magnetic_anisotropy import get_spin_axis

    atoms = read('structure.json')
    pbc = atoms.pbc.tolist()
    nd = np.sum(pbc)

    """Find the easy axis of magnetic materials"""
    theta, phi = get_spin_axis()

    results = {}
    results['phi0_km'] = None
    results['s0_km'] = None
    results['phi1_km'] = None
    results['s1_km'] = None
    results['phi2_km'] = None
    results['s2_km'] = None
    results['phi0_pi_km'] = None
    results['s0_pi_km'] = None

    if nd == 2:
        calc = GPAW(gs,
                    kpts=(kperp, kpar, 1),
                    fixdensity=True,
                    symmetry='off',
                    txt='gs_berry.txt')
        calc.get_potential_energy()
        calc.write('gs_berry.gpw', mode='all')
        phi_km, s_km = parallel_transport('gs_berry.gpw',
                                          direction=0,
                                          theta=theta,
                                          phi=phi)
        results['phi0_km'] = phi_km
        results['s0_km'] = s_km

        if world.rank == 0:
            os.system('rm gs_berry.gpw')

    elif nd == 3:
        """kx = 0"""
        calc = GPAW(gs,
                    kpts=(1, kperp, kpar),
                    fixdensity=True,
                    symmetry='off',
                    txt='gs_berry.txt')
        calc.get_potential_energy()
        calc.write('gs_berry.gpw', mode='all')
        phi_km, s_km = parallel_transport('gs_berry.gpw',
                                          direction=1,
                                          theta=theta,
                                          phi=phi)
        results['phi1_km'] = phi_km
        results['s1_km'] = s_km

        """ky = 0"""
        calc.set(kpts=(kpar, 1, kperp))
        calc.get_potential_energy()
        calc.write('gs_berry.gpw', mode='all')
        phi_km, s_km = parallel_transport('gs_berry.gpw',
                                          direction=2,
                                          theta=theta,
                                          phi=phi)
        results['phi2_km'] = phi_km
        results['s2_km'] = s_km

        """kz = 0"""
        calc.set(kpts=(kperp, kpar, 1))
        calc.get_potential_energy()
        calc.write('gs_berry.gpw', mode='all')
        phi_km, s_km = parallel_transport('gs_berry.gpw',
                                          direction=0,
                                          theta=theta,
                                          phi=phi)
        results['phi0_km'] = phi_km
        results['s0_km'] = s_km

        r"""kz = \pi"""
        from ase.dft.kpoints import monkhorst_pack
        kpts = monkhorst_pack((kperp, kpar, 1)) + [0, 0, 0.5]
        calc.set(kpts=kpts)
        calc.get_potential_energy()
        calc.write('gs_berry.gpw', mode='all')
        phi_km, s_km = parallel_transport('gs_berry.gpw',
                                          direction=0,
                                          theta=theta,
                                          phi=phi)
        results['phi0_pi_km'] = phi_km
        results['s0_pi_km'] = s_km

        if world.rank == 0:
            os.system('rm gs_berry.gpw')
    else:
        raise NotImplementedError('asr.berry@calculate is not implemented '
                                  'for <2D systems.')

    return CalculateResult(data=results)


def plot_phases(row, f0, f1, f2, fpi):
    import pylab as plt

    results = row.data['results-asr.berry@calculate.json']
    for f, label in [(f0, 0), (f1, 1), (f2, 2), (fpi, '0_pi')]:
        phit_km = results.get(f'phi{label}_km')
        if phit_km is None:
            continue
        St_km = results.get(f's{label}_km')
        if St_km is None:
            continue
        Nk = len(St_km)

        phi_km = np.zeros((len(phit_km) + 1, len(phit_km[0])), float)
        phi_km[1:] = phit_km
        phi_km[0] = phit_km[-1]
        S_km = np.zeros((len(phit_km) + 1, len(phit_km[0])), float)
        S_km[1:] = St_km
        S_km[0] = St_km[-1]
        S_km /= 2

        Nm = len(phi_km[0])
        phi_km = np.tile(phi_km, (1, 2))
        phi_km[:, Nm:] += 2 * np.pi
        S_km = np.tile(S_km, (1, 2))
        Nk = len(S_km)
        Nm = len(phi_km[0])

        shape = S_km.T.shape
        perm = np.argsort(S_km.T, axis=None)
        phi_km = phi_km.T.ravel()[perm].reshape(shape).T
        S_km = S_km.T.ravel()[perm].reshape(shape).T

        plt.figure()
        plt.scatter(np.tile(np.arange(Nk), Nm)[perm],
                    phi_km.T.reshape(-1),
                    cmap=plt.get_cmap('viridis'),
                    c=S_km.T.reshape(-1),
                    s=5,
                    marker='o')

        if 'results-asr.magnetic_anisotropy.json' in row.data:
            anis = row.data['results-asr.magnetic_anisotropy.json']
            dir = anis['spin_axis']
        else:
            dir = 'z'

        cbar = plt.colorbar()
        cbar.set_label(rf'$\langle S_{dir}\rangle/\hbar$', size=16)

        if f == f0:
            plt.title(r'$\tilde k_2=0$', size=22)
            plt.xlabel(r'$\tilde k_1$', size=20)
            plt.ylabel(r'$\gamma_0$', size=20)
        elif f == f1:
            plt.title(r'$\tilde k_0=0$', size=22)
            plt.xlabel(r'$\tilde k_2$', size=20)
            plt.ylabel(r'$\gamma_1$', size=20)
        if f == f2:
            plt.title(r'$\tilde k_1=0$', size=22)
            plt.xlabel(r'$\tilde k_0$', size=20)
            plt.ylabel(r'$\gamma_2$', size=20)
        if f == fpi:
            plt.title(r'$\tilde k_2=\pi$', size=22)
            plt.xlabel(r'$\tilde k_1$', size=20)
            plt.ylabel(r'$\gamma_0$', size=20)
        plt.xticks([0, Nk / 2, Nk],
                   [r'$-\pi$', r'$0$', r'$\pi$'], size=16)
        plt.yticks([0, np.pi, 2 * np.pi], [r'$0$', r'$\pi$', r'$2\pi$'], size=16)
        plt.axis([0, Nk, 0, 2 * np.pi])
        plt.tight_layout()
        plt.savefig(f)


def webpanel(result, row, key_descriptions):
    from asr.utils.hacks import gs_xcname_from_row

    xcname = gs_xcname_from_row(row)
    parameter_description = entry_parameter_description(
        row.data,
        'asr.gs@calculate')
    description = ('Topological invariant characterizing the occupied bands \n\n'
                   + parameter_description)
    datarow = [describe_entry('Band topology', description), result.Topology]

    summary = WebPanel(title='Summary',
                       columns=[[{'type': 'table',
                                  'header': ['Basic properties', ''],
                                  'rows': [datarow]}]])

    basicelec = WebPanel(title=f'Basic electronic properties ({xcname})',
                         columns=[[{'type': 'table',
                                    'header': ['Property', ''],
                                    'rows': [datarow]}]],
                         sort=15)

    berry_phases = WebPanel(
        title=describe_entry('Berry phase', panel_description),
        columns=[[fig('berry-phases0.png'),
                  fig('berry-phases0_pi.png')],
                 [fig('berry-phases1.png'),
                  fig('berry-phases2.png')]],
        plot_descriptions=[{'function': plot_phases,
                            'filenames': ['berry-phases0.png',
                                          'berry-phases1.png',
                                          'berry-phases2.png',
                                          'berry-phases0_pi.png']}])

    return [summary, basicelec, berry_phases]


@prepare_result
class Result(ASRResult):

    Topology: str

    key_descriptions = {'Topology': 'Band topology.'}
    formats = {"ase_webpanel": webpanel}


@command(module='asr.berry',
         requires=['results-asr.berry@calculate.json'],
         dependencies=['asr.berry@calculate'],
         returns=Result)
def main() -> Result:
    from pathlib import Path
    from ase.parallel import paropen

    data = {}
    if Path('topology.dat').is_file():
        f = paropen('topology.dat', 'r')
        top = f.readline()
        f.close()
        data['Topology'] = top
    else:
        f = paropen('topology.dat', 'w')
        print('Not checked!', file=f)
        f.close()
        data['Topology'] = 'Not checked'

    return data


if __name__ == '__main__':
    main.cli()
