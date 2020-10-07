from asr.core import command, option, read_json
import ase.units as units
from math import sqrt
import numpy as np


def get_wfs_overlap(i, f, calc_0, calc_q):
    """ Calculate the overlap between that the state i
        and the state f of the displaced geometry"""

    from gpaw.utilities.ps2ae import PS2AE
    from ase.units import Bohr

    wfs_0 = PS2AE(calc_0, h=0.05)
    wfs_q = PS2AE(calc_q, h=0.05)

    wf_0 = wfs_0.get_wave_function(i)
    wf_q = wfs_q.get_wave_function(f)

    overlap = wfs_q.gd.integrate(wf_0 * wf_q) * Bohr**3

    ei = calc_q.get_eigenvalues(band=i, spin=0)
    ej = calc_q.get_eigenvalues(band=j, spin=0)

    eigenvalues = [ei, ej]

    return overlap, eigenvalues


@command("asr.config_diagram")
@option('--folder', help='Folder of the displaced geometry', type=str)
@option('--npoints', help='How many displacement points.', type=int)
@option('--wfs', nargs=2, type=int,
        help='Calculate the overlap of wfs between states i and f')
def calculate(folder: str, npoints: int = 5, overlap: List[int] = None]):
    """Interpolate the geometry of the structure in the current
       folder with the displaced geometry in the 'folder' given
       as input of the recipe.
       The number of displacements between the two geometries
       is set with the 'npoints' input, and the energy, the
       modulus of the displacement and the overlap between the
       wavefunctions is saved."""

    from gpaw import GPAW, restart

    atoms, calc_0 = restart('gs.gpw', txt=None)
    atoms_Q, calc_Q = restart(folder + '/gs.gpw', txt=None)
    params = calc.todict()
    calc_q = GPAW(**params)
    calc_q.set(txt='cc-diagram.txt')

    displ_n = np.linspace(-1.0, 1.0, npoints, endpoint=True)
    m_a = atoms.get_masses()
    pos_ai = atoms.positions.copy()

    delta_R = atoms_Q.positions - atoms.positions
    delta_Q = sqrt(((delta_R**2).sum(axis=-1) * m_a).sum())
    # check if there is difference in the two geometries
    assert delta_Q >= 0.005, 'No displacement between the two geometries!' 
    print('delta_Q', delta_Q)

    # calculate zero-phonon line
    zpl = abs(atoms.get_potential_energy() - atoms_Q.get_potential_energy())

    Q_n = []
    energies_n = []
    overlap_n = []
    eigenvalues_n = []

    for displ in displ_n:
        Q2 = (((displ * delta_R)**2).sum(axis=-1) * m_a).sum()
        Q_n.append(sqrt(Q2) * np.sign(displ))
        atoms.positions += displ * delta_R
        energy = atoms.get_potential_energy()
        print(displ, energy)

        atoms.positions = pos_ai
        #energy = 0.06155**2 / 2 * 15.4669**2 * Q2
        energies_n.append(energy)

        if wfs is not None:
            i, f = wfs
            overlap, eigenvalues = get_wfs_overlap(i, f, calc_0, calc_q) 
            overlap_n.append(overlap)
            eigenvalues_n.append(eigenvalues)
            

    results = {'delta_Q': delta_Q,
               'Q_n': Q_n,
               'energies_n': energies_n,
               'ZPL': zpl}

    if wfs is not None:
        results['overlap'] = overlap_n
        results['eigenvalues'] = eigenvalues_n

    return results


def webpanel(row, key_descriptions):
    from asr.database.browser import fig

    panel = {'title': 'Configuration coordinate diagram',
             'columns': [[fig('cc_diagram.png')]],
             'plot_descriptions': [{'function': plot_cc_diagram,
                                    'filenames': ['cc_diagram.png']}],
             'sort': 13}

    return [panel]


@command("asr.config_diagram",
         webpanel=webpanel)#,
         #dependencies=["asr.config_diagram@calculate"])
@option('--folder1', help='Folder of the first parabola', type=str)
@option('--folder2', help='Folder of the first parabola', type=str)
def main(folder1: str = '.', folder2: str = 'excited'):
    """Estrapolate the frequencies of the ground and
       excited one-dimensional mode and their relative
       Huang-Rhys factors"""

    result_file = 'results-asr.config_diagram@calculate.json' 

    data_g = read_json(folder1 + '/' + result_file)
    data_e = read_json(folder2 + '/' + result_file)
    delta_Q = data_g['delta_Q']
    energies_gn = data_g['energies_n']
    energies_en = data_e['energies_n']
    Q_n = data_g['Q_n']
    zpl = data_g['ZPL']

    # Rescale ground energies by the minimum value
    energies_gn = np.array(energies_gn)
    energies_gn -= np.min(energies_gn)
    # Rescale excited energies by the minimum value
    energies_en = np.array(energies_en)
    energies_en -= np.min(energies_en)

    # Quadratic fit of the parabola
    zg = np.polyfit(Q_n, energies_gn, 2)
    ze = np.polyfit(Q_n, energies_en, 2)
    # Conversion factor
    s = np.sqrt(units._e * units._amu) * 1e-10 / units._hbar

    # Estrapolation of the effective frequencies
    omega_g = sqrt(2 * zg[0] / s**2)
    omega_e = sqrt(2 * ze[0] / s**2)

    # Estrapolation of the Huang-Rhys factors
    S_g = s**2 * delta_Q**2 * omega_g / 2
    S_e = s**2 * delta_Q**2 * omega_e / 2

    ground = {'energies_n': energies_gn,
              'omega': omega_g,
              'S': S_g}

    excited = {'energies_n': energies_en,
               'omega': omega_e,
               'S': S_e}

    results = {'Q_n': Q_n,
               'ZPL': zpl,
               'delta_Q': delta_Q,
               'ground': ground,
               'excited': excited}

    return results


def plot_cc_diagram(row, fname):
    from matplotlib import pyplot as plt

    data = row.data.get('results-asr.config_diagram.json')
    data_g = data['ground']
    data_e = data['excited']

    ene_g = data_g['energies_n']
    ene_e = data_e['energies_n']

    Q_n = np.array(data['Q_n'])
    ZPL = data['ZPL']
    delta_Q = data['delta_Q']
    q = np.linspace(Q_n[0] - delta_Q * 0.2, Q_n[-1] + delta_Q * 0.2, 100)

    omega_g = data_g['omega']
    omega_e = data_e['omega']

    s = np.sqrt(units._e * units._amu) * 1e-10 / units._hbar

    fig = plt.figure(figsize=(7, 6))
    ax = fig.gca()

    # Helping lines
    ax.plot(q, 1 / 2 * omega_g**2 * s**2 * q**2, '-C0')
    ax.plot(Q_n, ene_g, 'wo', ms=7, markeredgecolor='C0', markeredgewidth=0.9)
    # Ground state parabola
    ax.plot(q, 1 / 2 * omega_g**2 * s**2 * q**2, '-C0')
    ax.plot(Q_n, ene_g, 'wo', ms=7, markeredgecolor='C0', markeredgewidth=0.9)
    # Excited state parabola
    ax.plot(q + delta_Q, 1 / 2 * omega_e**2 * s**2 * q**2 + ZPL, '-C1')
    ax.plot(Q_n + delta_Q, ene_e + ZPL, 'wo', ms=7,
            markeredgecolor='C1', markeredgewidth=0.9)

    ax.set_xlabel(r'Q$\;(amu^{1/2}\AA)$', size=14)
    ax.set_ylabel('Energy (eV)', size=14)
    ax.set_xlim(-1.3 * delta_Q, 2 * delta_Q * 1.15)
    ax.set_ylim(-1 / 5 * ZPL, 1.1 * max(ene_e) + 6 / 5 * ZPL)

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


if __name__ == '__main__':
    main.cli()
