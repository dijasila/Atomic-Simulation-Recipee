import numpy as np
from asr.core import (
    command, ASRResult, prepare_result, atomsopt, calcopt,
)

from ase import Atoms
from ase.io import read

from asr.c2db.gs import calculate as gscalculate


# XXX: Hi Thomas. I removed the calculate step. and simply made it a
# normal function to not have to save the .gpw files. BR. Morten
def calculate(
        atoms,
        calculator,
) -> ASRResult:
    """Calculate two spin configurations."""
    from asr.utils import magnetic_atoms
    result = gscalculate(atoms=atoms, calculator=calculator)
    calc = result.calculation.load(fixdensity=False)
    atoms = calc.atoms

    nd = sum(atoms.pbc)
    if nd != 2:
        raise NotImplementedError('asr.c2db.exchange is only implemented '
                                  'for 2D systems.')

    magnetic = magnetic_atoms(atoms)
    assert sum(magnetic) in [1, 2], \
        ('Cannot handle %d magnetic atoms' % sum(magnetic))
    if sum(magnetic) == 2:
        calc.reset()
        calc.set(txt='gs_2mag.txt')
        atoms.calc = calc
        atoms.get_potential_energy()
        calc.write('gs_2mag.gpw')

        a1, a2 = np.where(magnetic)[0]
        magmoms_i = calc.get_magnetic_moments()
        assert np.round(np.abs(magmoms_i[a1] / magmoms_i[a2]), 1) == 1, \
            'The two magnetic moments differ'
        magmoms_e = np.zeros(len(atoms), float)
        magmoms_e[a1] = np.max(np.abs(magmoms_i))
        if np.sign(magmoms_i[a1]) == np.sign(magmoms_i[a2]):
            magmoms_e[a2] = -magmoms_e[a1]
        else:
            magmoms_e[a2] = magmoms_e[a1]
        atoms.set_initial_magnetic_moments(magmoms_e)
        calc.reset()
        calc.set(txt='exchange.txt')
        atoms.calc = calc
        atoms.get_potential_energy()
        calc.write('exchange.gpw')

    else:
        a1 = np.where(magnetic)[0]
        mag = np.max(np.abs(calc.get_magnetic_moments()))
        magmoms = np.zeros(len(atoms), float)
        magmoms[a1] = mag
        atoms.set_initial_magnetic_moments(magmoms)
        atoms = atoms.repeat((2, 1, 1))
        calc.reset()
        calc.set(txt='gs_2mag.txt')
        atoms.calc = calc
        atoms.get_potential_energy()
        calc.write('gs_2mag.gpw')

        magnetic = magnetic_atoms(atoms)
        a1, a2 = np.where(magnetic)[0]
        magmoms_i = calc.get_magnetic_moments()
        assert np.round(magmoms_i[a1] / magmoms_i[a2], 1) == 1, \
            'The two magnetic moments differ'
        mag = np.max(np.abs(magmoms_i))
        magmoms_e = np.zeros(len(atoms), float)
        magmoms_e[a1] = mag
        magmoms_e[a2] = -mag
        atoms.set_initial_magnetic_moments(magmoms_e)
        calc.reset()
        calc.set(txt='exchange.txt')
        atoms.calc = calc
        atoms.get_potential_energy()
        calc.write('exchange.gpw')


def get_parameters(gs, exchange, txt=False,
                   dis_cut=0.2, line=False, a0=None):
    """Extract Heisenberg parameters."""
    from gpaw import GPAW
    from gpaw.occupations import create_occ_calc
    from gpaw.spinorbit import soc_eigenstates
    from ase.dft.bandgap import bandgap

    calc_gs_2mag = GPAW(gs)
    calc_exchange = GPAW(exchange)
    m_gs = calc_gs_2mag.get_magnetic_moment()
    m_ex = calc_exchange.get_magnetic_moment()
    if np.abs(m_gs) > np.abs(m_ex):
        assert np.abs(m_gs) - np.abs(m_ex) > 0.1, \
            'AFM calculation did not converge to target state'
        calc_fm = calc_gs_2mag
        calc_afm = calc_exchange
    else:
        assert np.abs(m_ex) - np.abs(m_gs) > 0.1, \
            'AFM calculation did not converge to target state'
        calc_afm = calc_gs_2mag
        calc_fm = calc_exchange

    nbands = calc_afm.get_number_of_bands()
    atoms = calc_fm.atoms
    if a0 is None:
        a0 = np.argmax(np.abs(calc_fm.get_magnetic_moments()))
    el = atoms[a0].symbol
    a_i = []
    for i in range(len(atoms)):
        if atoms[i].symbol == el:
            a_i.append(i)
    atoms = atoms[a_i].repeat((3, 3, 1))
    dis_i = atoms.get_distances(0, range(len(atoms)), mic=True)
    dis0 = np.sort(dis_i)[1]
    N = len(np.where(np.sort(dis_i)[1:] / dis0 - 1 < dis_cut)[0])

    E_fm = calc_fm.get_potential_energy() / 2
    E_afm = calc_afm.get_potential_energy() / 2

    gap_fm, p1, p2 = bandgap(calc_fm, output=None)
    gap_afm, p1, p2 = bandgap(calc_afm, output=None)
    if gap_fm > 0 and gap_afm > 0:
        occcalc = create_occ_calc({'name': 'fermi-dirac', 'width': 0.001})
    else:
        occcalc = None

    E0_fm = soc_eigenstates(calc_fm, scale=0.0, occcalc=occcalc, n2=nbands,
                            theta=0, phi=0).calculate_band_energy() / 2

    E0_afm = soc_eigenstates(calc_afm, scale=0.0, occcalc=occcalc, n2=nbands,
                             theta=0, phi=0).calculate_band_energy() / 2

    E_fm_x, E_fm_y, E_fm_z = (
        soc_eigenstates(calc_fm, occcalc=occcalc, n2=nbands,
                        theta=theta, phi=phi).calculate_band_energy() / 2
        for theta, phi in [(90, 0), (90, 90), (0, 0)])

    E_afm_x, E_afm_y, E_afm_z = (
        soc_eigenstates(calc_afm, occcalc=occcalc, n2=nbands,
                        theta=theta, phi=phi).calculate_band_energy() / 2
        for theta, phi in [(90, 0), (90, 90), (0, 0)])

    E_fm_x = (E_fm_x + E_fm_y) / 2
    E_afm_x = (E_afm_x + E_afm_y) / 2

    dE_fm = (E_fm_x - E_fm_z)
    dE_afm = (E_afm_x - E_afm_z)

    S = np.abs(np.round(calc_fm.get_magnetic_moment() / 2))
    S = S / 2
    if S == 0:
        S = 1 / 2

    if line:
        if N == 4:
            N_afm = 2
            N_fm = 2
        elif N == 6:
            N_afm = 4
            N_fm = 2
        else:
            N_afm = N
            N_fm = 0
    else:
        if N == 6:
            N_afm = 4
            N_fm = 2
        elif N == 9:
            N_afm = 3
            N_fm = 6
        else:
            N_afm = N
            N_fm = 0

    J = (E_afm + E_afm_x - E0_afm - E_fm - E_fm_x + E0_fm) / (N_afm * S**2)

    if np.abs(S - 0.5) < 0.1:
        A = 0
        if J > 0 or (N == 4 and line):
            B = dE_fm / (N * S**2)
        else:
            B = -dE_afm / (N_afm - N_fm) / S**2
    else:
        A = dE_fm * (1 - N_fm / N_afm) + dE_afm * (1 + N_fm / N_afm)
        A /= (2 * S - 1) * S
        B = (dE_fm - dE_afm) / (N_afm * S**2)

    return J, A, B, S, N


def webpanel(result, context):
    from asr.database.browser import table, describe_entry, WebPanel

    if not context.is_magnetic:
        return []

    parameter_description = context.parameter_description('asr.c2db.exchange')

    explanation_J = ('The nearest neighbor exchange coupling\n\n'
                     + parameter_description)
    explanation_lam = ('The nearest neighbor isotropic exchange coupling\n\n'
                       + parameter_description)
    explanation_A = ('The single ion anisotropy\n\n'
                     + parameter_description)
    explanation_spin = ('The spin of magnetic atoms\n\n'
                        + parameter_description)
    explanation_N = ('The number of nearest neighbors\n\n'
                     + parameter_description)
    J = describe_entry('J', description=explanation_J)
    lam = describe_entry('lam', description=explanation_lam)
    A = describe_entry('A', description=explanation_A)
    spin = describe_entry('spin', description=explanation_spin)
    N_nn = describe_entry('N_nn', description=explanation_N)

    heisenberg_table = table(result, 'Heisenberg model',
                             [J, lam, A, spin, N_nn],
                             kd=context.descriptions)
    panel = WebPanel(title=f'Basic magnetic properties ({context.xcname})',
                     columns=[[heisenberg_table], []],
                     sort=11)
    return [panel]


@prepare_result
class Result(ASRResult):

    J: float
    A: float
    lam: float
    spin: float
    N_nn: int

    key_descriptions = {
        'J': "Nearest neighbor exchange coupling [meV]",
        'A': "Single-ion anisotropy (out-of-plane) [meV]",
        'lam': "Anisotropic exchange (out-of-plane) [meV]",
        'spin': "Maximum value of S_z at magnetic sites",
        'N_nn': "Number of nearest neighbors",
    }

    formats = {'webpanel2': webpanel}


@command(
    module='asr.c2db.exchange',
)
@atomsopt
@calcopt
def main(
        atoms: Atoms,
        calculator: dict = {**gscalculate.defaults.calculator,
                            'mixer': {'method': 'sum'}},
) -> Result:
    """Extract Heisenberg parameters."""
    # MixerSum is more stable towards finding the desired magnetic state
    calculate(
        atoms=atoms,
        calculator=calculator,
    )
    N_gs = len(atoms)
    N_exchange = len(read('gs_2mag.gpw'))
    if N_gs == N_exchange:
        line = False
    else:
        line = True

    J, A, B, S, N = get_parameters('gs_2mag.gpw', 'exchange.gpw', line=line)

    results = {'J': J * 1000,
               'A': A * 1000,
               'lam': B * 1000,
               'spin': S,
               'N_nn': N}

    return Result(data=results)


if __name__ == '__main__':
    main.cli()
