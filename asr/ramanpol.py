
import numpy as np
import sys
import typing
from pathlib import Path

from ase.io import read
from ase.parallel import parprint
from asr.core import (command, option, DictStr, ASRResult,
                      read_json, write_json, prepare_result)


def webpanel(result, row, key_descriptions):
    from asr.database.browser import fig

    # Make a table from the phonon modes
    data = row.data.get('results-asr.ramanpol.json')
    if data:
        table = []
        freqs_l = data['freqs_l']
        w_l, rep_l = count_deg(freqs_l)
        nph = len(w_l)
        for ii in range(nph):
            key = 'Mode {}'.format(ii + 1)
            table.append(
                (key,
                 np.array2string(
                     np.abs(w_l[ii]), precision=1), rep_l[ii]))
        opt = {'type': 'table',
               'header': ['Mode', 'Frequency (1/cm)', 'Degeneracy'],
               'rows': table}
    else:
        opt = None
    # Make the panel
    panel = {'title': 'Raman spectrum (RPA)',
             'columns': [[fig('Raman.png')], [opt]],
             'plot_descriptions':
                 [{'function': plot_raman, 'filenames': ['Raman.png']}],
             'sort': 22}

    return [panel]


@prepare_result
class Result(ASRResult):

    freqs_l: typing.List[float]
    wavelength_w: typing.List[float]
    amplitudes_vvwl: typing.List[typing.List[typing.List[typing.List[complex]]]]
    irrep_l: typing.List

    key_descriptions = {
        'freqs_l': 'Phonon frequencies (the Gamma point) [1/cm]',
        'wavelength_w': 'Laser excitation wavelength [nm]',
        'amplitudes_vvwl': 'Raman tensor [a.u.]',
        'irrep_l': 'Irreducible representation label',
    }
    formats = {'ase_webpanel': webpanel}


@command(
    'asr.ramanpol',
    requires=['structure.json'],
    returns=ASRResult)
@option('--disp', type=float, help='Displacement size')
@option('--d3/--nod3', is_flag=True, help='Enable DFT-D3 for phonon calculations')
@option('--wavelengths', type=list, help='Excitation wavelengths [nm]')
@option('--eta', type=float, help='Excitation broadening [eV]')
@option('--prefix', type=str, help='Prefix for filenames')
@option('--calc_ph', type=DictStr(), help='Calculator params. for phonon')
@option('--calc_chi', type=DictStr(), help='Calculator params. for chi')
@option('--removefiles', type=str, help='Remove intermediate files')
def main(
    disp: float = 0.05, eta: float = 0.2, d3: bool = True,
    wavelengths: typing.List[float] = [
        488.0, 532.0, 594.0, 612.0, 633.0, 708.0, 780.0,
        850.0, 1064.0, 1550.0, 2600.0, 4800.0, 10600.0],
    removefiles: str = 'chi', prefix: str = '',
    calc_ph: dict = {
        'name': 'gpaw',
        'mode': {'name': 'pw', 'ecut': 800},
        'xc': 'PBE',
        'basis': 'dzp',
        'kpts': {'density': 6.0, 'gamma': True},
        'occupations': {'name': 'fermi-dirac', 'width': 0.05},
        'convergence': {'forces': 1.0e-5},
        'symmetry': {'point_group': False},
        'txt': 'gs_phonons.txt'},
    calc_chi: dict = {
        'name': 'gpaw',
        'mode': {'name': 'pw', 'ecut': 800},
        'xc': 'PBE',
        'basis': 'dzp',
        'kpts': {'density': 15.0, 'gamma': True},
        'occupations': {'name': 'fermi-dirac', 'width': 0.05},
        'nbands': '200%',
        'convergence': {'bands': -5},
        'symmetry': {'point_group': False},
        'txt': 'gs_chi.txt'}, ) -> ASRResult:

    from ase.utils.timing import Timer
    from gpaw.mpi import world
    try:
        # atoms = read('structure.json')
        # import spglib
        # sg = spglib.get_spacegroup((
        #     atoms.cell,
        #     atoms.get_scaled_positions(),
        #     atoms.get_atomic_numbers()), symprec=1e-3)
        # print(sg)
        timer = Timer()
        with timer('Phonon calculations'):
            parprint('Starting a phonon calculation ...')
            if not Path(prefix + 'phonons.json').is_file():
                find_phonons(calculator=calc_ph, dftd3=d3,
                             disp=disp, prefix=prefix)
            ph = read_json(prefix + 'phonons.json')
            freqs_l = ph['freqs_l']
            u_lav = ph['u_lav']
            irrep_l = ph['irrep_l']
            parprint('The phonons are calculated and loaded.')
            if world.rank == 0:
                if removefiles == 'all':
                    for ff in Path().glob(prefix + 'phonons.*.json'):
                        if ff.is_file():
                            ff.unlink()

        with timer('Raman calculations'):
            parprint('Starting a Raman calculation ...')
            freqs_w = [1240 / wavelength for wavelength in wavelengths]
            if not Path(prefix + 'chis.json').is_file():
                get_chis(
                    calc_chi, freqs_l, u_lav, prefix=prefix, eta=eta,
                    removefiles=removefiles, freqs_w=freqs_w, disp=disp)
            parprint('All chi tensors are computed.')
            chis = read_json(prefix + 'chis.json')
            chi_livvw = chis['chi_livvw']
            I_vvwl = np.zeros((3, 3, len(freqs_w), len(freqs_l)), complex)
            for mode, _ in enumerate(freqs_l):
                if mode < 3:
                    continue
                chi_ivvw = chi_livvw[mode - 3]
                alp_vvw = (chi_ivvw[1] - chi_ivvw[0]) / (2 * disp)
                I_vvwl[:, :, :, mode] = alp_vvw
            parprint('The Raman tensors are fully computed.')

        # Make the output data
        results = {
            'amplitudes_vvwl': I_vvwl,
            'wavelength_w': wavelengths,
            'freqs_l': freqs_l,
            'irrep_l': irrep_l, }

    finally:
        if world.rank == 0:
            timer.write()

    return Result.fromdata(**results)


def calcspectrum(wavelength, w_l, I_l, ww, gamma=3, shift=0, temp=300):
    from ase.units import kB

    # Gaussian function definition
    def gauss(w, g):
        gauss = 1 / (g * np.sqrt(2 * np.pi)) * np.exp(-0.5 * w**2 / g**2)
        gauss[gauss < 1e-16] = 0
        return gauss

    cm = 1 / 8065.544
    kbT = kB * temp / cm
    rr = np.zeros(np.size(ww))
    freq = 1240 / wavelength / cm
    # print(freq)
    for wi, ri in zip(w_l, I_l):
        if wi > 1e-1:
            nw = 1 / (np.exp(wi / kbT) - 1)
            curr = (1 + nw) * np.abs(ri)**2 / wi * (freq - wi)**4
            rr = rr + curr * gauss(ww - wi - shift, gamma)
    return rr


def plot_raman(row, filename):

    import matplotlib.pyplot as plt

    # All required settings
    params = {'broadening': 3.0,  # in cm^-1
              'wavelength': 1,  # index
            #   'polarization': ['xx', 'yy', 'zz'],
              'polarization': ['xx', 'yy', 'zz', 'xy', 'xz', 'yz'],
              'temperature': 300}  # in K

    # Read the data from the disk
    data = row.data.get('results-asr.ramanpol.json')

    # Set the variables and parameters
    wavelength_w = data['wavelength_w']
    freqs_l = data['freqs_l']
    amplitudes_vvwl = data['amplitudes_vvwl']
    irrep_l = data['irrep_l']
    selpol = params['polarization']
    gamma = params['broadening']
    waveind = params['wavelength']

    # Check the data to be consistent
    ampshape = amplitudes_vvwl.shape
    freqshape = len(freqs_l)
    waveshape = len(wavelength_w)
    if (ampshape[0] != 3) or (ampshape[1] != 3) or (
            ampshape[2] != waveshape) or (ampshape[3] != freqshape):
        return

    # Make the figure panel and add y=0 axis
    ax = plt.figure().add_subplot(111)
    ax.axhline(y=0, color='k')

    # Make the spectrum
    maxw = min([int(np.max(freqs_l) + 200), int(1.2 * np.max(freqs_l))])
    minw = -maxw / 100
    ww = np.linspace(minw, maxw, 2 * maxw)
    rr = {}

    # I_hh = np.zeros((len(freqs_l)))
    # I_hv = np.zeros((len(freqs_l)))
    # for mode in range(len(freqs_l)):
    #     I_vv = amplitudes_vvwl[:, :, waveind, mode]
    #     ak = 1/3*(I_vv[0, 0]+I_vv[1, 1]+I_vv[2, 2])
    #     bk2 = 1/2*((I_vv[0, 0]-I_vv[1, 1])**2+(I_vv[0, 0]-I_vv[2, 2])**2+(I_vv[1, 1]-I_vv[2, 2])**2) \
    #         +3*(I_vv[0, 1]**2+I_vv[0, 2]**2+I_vv[1, 2]**2)
    #     I_hh[mode] = np.abs(ak**2+4/45*bk2)
    #     I_hv[mode] = 3/45*np.abs(bk2)
    # maxr = np.zeros(2)
    # rr[0] = calcspectrum(
    #     wavelength_w[waveind], freqs_l, I_hh,
    #     ww, gamma=gamma, temp=params['temperature'])
    # maxr[0] = np.max(rr[0])
    # rr[1] = calcspectrum(
    #     wavelength_w[waveind], freqs_l, I_hv,
    #     ww, gamma=gamma, temp=params['temperature'])
    # maxr[1] = np.max(rr[1])
    # ax.plot(ww, rr[0] / np.max(rr[0]), c='C0', label='HH')
    # ax.plot(ww, rr[1] / np.max(rr[1]), c='C1', label='HV')

    maxr = np.zeros(len(selpol))
    for ii, pol in enumerate(selpol):
        d_i = 'xyz'.index(pol[0])
        d_o = 'xyz'.index(pol[1])
        rr[pol] = calcspectrum(
            wavelength_w[waveind], freqs_l, amplitudes_vvwl[d_i, d_o, waveind],
            ww, gamma=gamma, temp=params['temperature'])
        maxr[ii] = np.max(rr[pol])
    for ipol, pol in enumerate(selpol):
        if ipol > 2:
            sty = '--'
        else:
            sty = '-'
        ax.plot(ww, rr[pol] / np.max(maxr), sty, c='C' + str(ipol), label=pol)

    # Add the axis labels
    ax.set_xlabel('Raman shift [cm$^{-1}$]')
    ax.set_ylabel('Raman intensity [a.u.]')
    ax.set_title(f'Excitation wavelength {wavelength_w[waveind]} [nm]')
    ax.set_ylim((-0.1, 1.1))
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlim((minw, maxw))
    # ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))

    # Add the legend to figure
    ax.legend()

    # Count the modes and their degeneracy factors
    w_l, rep_l = count_deg(freqs_l)

    # Add the phonon bars to the figure with showing their degeneracy factors
    for irrep, freq in zip(irrep_l, freqs_l):
        if irrep:
            irrep2 = ''
            for ch in irrep:
                if ch.isdigit() or ch == 'g' or ch == 'u':
                    irrep2 += '$_' + ch + '$'
                else:
                    irrep2 += ch
            plt.text(freq, -0.1, irrep2, ha='center', va='bottom', rotation=0)
    plt.bar(w_l, -0.04, width=maxw / 100, color='k')
    # for idx, rect in enumerate(pltbar):
    #     ax.text(rect.get_x() + rect.get_width() / 2., -0.1,
    #             str(int(rep_l[idx])), ha='center', va='bottom', rotation=0)

    # Remove the extra space and save the figure
    plt.tight_layout()
    plt.savefig(filename)

    # Now make the polarization resolved plot
    # psi = np.linspace(0, 2 * np.pi, 201)
    # ram_par, ram_perp = calc_polarized_raman(
    #     amplitudes_vvwl, freqs_l, waveind,
    #     theta=0.0, phi=0.0,
    #     pte=np.sin(psi), ptm=np.cos(psi))
    # ax = plt.subplot(111, projection='polar')
    # mode = 6
    # ax.plot(psi, np.abs(ram_par[mode]), 'C0', lw=1.0)
    # ax.plot(psi, np.abs(ram_perp[mode]), 'C1', lw=1.0)
    # # Set the y limits
    # ax.grid(True)
    # rmax1 = np.amax(np.abs(ram_par[mode]))
    # rmax2 = np.amax(np.abs(ram_perp[mode]))
    # rmax = max(rmax1, rmax2)
    # if np.abs(rmax) < 1e-6:
    #     rmax = 1e-4
    #     ax.plot(0, 0, 'o', color='b', markersize=5)
    # ax.set_rlim(0, 1.2 * rmax)
    # ax.set_rgrids([rmax], fmt=r'%4.2g')
    # labs = [r'  $\theta=0$', '45', '90', '135', '180', '225', '270', '315']
    # ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], labels=labs)

    # # Put a legend below current axis
    # ax.legend([r'Parallel: |$\chi^{(2)}_{\theta \theta \theta}$|',
    #            r'Perpendicular: |$\chi^{(2)}_{(\theta+90)\theta \theta}$|'],
    #           loc='upper center', bbox_to_anchor=(0.5, -0.15),
    #           fancybox=True, ncol=2)

    # # Remove the extra space and save the figure
    # plt.tight_layout()
    # plt.savefig(filename)

    return ax


def calc_polarized_raman(
        amplitudes_vvwl, freqs_l, wind,
        theta=0.0, phi=0.0,
        pte=[1.0], ptm=[0.0]):

    # Check the input arguments
    pte = np.array(pte)
    ptm = np.array(ptm)
    assert np.all(
        np.abs(pte) ** 2 + np.abs(ptm) ** 2) == 1, \
        '|pte|**2+|ptm|**2 should be one.'
    assert len(pte) == len(ptm), 'Size of pte and ptm should be the same.'

    # Useful variables
    costh = np.cos(theta)
    sinth = np.sin(theta)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    npsi = len(pte)

    # Transfer matrix between (x y z)/(atm ate k) unit vectors basis
    if theta == 0:
        in_bas = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        out_bas = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    else:
        in_bas = [[cosphi * costh, sinphi * costh, -sinth],
                    [-sinphi, cosphi, 0],
                    [sinth * cosphi, sinth * sinphi, costh]]
        out_bas = [[cosphi * costh, sinphi * costh, sinth],
                    [-sinphi, cosphi, 0],
                    [-sinth * cosphi, -sinth * sinphi, costh]]
    in_bas = np.array(in_bas)
    out_bas = np.array(out_bas)

    # in xyz coordinate
    Ein = np.zeros((3, npsi), dtype=complex)
    Epar = np.zeros((3, npsi), dtype=complex)
    Eperp = np.zeros((3, npsi), dtype=complex)
    for ii in range(3):
        Ein[ii] = (pte * in_bas[0][ii] + ptm * in_bas[1][ii])
        Epar[ii] = (pte * out_bas[0][ii] + ptm * out_bas[1][ii])
        Eperp[ii] = (ptm * out_bas[0][ii] - pte * out_bas[1][ii])

    # Compute the Raman
    nph = len(freqs_l)
    ram_par = np.zeros((nph, npsi), complex)
    ram_perp = np.zeros((nph, npsi), complex)
    for ii in range(3):
        for jj in range(3):
            for kk in range(nph): 
                ram_par[kk, :] += amplitudes_vvwl[ii, jj, wind, kk] * Ein[ii, :] * Epar[jj, :]
                ram_perp[kk, :] += amplitudes_vvwl[ii, jj, wind, kk] * Ein[ii, :] * Eperp[jj, :]
    ram_par = np.abs(ram_par)**2
    ram_perp = np.abs(ram_perp)**2

    # Return the output
    return ram_par, ram_perp


def count_deg(freqs_l, freq_err=2):

    # Degeneracy factor for modes
    w_l = [freqs_l[0]]
    rep_l = [1]
    # Loop over modes
    for wss in freqs_l[1:]:
        ind = len(w_l) - 1
        if np.abs(w_l[ind] - wss) > freq_err:
            w_l.append(wss)
            rep_l.append(1)
        else:
            rep_l[ind] += 1
    w_l = np.array(w_l)
    rep_l = np.array(rep_l)
    # Return the output
    return w_l, rep_l


def symmetrize_chi(atoms, chi_vvl):
    import spglib
    sg = spglib.get_symmetry((
        atoms.cell,
        atoms.get_scaled_positions(),
        atoms.get_atomic_numbers()), symprec=1e-3)
    op_scc = sg['rotations']
    cell_cv = atoms.cell
    op_svv = [np.linalg.inv(cell_cv).dot(op_cc.T).dot(cell_cv) for
              op_cc in op_scc]
    nop = len(op_svv)
    sym_chi_vvl = np.zeros_like(chi_vvl)
    for op_vv in op_svv:
        sym_chi_vvl += np.einsum('il,jm,lmn->ijn',
                                 op_vv, op_vv, chi_vvl)

    return sym_chi_vvl / nop


def get_chis(
    calculator, freqs_l, u_lav, removefiles='no', prefix='',
        freqs_w=[0.0, 2.33], eta=0.05, disp=0.05):

    from ase.calculators.calculator import get_calculator_class
    from gpaw.nlopt.matrixel import make_nlodata
    from gpaw.mpi import world
    from gpaw.nlopt.linear import get_chi_tensor

    name = calculator.pop('name')
    calculator['txt'] = prefix + calculator['txt']
    calc = get_calculator_class(name)(**calculator)

    atoms = read('structure.json')
    mass_a = atoms.get_masses()
    set_chi_ivvw = []
    mml_names = []
    for mode, freq in enumerate(freqs_l):
        if mode < 3:
            continue
        chi_ivvw = np.zeros((2, 3, 3, len(freqs_w)), complex)

        # Make the displaced structure
        atoms_new = atoms.copy()
        pos_Nav = atoms_new.get_positions()
        mode_av = np.real(u_lav[mode])
        n_a = np.linalg.norm(mode_av, axis=1)
        mode_av /= np.max(n_a)
        mode_av /= np.sqrt(mass_a)[:, np.newaxis]
        
        for ind in range(2):
            atoms_new.set_positions(pos_Nav + mode_av * disp * (2 * ind - 1))
            sign = ['+', '-'][ind % 2]
            mml_name = prefix + f'mml_{mode}{sign}.npz'
            mml_names.append(mml_name)
            if not Path(mml_name).is_file():
                gs_name = prefix + f'gs_{mode}{sign}.gpw'
                if not Path(gs_name).is_file():
                    parprint(f'Starting GS calculation for M{mode}{sign} ...')
                    atoms_new.calc = calc
                    atoms_new.get_potential_energy()
                    calc.write(gs_name, 'all')
                    parprint(f'The GS calculation for M{mode}{sign} is done.')
                    sys.stdout.flush()

                # Calculate momentum matrix:
                make_nlodata(gs_name=gs_name, out_name=mml_name)

                if removefiles in ['all', 'gs', 'chi']:
                    world.barrier()
                    if world.rank == 0:
                        ff = Path(gs_name)
                        if ff.is_file():
                            ff.unlink()

            parprint(f'Starting a chi calculation for mode {mode}{sign} ...')
            chi_vvw = get_chi_tensor(
                freqs=freqs_w, eta=eta,
                mml_name=mml_name)
            sym_chi_vvw = symmetrize_chi(atoms_new, chi_vvw)
            # sym_chi_vvw = chi_vvw
            parprint(f'The chi calculation for mode {mode}{sign} is done.')
            sys.stdout.flush()

            chi_ivvw[ind] = sym_chi_vvw

        set_chi_ivvw.append(chi_ivvw)

    # Remove extra files
    if removefiles in ['all', 'chi']:
        world.barrier()
        if world.rank == 0:
            for mml_name in mml_names:
                ff = Path(mml_name)
                if ff.is_file():
                    ff.unlink()
    results = {
        'freqs_w': freqs_w,
        'chi_livvw': np.array(set_chi_ivvw),
        'disp': disp}
    write_json(prefix + 'chis.json', results)


def find_phonons(calculator, dftd3=False, disp=0.05, prefix=''):

    from ase.calculators.calculator import get_calculator_class
    from ase.calculators.dftd3 import DFTD3
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.units import THzToCm
    from ase.atoms import Atoms
    from gpaw.mpi import world

    name = calculator.pop('name')
    calculator['txt'] = prefix + calculator['txt']
    calc = get_calculator_class(name)(**calculator)
    if dftd3:
        calc = DFTD3(dft=calc, cutoff=60)

    # Remove empty files:
    if world.rank == 0:
        for ff in Path().glob(prefix + 'phonons.*.json'):
            if ff.stat().st_size == 0:
                ff.unlink()

    # Make the Phonopy object
    atoms = read('structure.json')
    supercell = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    phonopy_atoms = PhonopyAtoms(
        symbols=atoms.symbols,
        cell=atoms.get_cell(),
        scaled_positions=atoms.get_scaled_positions())

    phonon = Phonopy(phonopy_atoms, supercell, symprec=1e-3)
    phonon.generate_displacements(distance=disp, is_plusminus=True)
    displaced_sc = phonon.get_supercells_with_displacements()
    scell = displaced_sc[0]
    atoms_new = Atoms(
        symbols=scell.get_chemical_symbols(),
        scaled_positions=scell.get_scaled_positions(),
        cell=scell.get_cell(),
        pbc=atoms.pbc)

    set_of_forces = []
    for ind, cell in enumerate(displaced_sc):
        # Sign of the displacement
        sign = ['+', '-'][ind % 2]
        filename = prefix + 'phonons.{0}{1}.json'.format(ind // 2, sign)
        if Path(filename).is_file():
            forces = read_json(filename)['force']
        else:
            atoms_new.set_scaled_positions(cell.get_scaled_positions())
            atoms_new.calc = calc
            forces = atoms_new.get_forces()

        drift_force = forces.sum(axis=0)
        for force in forces:
            force -= drift_force / forces.shape[0]
        write_json(filename, {'force': forces})
        set_of_forces.append(forces)

    phonon.produce_force_constants(
        forces=set_of_forces, calculate_full_force_constants=False)
    phonon.symmetrize_force_constants()

    # Get the phonons
    q_c = np.array([0, 0, 0])
    freqs_l, u_ll = phonon.get_frequencies_with_eigenvectors(q_c)
    freqs_l *= THzToCm
    u_lav = u_ll.T.reshape(3 * len(atoms), len(atoms), 3)
    irreps = []
    phonon.set_irreps(q_c)
    ob = phonon._irreps
    for _, (deg, irr) in enumerate(
            zip(ob._degenerate_sets, ob._ir_labels)):
        irreps += [irr] * len(deg)

    results = {
        'freqs_l': freqs_l,
        'u_lav': u_lav,
        'irrep_l': irreps}
    write_json(prefix + 'phonons.json', results)


if __name__ == "__main__":
    main.cli()
