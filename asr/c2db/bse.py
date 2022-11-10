"""Bethe Salpeter absorption spectrum."""
from click import Choice
import typing
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.units import alpha, Ha, Bohr

from asr.core import (
    command, option, file_barrier, ASRResult, prepare_result,
    atomsopt, calcopt, ExternalFile,
)
from asr.database.browser import (
    fig, table, make_panel_description, describe_entry)
from asr.utils.kpts import get_kpts_size

from asr.c2db.gs import calculate as gscalculate
from asr.c2db.gs import main as gsmain
from asr.c2db.magstate import main as calcmagstate

panel_description = make_panel_description(
    """The optical absorption calculated from the Bethe–Salpeter Equation
(BSE). The BSE two-particle Hamiltonian is constructed using the wave functions
from a DFT calculation with the direct band gap adjusted to match the direct
band gap from a G0W0 calculation. Spin–orbit interactions are included.  The
result of the random phase approximation (RPA) with the same direct band gap
adjustment as used for BSE but without spin–orbit interactions, is also shown.
""",
    articles=['C2DB'],
)


@prepare_result
class BSECalculateResult(ASRResult):

    bse_polx: ExternalFile
    bse_poly: ExternalFile
    bse_polz: ExternalFile
    bse_eigx: ExternalFile
    bse_eigy: ExternalFile
    bse_eigz: ExternalFile

    key_descriptions: dict = {}

    for direction in ['x', 'y', 'z']:
        key_descriptions[f'bse_pol{direction}'] = (
            'External file handle for bse polarizability '
            f'for {direction} polarized fields.'
        )
        key_descriptions[f'bse_eig{direction}'] = (
            'External file handle for bse eigenvalues '
            f'for {direction} polarized fields.'
        )


@command()
@atomsopt
@calcopt
@option('--kptdensity', help='K-point density', type=float)
@option('--ecut', help='Plane wave cutoff', type=float)
@option('--nv_s', help='Valence bands included', type=float)
@option('--nc_s', help='Conduction bands included', type=float)
@option('--mode', help='Irreducible response',
        type=Choice(['RPA', 'BSE', 'TDHF']))
@option('--bandfactor', type=int,
        help='Number of unoccupied bands = (#occ. bands) * bandfactor)')
def calculate(
        atoms: Atoms,
        calculator: dict = gscalculate.defaults.calculator,
        kptdensity: float = 20.0,
        ecut: float = 50.0,
        mode: str = 'BSE',
        bandfactor: int = 6,
        nv_s: float = -2.3,
        nc_s: float = 2.3,
) -> ASRResult:
    """Calculate BSE polarizability."""
    from ase.dft.bandgap import bandgap
    from gpaw.mpi import world
    from gpaw.response.bse import BSE
    from gpaw.occupations import FermiDirac

    ND = sum(atoms.pbc)
    if ND == 3:
        eta = 0.1
        kpts = {'density': kptdensity, 'gamma': True, 'even': True}
        truncation = None
    elif ND == 2:
        eta = 0.05
        kpts = get_kpts_size(atoms=atoms, kptdensity=kptdensity)
        truncation = '2D'

    else:
        raise NotImplementedError(
            'asr for BSE not implemented for 0D and 1D structures')

    gsres = gscalculate(atoms=atoms, calculator=calculator)
    calc_gs = gsres.calculation.load()
    spin = calc_gs.get_number_of_spins() == 2
    nval = calc_gs.wfs.nvalence
    nocc = int(nval / 2)
    nbands = bandfactor * nocc
    Nk = len(calc_gs.get_ibz_k_points())
    gap, v, c = bandgap(calc_gs, direct=True, output=None)

    if isinstance(nv_s, float):
        ev = calc_gs.get_eigenvalues(kpt=v[1], spin=v[0])[v[2]]
        nv_sk = np.zeros((spin + 1, Nk), int)
        for s in range(spin + 1):
            for k in range(Nk):
                e_n = calc_gs.get_eigenvalues(kpt=k, spin=s)
                e_n -= ev
                x = e_n[np.where(e_n < 0)]
                x = x[np.where(x > nv_s)]
                nv_sk[s, k] = len(x)
        nv_s = np.max(nv_sk, axis=1)
    if isinstance(nc_s, float):
        ec = calc_gs.get_eigenvalues(kpt=c[1], spin=c[0])[c[2]]
        nc_sk = np.zeros((spin + 1, Nk), int)
        for s in range(spin + 1):
            for k in range(Nk):
                e_n = calc_gs.get_eigenvalues(kpt=k, spin=s)
                e_n -= ec
                x = e_n[np.where(e_n > 0)]
                x = x[np.where(x < nc_s)]
                nc_sk[s, k] = len(x)
        nc_s = np.max(nc_sk, axis=1)

    nv_s = [np.max(nv_s), np.max(nv_s)]
    nc_s = [np.max(nc_s), np.max(nc_s)]

    valence_bands = []
    conduction_bands = []
    for s in range(spin + 1):
        gap, v, c = bandgap(calc_gs, direct=True, spin=s, output=None)
        valence_bands.append(range(c[2] - nv_s[s], c[2]))
        conduction_bands.append(range(c[2], c[2] + nc_s[s]))

    if not Path('gs_bse.gpw').is_file():
        calc = gsres.calculation.load(
            txt='gs_bse.txt',
            fixdensity=True,
            nbands=int(nbands * 1.5),
            convergence={'bands': nbands},
            occupations=FermiDirac(width=1e-4),
            kpts=kpts
        )
        calc.get_potential_energy()
        with file_barrier(['gs_bse.gpw']):
            calc.write('gs_bse.gpw', mode='all')

    world.barrier()

    bse = BSE('gs_bse.gpw',
              spinors=True,
              ecut=ecut,
              valence_bands=valence_bands,
              conduction_bands=conduction_bands,
              nbands=nbands,
              mode=mode,
              truncation=truncation,
              txt='bse.txt')

    w_w = np.linspace(-2.0, 8.0, 10001)

    pbc = atoms.pbc
    w_w, alphax_w = bse.get_polarizability(eta=eta,
                                           filename='bse_polx.csv',
                                           direction=0,
                                           write_eig='bse_eigx.dat',
                                           pbc=pbc,
                                           w_w=w_w)

    w_w, alphay_w = bse.get_polarizability(eta=eta,
                                           filename='bse_poly.csv',
                                           direction=1,
                                           write_eig='bse_eigy.dat',
                                           pbc=pbc,
                                           w_w=w_w)

    w_w, alphaz_w = bse.get_polarizability(eta=eta,
                                           filename='bse_polz.csv',
                                           direction=2,
                                           write_eig='bse_eigz.dat',
                                           pbc=pbc,
                                           w_w=w_w)

    # XXX below cleanup code fails to check whether removal even succeeded!
    # Which it won't.  We need proper mechanisms for these things.
    #
    # if world.rank == 0:
    #    os.system('rm gs_bse.gpw')
    #    os.system('rm gs_nosym.gpw')

    return BSECalculateResult.fromdata(
        bse_polx=ExternalFile.fromstr('bse_polx.csv'),
        bse_poly=ExternalFile.fromstr('bse_poly.csv'),
        bse_polz=ExternalFile.fromstr('bse_polz.csv'),
        bse_eigx=ExternalFile.fromstr('bse_eigx.dat'),
        bse_eigy=ExternalFile.fromstr('bse_eigy.dat'),
        bse_eigz=ExternalFile.fromstr('bse_eigz.dat'),
    )


def absorption(context, filename, direction='x'):
    import matplotlib.pyplot as plt
    dim = context.ndim

    # magstate = context.magstate().result['magstate']
    # gs_result = context.gs_results()

    # gap_dir = gs_result['gap_dir']
    # gap_dir_nosoc = gs_result['gap_dir_nosoc']

    # XXX Not sure what's happening here, we can't just mash gaps
    # into the webpanel and expect the reader to know what we are showing.
    #
    # I'll use the gap from GS until this is resolved.
    #
    # for method in ['_gw', '_hse', '_gllbsc', '']:
    #     gapkey = f'gap_dir{method}'
    #     if gapkey in row:
    #         gap_dir_x = row.get(gapkey)
    #         delta_bse = gap_dir_x - gap_dir
    #         delta_rpa = gap_dir_x - gap_dir_nosoc
    #         break

    # delta_bse = gap_dir_
    # qp_gap = gap_dir + delta_bse

    # if magstate != 'NM':
    #     qp_gap = gap_dir_nosoc + delta_rpa
    #     delta_bse = delta_rpa
    qp_gap = 0.0  # XXX Help
    delta_bse = 0.0  # XXX Help

    ax = plt.figure().add_subplot(111)

    result = context.get_record('asr.c2db.bse').result
    bse_data = np.array(result[f'bse_alpha{direction}_w'])
    wbse_w = bse_data[:, 0] + delta_bse
    if dim == 2:
        sigma_w = -1j * 4 * np.pi * (bse_data[:, 1] + 1j * bse_data[:, 2])
        sigma_w *= wbse_w * alpha / Ha / Bohr
        absbse_w = np.real(sigma_w) * np.abs(2 / (2 + sigma_w))**2 * 100
    else:
        absbse_w = 4 * np.pi * bse_data[:, 2]
    ax.plot(wbse_w, absbse_w, '-', c='0.0', label='BSE')
    xmax = wbse_w[-1]

    # TODO: Sometimes RPA pol doesn't exist, what to do?
    # Answer: Nothing, that's someone else's problem, not asr.c2db.bse.
    #
    # data = row.data.get('results-asr.polarizability.json')
    # if pol_data:
    #    wrpa_w = pol_data['frequencies'] + delta_rpa
    #    wrpa_w = pol_data['frequencies'] + delta_rpa
    #    if dim == 2:
    #        sigma_w = -1j * 4 * np.pi * pol_data[f'alpha{direction}_w']
    #        sigma_w *= wrpa_w * alpha / Ha / Bohr
    #        absrpa_w = np.real(sigma_w) * np.abs(2 / (2 + sigma_w))**2 * 100
    #    else:
    #        absrpa_w = 4 * np.pi * np.imag(pol_data[f'alpha{direction}_w'])
    #    ax.plot(wrpa_w, absrpa_w, '-', c='C0', label='RPA')
    #    ymax = max(np.concatenate([absbse_w[wbse_w < xmax],
    #                               absrpa_w[wrpa_w < xmax]])) * 1.05
    # else:
    ymax = max(absbse_w[wbse_w < xmax]) * 1.05

    ax.plot([qp_gap, qp_gap], [0, ymax], '--', c='0.5',
            label='Direct QP gap')

    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0.0, ymax)
    ax.set_title(f'Polarization: {direction}')
    ax.set_xlabel('Energy [eV]')
    if dim == 2:
        ax.set_ylabel('Absorbance [%]')
    else:
        ax.set_ylabel(r'$\varepsilon(\omega)$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)

    return ax


def webpanel(result, context):
    from functools import partial

    E_B = table(result, 'Property', ['E_B'], context.descriptions)

    if context.ndim == 2:
        funcx = partial(absorption, direction='x')
        funcz = partial(absorption, direction='z')

        panel = {'title': describe_entry('Optical absorption (BSE and RPA)',
                                         panel_description),
                 'columns': [[fig('absx.png'), E_B],
                             [fig('absz.png')]],
                 'plot_descriptions': [{'function': funcx,
                                        'filenames': ['absx.png']},
                                       {'function': funcz,
                                        'filenames': ['absz.png']}]}
    else:
        funcx = partial(absorption, direction='x')
        funcy = partial(absorption, direction='y')
        funcz = partial(absorption, direction='z')

        panel = {'title': 'Optical absorption (BSE and RPA)',
                 'columns': [[fig('absx.png'), fig('absz.png')],
                             [fig('absy.png'), E_B]],
                 'plot_descriptions': [{'function': funcx,
                                        'filenames': ['absx.png']},
                                       {'function': funcy,
                                        'filenames': ['absy.png']},
                                       {'function': funcz,
                                        'filenames': ['absz.png']}]}
    return [panel]


@prepare_result
class Result(ASRResult):

    E_B: float
    bse_alphax_w: typing.List[float]
    bse_alphay_w: typing.List[float]
    bse_alphaz_w: typing.List[float]

    key_descriptions = {
        "E_B": ('The exciton binding energy from the Bethe–Salpeter '
                'equation (BSE) [eV].'),
        'bse_alphax_w': 'BSE polarizability x-direction.',
                        'bse_alphay_w': 'BSE polarizability y-direction.',
                        'bse_alphaz_w': 'BSE polarizability z-direction.'}

    formats = {'webpanel2': webpanel}


@command()
@atomsopt
@calcopt
@option('--kptdensity', help='K-point density', type=float)
@option('--ecut', help='Plane wave cutoff', type=float)
@option('--nv_s', help='Valence bands included', type=float)
@option('--nc_s', help='Conduction bands included', type=float)
@option('--mode', help='Irreducible response',
        type=Choice(['RPA', 'BSE', 'TDHF']))
@option('--bandfactor', type=int,
        help='Number of unoccupied bands = (#occ. bands) * bandfactor)')
def main(
        atoms: Atoms,
        calculator: dict = gscalculate.defaults.calculator,
        kptdensity: float = 6.0,
        ecut: float = 50.0,
        mode: str = 'BSE',
        bandfactor: int = 6,
        nv_s: float = -2.3,
        nc_s: float = 2.3,
) -> Result:

    res = calculate(
        atoms=atoms,
        calculator=calculator,
        kptdensity=kptdensity,
        ecut=ecut,
        mode=mode,
        bandfactor=bandfactor,
        nv_s=nv_s,
        nc_s=nc_s,
    )

    alphax_w = np.loadtxt(res.bse_polx, delimiter=',')
    data = {'bse_alphax_w': alphax_w.astype(np.float32)}

    if Path(res.bse_poly).is_file():
        alphay_w = np.loadtxt(res.bse_poly, delimiter=',')
        data['bse_alphay_w'] = alphay_w.astype(np.float32)
    else:
        data['bse_alphay_w'] = None

    if Path(res.bse_polz).is_file():
        alphaz_w = np.loadtxt(res.bse_polz, delimiter=',')
        data['bse_alphaz_w'] = alphaz_w.astype(np.float32)
    else:
        data['bse_alphaz_w'] = None

    if Path(res.bse_eigx).is_file():
        E = np.loadtxt(res.bse_eigx)[0, 1]

        magstateresults = calcmagstate(
            atoms=atoms, calculator=calculator)
        magstate = magstateresults['magstate']

        gsresults = gsmain(
            atoms=atoms, calculator=calculator)

        if magstate == 'NM':
            E_B = gsresults['gap_dir'] - E
        else:
            E_B = gsresults['gap_dir_nosoc'] - E

        data['E_B'] = E_B
    else:
        data['E_B'] = None

    return Result(data=data)


if __name__ == '__main__':
    main.cli()
