"""Optical polarizability."""
import typing
from pathlib import Path

from click import Choice
import numpy as np
from ase.io import read

from asr.core import command, option, ASRResult, prepare_result
from asr.database.browser import (
    table,
    fig,
    describe_entry,
    make_panel_description)
from asr.utils.kpts import get_kpts_size


panel_description = make_panel_description(
    """The frequency-dependent polarisability in the long wave length limit (q=0)
calculated in the random phase approximation (RPA) without spin–orbit
interactions. For metals a Drude term accounts for intraband transitions. The
contribution from polar lattice vibrations is added (see infrared
polarisability) and may be visible at low frequencies.""",
    articles=['C2DB'],
)


def webpanel(result, row, key_descriptions):
    explanation = 'Static interband polarizability along the'
    alphax_el = describe_entry('alphax_el', description=explanation + " x-direction")
    alphay_el = describe_entry('alphay_el', description=explanation + " y-direction")
    alphaz_el = describe_entry('alphaz_el', description=explanation + " z-direction")

    explanation = 'Static lattice polarizability along the'
    alphax_lat = describe_entry('alphax_lat', description=explanation + " x-direction")
    alphay_lat = describe_entry('alphay_lat', description=explanation + " y-direction")
    alphaz_lat = describe_entry('alphaz_lat', description=explanation + " z-direction")

    explanation = 'Total static polarizability along the'
    alphax = describe_entry('alphax', description=explanation + " x-direction")
    alphay = describe_entry('alphay', description=explanation + " y-direction")
    alphaz = describe_entry('alphaz', description=explanation + " z-direction")

    opt = table(row, 'Property', [
        alphax_el, alphay_el, alphaz_el,
        alphax_lat, alphay_lat, alphaz_lat,
        alphax, alphay, alphaz,
    ], key_descriptions)

    panel = {'title': describe_entry('Optical polarizability (RPA)',
                                     panel_description),
             'columns': [[fig('rpa-pol-x.png'), fig('rpa-pol-z.png')],
                         [fig('rpa-pol-y.png'), opt]],
             'plot_descriptions':
                 [{'function': polarizability,
                   'filenames': ['rpa-pol-x.png',
                                 'rpa-pol-y.png',
                                 'rpa-pol-z.png']}],
             'sort': 20}

    return [panel]


@prepare_result
class Result(ASRResult):
    alphax_el: typing.List[complex]
    alphay_el: typing.List[complex]
    alphaz_el: typing.List[complex]
    alphax_w: typing.List[complex]
    alphay_w: typing.List[complex]
    alphaz_w: typing.List[complex]
    alpha0x_w: typing.List[complex]
    alpha0y_w: typing.List[complex]
    alpha0z_w: typing.List[complex]
    plasmafreq_vv: typing.List[typing.List[float]]
    frequencies: typing.List[float]

    key_descriptions = {
        "alphax_el": "Static interband polarizability (x) [Ang]",
        "alphay_el": "Static interband polarizability (y) [Ang]",
        "alphaz_el": "Static interband polarizability (z) [Ang]",
        "alphax_w": "Interband polarizability (x) [Ang]",
        "alphay_w": "Interband polarizability (y) [Ang]",
        "alphaz_w": "Interband polarizability (z) [Ang]",
        "alpha0x_w": "Interband polarizability without local field effects (x) [Ang]",
        "alpha0y_w": "Interband polarizability without local field effects (y) [Ang]",
        "alpha0z_w": "Interband polarizability without local field effects (z) [Ang]",
        "plasmafreq_vv": "Plasmafrequency tensor.",
        "frequencies": "Frequency grid [eV]."
    }

    formats = {"ase_webpanel": webpanel}


@command('asr.polarizability',
         dependencies=['asr.structureinfo', 'asr.gs@calculate'],
         requires=['gs.gpw'],
         returns=Result)
@option(
    '--gs', help='Ground state on which response is based',
    type=str)
@option('--kptdensity', help='K-point density',
        type=float)
@option('--ecut', help='Plane wave cutoff',
        type=float)
@option('--xc', help='XC interaction', type=Choice(['RPA', 'ALDA']))
@option('--bandfactor', type=int,
        help='Number of unoccupied bands = (#occ. bands) * bandfactor)')
def main(gs: str = 'gs.gpw', kptdensity: float = 20.0, ecut: float = 50.0,
         xc: str = 'RPA', bandfactor: int = 5) -> Result:
    """Calculate linear response polarizability or dielectricfunction (only in 3D)."""
    from gpaw import GPAW
    from gpaw.mpi import world
    from gpaw.response.df import DielectricFunction

    atoms = read('structure.json')
    pbc = atoms.pbc.tolist()

    dfkwargs = {
        'eta': 0.05,
        'domega0': 0.005,
        'ecut': ecut,
        'name': 'chi',
        'intraband': False
    }

    ND = np.sum(pbc)
    if ND == 3 or ND == 1:
        kpts = {'density': kptdensity, 'gamma': False, 'even': True}
    elif ND == 2:
        kpts = get_kpts_size(atoms=atoms, kptdensity=kptdensity)
        volume = atoms.get_volume()
        if volume < 120:
            nblocks = world.size // 4
        else:
            nblocks = world.size // 2
        dfkwargs.update({
            'nblocks': nblocks,
            'integrationmode': 'tetrahedron integration',
            'truncation': '2D'
        })

    else:
        raise NotImplementedError(
            'Polarizability not implemented for 1D and 2D structures')

    try:
        if not Path('es.gpw').is_file():
            calc_old = GPAW(gs, txt=None)
            nval = calc_old.wfs.nvalence

            calc = GPAW(
                gs,
                txt='es.txt',
                fixdensity=True,
                nbands=(bandfactor + 1) * nval,
                convergence={'bands': bandfactor * nval},
                occupations={'name': 'fermi-dirac',
                             'width': 1e-4},
                kpts=kpts)
            calc.get_potential_energy()
            calc.write('es.gpw', mode='all')

        df = DielectricFunction('es.gpw', **dfkwargs)
        alpha0x, alphax = df.get_polarizability(
            q_c=[0, 0, 0], direction='x', filename=None,
            xc=xc)
        alpha0y, alphay = df.get_polarizability(
            q_c=[0, 0, 0], direction='y', filename=None,
            xc=xc)
        alpha0z, alphaz = df.get_polarizability(
            q_c=[0, 0, 0], direction='z', filename=None,
            xc=xc)

        plasmafreq_vv = df.chi0.plasmafreq_vv

        frequencies = df.get_frequencies()
        data = {
            'alpha0x_w': np.array(alpha0x),
            'alphax_w': np.array(alphax),
            'alpha0y_w': np.array(alpha0y),
            'alphay_w': np.array(alphay),
            'alpha0z_w': np.array(alpha0z),
            'alphaz_w': np.array(alphaz),
            'plasmafreq_vv': plasmafreq_vv,
            'frequencies': frequencies
        }

        data['alphax_el'] = data['alphax_w'][0].real
        data['alphay_el'] = data['alphay_w'][0].real
        data['alphaz_el'] = data['alphaz_w'][0].real

    finally:
        world.barrier()
        if world.rank == 0:
            for filename in ['es.gpw', 'chi+0+0+0.pckl']:
                es_file = Path(filename)
                if es_file.is_file():
                    es_file.unlink()

    return data


def polarizability(row, fx, fy, fz):
    import matplotlib.pyplot as plt

    data = row.data.get('results-asr.polarizability.json')

    if data is None:
        return
    frequencies = data['frequencies']
    i2 = abs(frequencies - 50.0).argmin()
    frequencies = frequencies[:i2]
    alphax_w = data['alphax_w'][:i2]
    alphay_w = data['alphay_w'][:i2]
    alphaz_w = data['alphaz_w'][:i2]

    infrared = row.data.get('results-asr.infraredpolarizability.json')
    if infrared:
        from scipy.interpolate import interp1d
        omegatmp_w = infrared['omega_w']
        alpha_wvv = infrared['alpha_wvv']
        alphax = interp1d(omegatmp_w, alpha_wvv[:, 0, 0], fill_value=0,
                          bounds_error=False)
        alphax_w = (alphax_w + alphax(frequencies))
        alphay = interp1d(omegatmp_w, alpha_wvv[:, 1, 1], fill_value=0,
                          bounds_error=False)
        alphay_w = (alphay_w + alphay(frequencies))
        alphaz = interp1d(omegatmp_w, alpha_wvv[:, 2, 2], fill_value=0,
                          bounds_error=False)
        alphaz_w = (alphaz_w + alphaz(frequencies))

    ax = plt.figure().add_subplot(111)
    ax1 = ax
    try:
        wpx = row.plasmafrequency_x
        if wpx > 0.01:
            alphaxfull_w = alphax_w - wpx**2 / (2 * np.pi * (frequencies + 1e-9)**2)
            ax.plot(
                frequencies,
                np.real(alphaxfull_w),
                '-',
                c='C1',
                label='real')
            ax.plot(
                frequencies,
                np.real(alphax_w),
                '--',
                c='C1',
                label='real (interband)')
        else:
            ax.plot(frequencies, np.real(alphax_w), c='C1', label='real')
    except AttributeError:
        ax.plot(frequencies, np.real(alphax_w), c='C1', label='real')
    ax.plot(frequencies, np.imag(alphax_w), c='C0', label='imag')

    plot_polarizability(ax, frequencies, alphax_w, filename=fx, direction='x')

    ax = plt.figure().add_subplot(111)
    ax2 = ax
    try:
        wpy = row.plasmafrequency_y
        if wpy > 0.01:
            alphayfull_w = alphay_w - wpy**2 / (2 * np.pi * (frequencies + 1e-9)**2)
            ax.plot(
                frequencies,
                np.real(alphayfull_w),
                '-',
                c='C1',
                label='real')
            ax.plot(
                frequencies,
                np.real(alphay_w),
                '--',
                c='C1',
                label='real (interband)')
        else:
            ax.plot(frequencies, np.real(alphay_w), c='C1', label='real')
    except AttributeError:
        ax.plot(frequencies, np.real(alphay_w), c='C1', label='real')

    ax.plot(frequencies, np.imag(alphay_w), c='C0', label='imag')
    plot_polarizability(ax, frequencies, alphay_w, filename=fy, direction='y')

    ax3 = plt.figure().add_subplot(111)
    ax3.plot(frequencies, np.real(alphaz_w), c='C1', label='real')
    ax3.plot(frequencies, np.imag(alphaz_w), c='C0', label='imag')
    plot_polarizability(ax3, frequencies, alphaz_w, filename=fz, direction='z')

    return ax1, ax2, ax3


def ylims(ws, data, wstart=0.0):
    i = abs(ws - wstart).argmin()
    x = data[i:]
    x1, x2 = x.real, x.imag
    y1 = min(x1.min(), x2.min()) * 1.02
    y2 = max(x1.max(), x2.max()) * 1.02
    return y1, y2


def plot_polarizability(ax, frequencies, alpha_w, filename, direction):
    ax.set_title(f'Polarization: {direction}')
    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel(r'Polarizability [$\mathrm{\AA}$]')
    ax.set_ylim(ylims(ws=frequencies, data=alpha_w, wstart=0.5))
    ax.legend()
    ax.set_xlim((0, 10))
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(filename)


if __name__ == '__main__':
    main.cli()
