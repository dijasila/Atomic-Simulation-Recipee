from asr.core import command, option, ASRResult, prepare_result
from typing import List
from itertools import product
import numpy as np


def sphere_points(distance=None):
    '''Calculates equidistant points on the upper half sphere

    Returns list of spherical coordinates (thetas, phis) in degrees
    '''

    import math
    N = math.ceil(129600 / (math.pi) * 1 / distance**2)

    A = 4 * math.pi
    a = A / N
    d = math.sqrt(a)

    # Even number of theta angles ensure 90 deg is included
    Mtheta = round(math.pi / (2 * d)) * 2
    dtheta = math.pi / Mtheta
    dphi = a / dtheta
    points = []

    # Limit theta loop to upper half-sphere
    for m in range(Mtheta // 2 + 1):
        # m = 0 ensure 0 deg is included, Mphi = 1 is used in this case
        theta = math.pi * m / Mtheta
        Mphi = max(round(2 * math.pi * math.sin(theta) / dphi), 1)
        for n in range(Mphi):
            phi = 2 * math.pi * n / Mphi
            points.append([theta, phi])
    thetas, phis = np.array(points).T

    if not any(thetas - np.pi / 2 < 1e-14):
        import warnings
        warnings.warn('xy-plane not included in sampling')

    return thetas * 180 / math.pi, phis * 180 / math.pi % 180


class GroundState:
    def __init__(self, gpw):
        self.gpw = gpw

    def is_collinear(self):
        from gpaw import GPAW
        calc = GPAW(self.gpw)

        is_collinear = 'qspiral' not in calc.parameters['mode'].keys()

        if not is_collinear:
            qn = calc.parameters['mode']['qspiral']
            if np.linalg.norm(qn) < 1e-14:
                is_collinear = True

        return is_collinear

    def band_energy(self, projected, theta, phi, occcalc):
        from gpaw.spinorbit import soc_eigenstates
        return soc_eigenstates(
            calc=self.gpw, projected=projected,
            theta=theta, phi=phi,
            occcalc=occcalc).calculate_band_energy()


@prepare_result
class PreResult(ASRResult):
    soc_tp: np.ndarray
    theta_tp: np.ndarray
    phi_tp: np.ndarray
    projected_soc: bool
    key_descriptions = {'soc_tp': 'Spin Orbit correction [eV]',
                        'theta_tp': 'Orientation of magnetic order from z->x [deg]',
                        'phi_tp': 'Orientation of magnetic order from x->y [deg]',
                        'projected_soc': 'Projected SOC for non-collinear spin spirals'}

    def stereographic_projection(self):
        
        def stereo_project_point(inpoint, axis=0, r=1, max_norm=1):
            point = np.divide(inpoint * r, inpoint[axis] + r)
            point[axis] = 0
            return point

        theta, phi = self['theta_tp'], self['phi_tp']
        theta = theta * np.pi / 180
        phi = phi * np.pi / 180
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        points = np.array([x, y, z]).T
        projected_points = []
        for p in points:
            projected_points.append(stereo_project_point(p, axis=2))

        return projected_points



@command(module='asr.spinorbit',
         resources='1:4h')
@option('--gpw', help='The file path for the GPAW calculator context.', type=str)
@option('--distance', type=float,
        help='Distance between sample points on the sphere')
@option('--projected_soc', type=bool,
        help='For non-collinear spin spirals, projected SOC should be applied (True)')
@option('--width', type=float,
        help='The fermi smearing of the SOC calculation (eV)')
def calculate(gpw: str = 'gs.gpw', distance: float = 2.0,
              projected_soc: bool = None, width: float = 0.001) -> ASRResult:
    '''Calculates the spin-orbit coupling at equidistant points on a unit sphere. '''

    gs = GroundState(gpw)
    return _calculate(gs, distance, projected_soc, width)


def _calculate(gs: GroundState, distance: float,
               projected_soc: bool, width: float) -> ASRResult:

    from gpaw.occupations import create_occ_calc
    occcalc = create_occ_calc({'name': 'fermi-dirac', 'width': width})

    projected_soc = not gs.is_collinear() if projected_soc is None else projected_soc
    theta_tp, phi_tp = sphere_points(distance=distance)

    soc_tp = np.array([])
    for theta, phi in zip(theta_tp, phi_tp):
        en_soc = gs.band_energy(projected=projected_soc, theta=theta, phi=phi,
                                occcalc=occcalc)
        # Noise should not be an issue since it is the same for the calculator
        # en_soc_0 = soc_eigenstates(calc, projected=projected, scale=0.0,
        #                            theta=theta, phi=phi).calculate_band_energy()
        soc_tp = np.append(soc_tp, en_soc)

    return PreResult.fromdata(soc_tp=soc_tp, theta_tp=theta_tp, phi_tp=phi_tp,
                              projected_soc=projected_soc)


def webpanel(result, row, key_descriptions):
    from asr.database.browser import fig
    rows = [['Spinorbit bandwidth', str(np.round(result.get('soc_bw'), 1))],
            ['Spinorbit Minimum (&theta;, &phi;)', '('
             + str(np.round(result.get('theta_min'), 1))
             + ', ' + str(np.round(result.get('phi_min')[1], 1)) + ')']]
    spiraltable = {'type': 'table',
                   'header': ['Property', 'Value'],
                   'rows': rows}

    panel = {'title': 'Spin spirals',
             'columns': [[fig('spinorbit.png')], [spiraltable]],
             'plot_descriptions': [{'function': plot_stereographic_energies,
                                    'filenames': ['spinorbit.png']}],
             'sort': 1}
    return [panel]


@prepare_result
class Result(ASRResult):
    soc_bw: float
    theta_min: float
    phi_min: float
    projected_soc: bool
    key_descriptions = {'soc_bw': 'Bandwidth of SOC energies [meV]',
                        'theta_min': 'Angles from z->x [deg]',
                        'phi_min': 'Angles from x->y [deg]',
                        'projected_soc': 'Projected SOC for spin spirals'}
    formats = {'ase_webpanel': webpanel}


@command(module='asr.spinorbit',
         dependencies=['asr.spinorbit@calculate'],
         resources='1:1m',
         returns=Result)
def main() -> Result:
    from asr.core import read_json

    results = read_json('results-asr.spinorbit@calculate.json')
    soc_tp = results['soc_tp']
    theta_tp = results['theta_tp']
    phi_tp = results['phi_tp']
    projected_soc = results['projected_soc']

    tp_min = np.argmin(soc_tp)
    theta_min = theta_tp[tp_min]
    phi_min = phi_tp[tp_min]
    soc_bw = 1e3 * (np.max(soc_tp) - np.min(soc_tp))

    return Result.fromdata(soc_bw=soc_bw, theta_min=theta_min, phi_min=phi_min,
                           projected_soc=projected_soc)


def plot_stereographic_energies(row, fname, display_sampling=False):
    from matplotlib.colors import Normalize
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from scipy.interpolate import griddata

    socdata = row.data.get('results-asr.spinorbit@calculate.json')
    soc = (socdata['soc_tp'] - min(socdata['soc_tp'])) * 10**3

    plt.figure(figsize=(5 * 1.25, 5))
    ax = plt.gca()
    norm = Normalize(vmin=min(soc), vmax=max(soc))

    X, Y, Z = np.array(projected_points).T

    xi = np.linspace(min(X), max(X), 100)
    yi = np.linspace(min(Y), max(Y), 100)
    zi = griddata((X, Y), soc, (xi[None, :], yi[:, None]))
    ax.contour(xi, yi, zi, 15, linewidths=0.5, colors='k', norm=norm)
    ax.contourf(xi, yi, zi, 15, cmap=plt.cm.jet, norm=norm)
    if display_sampling:
        ax.scatter(X, Y, marker='o', c='k', s=5)

    ax.axis('equal')
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax)
    cbar.ax.set_ylabel(r'$E_{soc} [meV]$')
    plt.savefig(fname)


if __name__ == '__main__':
    main.cli()
