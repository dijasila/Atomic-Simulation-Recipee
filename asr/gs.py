"""Electronic ground state properties."""
from asr.core import command, option, DictStr, ASRResult, prepare_result
from asr.database.browser import (
    table, fig,
    entry_parameter_description,
    describe_entry, WebPanel,
    make_panel_description
)


import numpy as np
import typing

panel_description = make_panel_description(
    """
Electronic properties derived from a ground state density functional theory
calculation.
""",
    articles=['C2DB'],
)


@command(module='asr.gs',
         creates=['gs.gpw'],
         requires=['structure.json'],
         resources='8:10h')
@option('-c', '--calculator', help='Calculator params.', type=DictStr())
def calculate(calculator: dict = {
        'name': 'gpaw',
        'mode': {'name': 'pw', 'ecut': 800},
        'xc': 'PBE',
        'basis': 'dzp',
        'kpts': {'density': 12.0, 'gamma': True},
        'occupations': {'name': 'fermi-dirac',
                        'width': 0.05},
        'convergence': {'bands': 'CBM+3.0'},
        'nbands': '200%',
        'txt': 'gs.txt',
        'charge': 0}) -> ASRResult:
    """Calculate ground state file.

    This recipe saves the ground state to a file gs.gpw based on the structure
    in 'structure.json'. This can then be processed by asr.gs@postprocessing
    for storing any derived quantities. See asr.gs@postprocessing for more
    information.
    """
    from ase.io import read
    from ase.calculators.calculator import PropertyNotImplementedError
    from asr.relax import set_initial_magnetic_moments
    atoms = read('structure.json')

    if not atoms.has('initial_magmoms'):
        set_initial_magnetic_moments(atoms)

    nd = np.sum(atoms.pbc)
    if nd == 2:
        assert not atoms.pbc[2], \
            'The third unit cell axis should be aperiodic for a 2D material!'
        calculator['poissonsolver'] = {'dipolelayer': 'xy'}

    from ase.calculators.calculator import get_calculator_class
    name = calculator.pop('name')
    calc = get_calculator_class(name)(**calculator)

    atoms.calc = calc
    atoms.get_forces()
    try:
        atoms.get_stress()
    except PropertyNotImplementedError:
        pass
    atoms.get_potential_energy()
    atoms.calc.write('gs.gpw')

    return ASRResult()


def _get_parameter_description(row):
    desc = entry_parameter_description(
        row.data,
        'asr.gs@calculate',
        exclude_keys=set(['txt', 'fixdensity', 'verbose', 'symmetry',
                          'idiotproof', 'maxiter', 'hund', 'random',
                          'experimental', 'basis', 'setups']))
    return desc


def _explain_bandgap(row, gap_name):
    parameter_description = _get_parameter_description(row)

    if gap_name == 'gap':
        name = 'Band gap'
        adjective = ''
    elif gap_name == 'gap_dir':
        name = 'Direct band gap'
        adjective = 'direct '
    else:
        raise ValueError(f'Bad gapname {gap_name}')

    txt = (f'The {adjective}electronic single-particle band gap '
           'including spin–orbit effects.')

    description = f'{txt}\n\n{parameter_description}'
    return describe_entry(name, description=description)


def vbm_or_cbm_row(title, quantity_name, reference_explanation, value):
    description = (f'Energy of the {quantity_name} relative to the '
                   f'{reference_explanation}. '
                   'Spin–orbit coupling is included.')
    return [describe_entry(title, description=description), f'{value:.2f} eV']


def webpanel(result, row, key_descriptions):
    # for defect systems we don't want to show this panel
    if row.get('defect_name') is not None:
        return []

    parameter_description = _get_parameter_description(row)

    explained_keys = []

    def make_gap_row(name):
        value = result[name]
        description = _explain_bandgap(row, name)
        return [description, f'{value:0.2f} eV']

    gap_row = make_gap_row('gap')
    direct_gap_row = make_gap_row('gap_dir')

    for key in ['dipz', 'evacdiff', 'workfunction', 'dos_at_ef_soc']:
        if key in result.key_descriptions:
            key_description = result.key_descriptions[key]
            explanation = (f'{key_description} '
                           '(Including spin–orbit effects).\n\n'
                           + parameter_description)
            explained_key = describe_entry(key, description=explanation)
        else:
            explained_key = key
        explained_keys.append(explained_key)

    t = table(result, 'Property',
              explained_keys,
              key_descriptions)

    t['rows'] += [gap_row, direct_gap_row]

    if result.gap > 0:
        if result.get('evac'):
            eref = result.evac
            vbm_title = 'Valence band maximum relative to vacuum level'
            cbm_title = 'Conduction band minimum relative to vacuum level'
            reference_explanation = (
                'the asymptotic value of the '
                'electrostatic potential in the vacuum region')
        else:
            eref = result.efermi
            vbm_title = 'Valence band maximum relative to Fermi level'
            cbm_title = 'Conduction band minimum relative to Fermi level'
            reference_explanation = 'the Fermi level'

        vbm_displayvalue = result.vbm - eref
        cbm_displayvalue = result.cbm - eref
        info = [
            vbm_or_cbm_row(vbm_title, 'valence band maximum (VBM)',
                           reference_explanation, vbm_displayvalue),
            vbm_or_cbm_row(cbm_title, 'conduction band minimum (CBM)',
                           reference_explanation, cbm_displayvalue)
        ]

        t['rows'].extend(info)

    from asr.utils.hacks import gs_xcname_from_row
    xcname = gs_xcname_from_row(row)
    title = f'Basic electronic properties ({xcname})'

    panel = WebPanel(
        title=describe_entry(title, panel_description),
        columns=[[t], [fig('bz-with-gaps.png')]],
        sort=10)

    summary = WebPanel(
        title=describe_entry(
            'Summary',
            description='This panel contains a summary of '
            'basic properties of the material.'),
        columns=[[{
            'type': 'table',
            'header': ['Basic properties', ''],
            'rows': [gap_row],
        }]],
        plot_descriptions=[{'function': bz_with_band_extremums,
                            'filenames': ['bz-with-gaps.png']}],
        sort=10)

    return [panel, summary]


def bz_with_band_extremums(row, fname):
    from ase.geometry.cell import Cell
    from matplotlib import pyplot as plt
    import numpy as np
    from asr.utils.symmetry import c2db_symmetry_eps

    ndim = sum(row.pbc)

    # Standardize the cell rotation via Bravais lattice roundtrip:
    lat = Cell(row.cell).get_bravais_lattice(pbc=row.pbc,
                                             eps=c2db_symmetry_eps)
    cell = lat.tocell()

    plt.figure(figsize=(4, 4))
    lat.plot_bz(vectors=False, pointstyle={'c': 'k', 'marker': '.'})

    gsresults = row.data.get('results-asr.gs.json')
    cbm_c = gsresults['k_cbm_c']
    vbm_c = gsresults['k_vbm_c']
    op_scc = row.data[
        'results-asr.structureinfo.json']['spglib_dataset']['rotations']
    if cbm_c is not None:
        if not row.is_magnetic:
            op_scc = np.concatenate([op_scc, -op_scc])
        ax = plt.gca()
        icell_cv = cell.reciprocal()
        vbm_style = {'marker': 'o', 'facecolor': 'w',
                     'edgecolors': 'C0', 's': 50, 'lw': 2,
                     'zorder': 4}
        cbm_style = {'c': 'C1', 'marker': 'o', 's': 20, 'zorder': 5}
        cbm_sc = np.dot(op_scc.transpose(0, 2, 1), cbm_c)
        vbm_sc = np.dot(op_scc.transpose(0, 2, 1), vbm_c)
        cbm_sv = np.dot(cbm_sc, icell_cv)
        vbm_sv = np.dot(vbm_sc, icell_cv)

        if ndim < 3:
            ax.scatter([vbm_sv[:, 0]], [vbm_sv[:, 1]], **vbm_style, label='VBM')
            ax.scatter([cbm_sv[:, 0]], [cbm_sv[:, 1]], **cbm_style, label='CBM')

            # We need to keep the limits set by ASE in 3D, else the aspect
            # ratio goes haywire.  Hence this bit is also for ndim < 3 only.
            xlim = np.array(ax.get_xlim()) * 1.4
            ylim = np.array(ax.get_ylim()) * 1.4
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        else:
            ax.scatter([vbm_sv[:, 0]], [vbm_sv[:, 1]],
                       [vbm_sv[:, 2]], **vbm_style, label='VBM')
            ax.scatter([cbm_sv[:, 0]], [cbm_sv[:, 1]],
                       [cbm_sv[:, 2]], **cbm_style, label='CBM')

        plt.legend(loc='upper center', ncol=3, prop={'size': 9})

    plt.tight_layout()
    plt.savefig(fname)


@prepare_result
class GapsResult(ASRResult):

    gap: float
    vbm: float
    cbm: float
    gap_dir: float
    vbm_dir: float
    cbm_dir: float
    k_vbm_c: typing.Tuple[float, float, float]
    k_cbm_c: typing.Tuple[float, float, float]
    k_vbm_dir_c: typing.Tuple[float, float, float]
    k_cbm_dir_c: typing.Tuple[float, float, float]
    skn1: typing.Tuple[int, int, int]
    skn2: typing.Tuple[int, int, int]
    skn1_dir: typing.Tuple[int, int, int]
    skn2_dir: typing.Tuple[int, int, int]
    efermi: float

    key_descriptions: typing.Dict[str, str] = dict(
        efermi='Fermi level [eV].',
        gap='Band gap [eV].',
        vbm='Valence band maximum [eV].',
        cbm='Conduction band minimum [eV].',
        gap_dir='Direct band gap [eV].',
        vbm_dir='Direct valence band maximum [eV].',
        cbm_dir='Direct conduction band minimum [eV].',
        k_vbm_c='Scaled k-point coordinates of valence band maximum (VBM).',
        k_cbm_c='Scaled k-point coordinates of conduction band minimum (CBM).',
        k_vbm_dir_c='Scaled k-point coordinates of direct valence band maximum (VBM).',
        k_cbm_dir_c='Scaled k-point coordinates of direct calence band minimum (CBM).',
        skn1="(spin,k-index,band-index)-tuple for valence band maximum.",
        skn2="(spin,k-index,band-index)-tuple for conduction band minimum.",
        skn1_dir="(spin,k-index,band-index)-tuple for direct valence band maximum.",
        skn2_dir="(spin,k-index,band-index)-tuple for direct conduction band minimum.",
    )


def gaps(calc, soc=True) -> GapsResult:
    # ##TODO min kpt dens? XXX
    # inputs: gpw groundstate file, soc?, direct gap? XXX
    from functools import partial
    from asr.utils.gpw2eigs import calc2eigs
    from asr.magnetic_anisotropy import get_spin_axis

    if soc:
        ibzkpts = calc.get_bz_k_points()
    else:
        ibzkpts = calc.get_ibz_k_points()

    (evbm_ecbm_gap,
     skn_vbm, skn_cbm) = get_gap_info(soc=soc, direct=False,
                                      calc=calc)
    (evbm_ecbm_direct_gap,
     direct_skn_vbm, direct_skn_cbm) = get_gap_info(soc=soc, direct=True,
                                                    calc=calc)

    k_vbm, k_cbm = skn_vbm[1], skn_cbm[1]
    direct_k_vbm, direct_k_cbm = direct_skn_vbm[1], direct_skn_cbm[1]

    get_kc = partial(get_1bz_k, ibzkpts, calc)

    k_vbm_c = get_kc(k_vbm)
    k_cbm_c = get_kc(k_cbm)
    direct_k_vbm_c = get_kc(direct_k_vbm)
    direct_k_cbm_c = get_kc(direct_k_cbm)

    if soc:
        theta, phi = get_spin_axis()
        _, efermi = calc2eigs(calc, soc=True,
                              theta=theta, phi=phi)
    else:
        efermi = calc.get_fermi_level()

    return GapsResult.fromdata(
        gap=evbm_ecbm_gap[2],
        vbm=evbm_ecbm_gap[0],
        cbm=evbm_ecbm_gap[1],
        gap_dir=evbm_ecbm_direct_gap[2],
        vbm_dir=evbm_ecbm_direct_gap[0],
        cbm_dir=evbm_ecbm_direct_gap[1],
        k_vbm_c=k_vbm_c,
        k_cbm_c=k_cbm_c,
        k_vbm_dir_c=direct_k_vbm_c,
        k_cbm_dir_c=direct_k_cbm_c,
        skn1=skn_vbm,
        skn2=skn_cbm,
        skn1_dir=direct_skn_vbm,
        skn2_dir=direct_skn_cbm,
        efermi=efermi
    )


def get_1bz_k(ibzkpts, calc, k_index):
    from gpaw.kpt_descriptor import to1bz
    k_c = ibzkpts[k_index] if k_index is not None else None
    if k_c is not None:
        k_c = to1bz(k_c[None], calc.wfs.gd.cell_cv)[0]
    return k_c


def get_gap_info(soc, direct, calc):
    from ase.dft.bandgap import bandgap
    from asr.utils.gpw2eigs import calc2eigs
    from asr.magnetic_anisotropy import get_spin_axis
    # e1 is VBM, e2 is CBM
    if soc:
        theta, phi = get_spin_axis()
        e_km, efermi = calc2eigs(calc,
                                 soc=True, theta=theta, phi=phi)
        # km1 is VBM index tuple: (s, k, n), km2 is CBM index tuple: (s, k, n)
        gap, km1, km2 = bandgap(eigenvalues=e_km, efermi=efermi, direct=direct,
                                output=None)
        if km1[0] is not None:
            e1 = e_km[km1]
            e2 = e_km[km2]
        else:
            e1, e2 = None, None
        x = (e1, e2, gap), (0,) + tuple(km1), (0,) + tuple(km2)
    else:
        g, skn1, skn2 = bandgap(calc, direct=direct, output=None)
        if skn1[1] is not None:
            e1 = calc.get_eigenvalues(spin=skn1[0], kpt=skn1[1])[skn1[2]]
            e2 = calc.get_eigenvalues(spin=skn2[0], kpt=skn2[1])[skn2[2]]
        else:
            e1, e2 = None, None
        x = (e1, e2, g), skn1, skn2
    return x


@prepare_result
class VacuumLevelResults(ASRResult):
    z_z: np.ndarray
    v_z: np.ndarray
    evacdiff: float
    dipz: float
    evac1: float
    evac2: float
    evacmean: float
    efermi_nosoc: float

    key_descriptions = {
        'z_z': 'Grid points for potential [Å].',
        'v_z': 'Electrostatic potential [eV].',
        'evacdiff': 'Difference of vacuum levels on both sides of slab [eV].',
        'dipz': 'Out-of-plane dipole [e · Å].',
        'evac1': 'Top side vacuum level [eV].',
        'evac2': 'Bottom side vacuum level [eV]',
        'evacmean': 'Average vacuum level [eV].',
        'efermi_nosoc': 'Fermi level without SOC [eV].'}


def vacuumlevels(atoms, calc, n=8) -> VacuumLevelResults:
    """Get the vacuum level(s).

    Special case for 2D:

        Get the vacuumlevels on both sides of a material.
        Assumes the 2D material periodic directions are x and y.
        Assumes that the 2D material is centered in the z-direction of
        the unit cell.

    Parameters
    ----------
    atoms: Atoms
       The Atoms object.
    calc: GPAW-calculator
        The GPAW object.  Provides the electrostatic potential.
    n: int
        Number of gridpoints away from the edge to evaluate the vacuum levels.
    """
    import numpy as np

    z_z = None
    v_z = None
    devac = None
    dipz = None
    evac1 = None
    evac2 = None
    evacmean = None

    ves = calc.get_electrostatic_potential()

    nperiodic = atoms.pbc.sum()
    if nperiodic < 2:
        evacmean = 0.0
        for v, periodic in zip([ves[0], ves[:, 0], ves[:, :, 0]],
                               atoms.pbc):
            if not periodic:
                evacmean += v.mean() / (3 - nperiodic)
    elif nperiodic == 2:
        # Record electrostatic potential as a function of z
        assert not atoms.pbc[2]
        v_z = ves.mean(0).mean(0)
        z_z = np.linspace(0, atoms.cell[2, 2], len(v_z), endpoint=False)
        dipz = atoms.get_dipole_moment()[2]
        devac = evacdiff(atoms)
        evac1 = v_z[n]
        evac2 = v_z[-n]
        evacmean = (v_z[n] + v_z[-n]) / 2

    return VacuumLevelResults.fromdata(
        z_z=z_z,
        v_z=v_z,
        dipz=dipz,
        evacdiff=devac,
        evac1=evac1,
        evac2=evac2,
        evacmean=evacmean,
        efermi_nosoc=calc.get_fermi_level())


def evacdiff(atoms):
    """Derive vacuum energy level difference from the dipole moment.

    Calculate vacuum energy level difference from the dipole moment of
    a slab assumed to be in the xy plane

    Returns
    -------
    out: float
        vacuum level difference in eV
    """
    import numpy as np
    from ase.units import Bohr, Hartree

    A = np.linalg.det(atoms.cell[:2, :2] / Bohr)
    dipz = atoms.get_dipole_moment()[2] / Bohr
    evacsplit = 4 * np.pi * dipz / A * Hartree

    return evacsplit


@prepare_result
class Result(ASRResult):
    """Container for ground state results.

    Examples
    --------
    >>> res = Result(data=dict(etot=0), strict=False)
    >>> res.etot
    0
    """

    forces: np.ndarray
    stresses: np.ndarray
    etot: float
    evac: float
    evacdiff: float
    dipz: float
    efermi: float
    gap: float
    vbm: float
    cbm: float
    gap_dir: float
    vbm_dir: float
    cbm_dir: float
    gap_dir_nosoc: float
    gap_nosoc: float
    gaps_nosoc: GapsResult
    k_vbm_c: typing.Tuple[float, float, float]
    k_cbm_c: typing.Tuple[float, float, float]
    k_vbm_dir_c: typing.Tuple[float, float, float]
    k_cbm_dir_c: typing.Tuple[float, float, float]
    skn1: typing.Tuple[int, int, int]
    skn2: typing.Tuple[int, int, int]
    skn1_dir: typing.Tuple[int, int, int]
    skn2_dir: typing.Tuple[int, int, int]
    workfunction: float
    vacuumlevels: VacuumLevelResults

    key_descriptions = dict(
        etot='Total energy [eV].',
        workfunction="Workfunction [eV]",
        forces='Forces on atoms [eV/Å].',
        stresses='Stress on unit cell [eV/Å^dim].',
        evac='Vacuum level [eV].',
        evacdiff='Vacuum level shift (Vacuum level shift) [eV].',
        dipz='Out-of-plane dipole [e · Å].',
        efermi='Fermi level [eV].',
        gap='Band gap [eV].',
        vbm='Valence band maximum [eV].',
        cbm='Conduction band minimum [eV].',
        gap_dir='Direct band gap [eV].',
        vbm_dir='Direct valence band maximum [eV].',
        cbm_dir='Direct conduction band minimum [eV].',
        gap_dir_nosoc='Direct gap without SOC [eV].',
        gap_nosoc='Gap without SOC [eV].',
        gaps_nosoc='Container for bandgap results without SOC.',
        vacuumlevels='Container for results that relate to vacuum levels.',
        k_vbm_c='Scaled k-point coordinates of valence band maximum (VBM).',
        k_cbm_c='Scaled k-point coordinates of conduction band minimum (CBM).',
        k_vbm_dir_c='Scaled k-point coordinates of direct valence band maximum (VBM).',
        k_cbm_dir_c='Scaled k-point coordinates of direct calence band minimum (CBM).',
        skn1="(spin,k-index,band-index)-tuple for valence band maximum.",
        skn2="(spin,k-index,band-index)-tuple for conduction band minimum.",
        skn1_dir="(spin,k-index,band-index)-tuple for direct valence band maximum.",
        skn2_dir="(spin,k-index,band-index)-tuple for direct conduction band minimum.",
    )

    formats = {"ase_webpanel": webpanel}


@command(module='asr.gs',
         requires=['gs.gpw', 'structure.json',
                   'results-asr.magnetic_anisotropy.json'],
         dependencies=['asr.gs@calculate', 'asr.magnetic_anisotropy',
                       'asr.structureinfo'],
         returns=Result)
def main() -> Result:
    """Extract derived quantities from groundstate in gs.gpw."""
    from ase.io import read
    from asr.calculators import get_calculator

    # Just some quality control before we start
    atoms = read('structure.json')
    calc = get_calculator()('gs.gpw')
    pbc = atoms.pbc
    ndim = np.sum(pbc)

    if ndim == 2:
        assert not pbc[2], \
            'The third unit cell axis should be aperiodic for a 2D material!'
        # For 2D materials we check that the calculater used a dipole
        # correction if the material has an out-of-plane dipole

        # Small hack
        atoms = calc.atoms
        atoms.calc = calc
        evacdiffmin = 10e-3
        if evacdiff(calc.atoms) > evacdiffmin:
            assert calc.todict().get('poissonsolver', {}) == \
                {'dipolelayer': 'xy'}, \
                ('The ground state has a finite dipole moment along aperiodic '
                 'axis but calculation was without dipole correction.')

    # Now that some checks are done, we can extract information
    forces = calc.get_property('forces', allow_calculation=False)
    stresses = calc.get_property('stress', allow_calculation=False)
    etot = calc.get_potential_energy()

    gaps_nosoc = gaps(calc, soc=False)
    gaps_soc = gaps(calc, soc=True)
    vac = vacuumlevels(atoms, calc)
    workfunction = vac.evacmean - gaps_soc.efermi if vac.evacmean else None
    return Result.fromdata(
        forces=forces,
        stresses=stresses,
        etot=etot,
        gaps_nosoc=gaps_nosoc,
        gap_dir_nosoc=gaps_nosoc.gap_dir,
        gap_nosoc=gaps_nosoc.gap,
        gap=gaps_soc.gap,
        vbm=gaps_soc.vbm,
        cbm=gaps_soc.cbm,
        gap_dir=gaps_soc.gap_dir,
        vbm_dir=gaps_soc.vbm_dir,
        cbm_dir=gaps_soc.cbm_dir,
        k_vbm_c=gaps_soc.k_vbm_c,
        k_cbm_c=gaps_soc.k_cbm_c,
        k_vbm_dir_c=gaps_soc.k_vbm_dir_c,
        k_cbm_dir_c=gaps_soc.k_cbm_dir_c,
        skn1=gaps_soc.skn1,
        skn2=gaps_soc.skn2,
        skn1_dir=gaps_soc.skn1_dir,
        skn2_dir=gaps_soc.skn2_dir,
        efermi=gaps_soc.efermi,
        vacuumlevels=vac,
        dipz=vac.dipz,
        evac=vac.evacmean,
        evacdiff=vac.evacdiff,
        workfunction=workfunction)


if __name__ == '__main__':
    main.cli()
