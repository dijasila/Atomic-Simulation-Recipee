"""Electronic ground state properties."""
import pathlib
from ase import Atoms
from ase.io import read
from asr.core import (
    command, option, DictStr, ASRResult, prepare_result, AtomsFile,
)
from asr.core.command import ASRControl
from asr.calculators import (
    set_calculator_hook, Calculation, get_calculator_class, Calculation)

from asr.database.browser import (
    table, fig,
    entry_parameter_description,
    describe_entry, WebPanel,
    make_panel_description,
    cache_webpanel,
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


@prepare_result
class GroundStateCalculationResult(ASRResult):

    calculation: Calculation

    key_descriptions = dict(calculation='Calculation object')


def migrate_calculate_record(cache, selection):
    """Migrate old ground state records."""
    orig_records = cache.select(**selection)

    assert len(orig_records) == 1, [
        orig_record.migration_id for orig_record in orig_records]

    orig_record = orig_records[0]
    migrated_record = orig_record.copy()
    run_spec = migrated_record.run_specification
    calculation = migrated_record.result.calculation
    calc = calculation.load()
    calc_parameters = {'name': calculation.cls_name, **calc.parameters}
    run_spec.parameters.atoms = calc.atoms
    run_spec.parameters.calculator = calc_parameters
    migrated_record.run_specification = run_spec
    return orig_record, migrated_record


def get_gs_calculate_migrations(cache):
    from asr.core.migrate import Migration
    selection = {
        'run_specification.name': 'asr.gs::calculate',
        'migration_id': 'Migrate resultsfile results-asr.gs@calculate.json',
    }

    if cache.select(**selection):
        return [
            Migration(
                migrate_calculate_record,
                name='Add calculator parameter to gs.calculate record.',
                args=(cache, selection)
            ),
        ]
    else:
        return []


@command(module='asr.gs',
         argument_hooks=[set_calculator_hook],
         pass_control=True,
         returns=GroundStateCalculationResult,
         migrations=get_gs_calculate_migrations)
@option('-a', '--atoms', help='Atomic structure.',
        type=AtomsFile(), default='structure.json')
@option('-c', '--calculator', help='Calculator params.', type=DictStr())
def calculate(
        atoms: Atoms,
        calculator: dict = {
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
            'charge': 0
        },
        *,
        asrcontrol: ASRControl,
) -> GroundStateCalculationResult:
    """Calculate ground state file.

    This recipe saves the ground state to a file gs.gpw based on the structure
    in 'structure.json'. This can then be processed by asr.gs@postprocessing
    for storing any derived quantities. See asr.gs@postprocessing for more
    information.
    """
    from ase.calculators.calculator import PropertyNotImplementedError
    from asr.relax import set_initial_magnetic_moments

    if not atoms.has('initial_magmoms'):
        set_initial_magnetic_moments(atoms)

    nd = np.sum(atoms.pbc)
    if nd == 2:
        assert not atoms.pbc[2], \
            'The third unit cell axis should be aperiodic for a 2D material!'
        calculator['poissonsolver'] = {'dipolelayer': 'xy'}

    name = calculator.pop('name')
    calc = get_calculator_class(name)(**calculator)

    atoms.calc = calc
    atoms.get_forces()
    try:
        atoms.get_stress()
    except PropertyNotImplementedError:
        pass
    atoms.get_potential_energy()
    calculation = calc.save(id='gs')
    for i, filename in enumerate(calculation.paths):
        side_effect = asrcontrol.register_side_effect(filename)
        calculation.paths[i] = side_effect.path
    return GroundStateCalculationResult.fromdata(calculation=calculation)


@cache_webpanel(
    'asr.gs@main',
    '+,calculator.mode.ecut',
    '+,calculator.kpts.density',
)
def webpanel(result, row, key_descriptions):

    parameter_description = entry_parameter_description(
        row.data,
        'asr.gs@calculate',
        exclude_keys=set(['txt', 'fixdensity', 'verbose', 'symmetry',
                          'idiotproof', 'maxiter', 'hund', 'random',
                          'experimental', 'basis', 'setups']))

    explained_keys = []
    for key in ['gap', 'gap_dir',
                'dipz', 'evacdiff', 'workfunction', 'dos_at_ef_soc']:
        if key in result.key_descriptions:
            key_description = result.key_descriptions[key]
            explanation = (f'{key_description} '
                           '(Including spin-orbit effects).\n\n'
                           + parameter_description)
            explained_key = describe_entry(key, description=explanation)
        else:
            explained_key = key
        explained_keys.append(explained_key)

    gap = describe_entry('gap', description=explanation)
    t = table(result, 'Property',
              explained_keys,
              key_descriptions)

    gap = result.gap

    if gap > 0:
        if result.get('evac'):
            t['rows'].extend(
                [['Valence band maximum wrt. vacuum level',
                  f'{result.vbm - result.evac:.2f} eV'],
                 ['Conduction band minimum wrt. vacuum level',
                  f'{result.cbm - result.evac:.2f} eV']])
        else:
            t['rows'].extend(
                [['Valence band maximum wrt. Fermi level',
                  f'{result.vbm - result.efermi:.2f} eV'],
                 ['Conduction band minimum wrt. Fermi level',
                  f'{result.cbm - result.efermi:.2f} eV']])

    panel = WebPanel(
        title=describe_entry(
            'Basic electronic properties (PBE)',
            panel_description),
        columns=[[t], [fig('bz-with-gaps.png')]],
        sort=10)

    parameter_description = entry_parameter_description(
        row.data,
        'asr.gs@calculate',
        exclude_keys=set(['txt', 'fixdensity', 'verbose', 'symmetry',
                          'idiotproof', 'maxiter', 'hund', 'random',
                          'experimental', 'basis', 'setups']))
    description = ('The electronic band gap including spin-orbit effects. \n\n'
                   + parameter_description)
    datarow = [describe_entry('Band gap (PBE)',
                              description=description),
               f'{result.gap:0.2f} eV']
    summary = WebPanel(
        title=describe_entry(
            'Summary',
            description='This panel contains a summary of the most '
            'important properties of this material.'),
        columns=[[{'type': 'table',
                   'header': ['Electronic properties', ''],
                   'rows': [datarow]}]],
        plot_descriptions=[{'function': bz_with_band_extremums,
                            'filenames': ['bz-with-gaps.png']}],
        sort=10)

    return [panel, summary]


def bz_with_band_extremums(row, fname):
    from ase.geometry.cell import Cell
    from matplotlib import pyplot as plt
    from asr.structureinfo import main as structinfo
    import numpy as np
    ndim = sum(row.pbc)
    cell = Cell(row.cell)
    lat = cell.get_bravais_lattice(pbc=row.pbc)
    plt.figure(figsize=(4, 4))
    lat.plot_bz(vectors=False, pointstyle={'c': 'k', 'marker': '.'})
    gsresults = main.select(cache=row.cache)[0].result
    cbm_c = gsresults['k_cbm_c']
    vbm_c = gsresults['k_vbm_c']
    structresult = structinfo.get(cache=row.cache).result
    op_scc = structresult['spglib_dataset']['rotations']
    if cbm_c is not None:
        if not row.is_magnetic:
            op_scc = np.concatenate([op_scc, -op_scc])
        ax = plt.gca()
        icell_cv = np.linalg.inv(row.cell).T
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
        else:
            ax.scatter([vbm_sv[:, 0]], [vbm_sv[:, 1]],
                       [vbm_sv[:, 2]], **vbm_style, label='VBM')
            ax.scatter([cbm_sv[:, 0]], [cbm_sv[:, 1]],
                       [cbm_sv[:, 2]], **cbm_style, label='CBM')

        xlim = np.array(ax.get_xlim()) * 1.4
        ylim = np.array(ax.get_ylim()) * 1.4
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
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


def gaps(atoms, calc, calculator, soc=True) -> GapsResult:
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
     skn_vbm, skn_cbm) = get_gap_info(atoms,
                                      soc=soc, direct=False,
                                      calc=calc, calculator=calculator)
    (evbm_ecbm_direct_gap,
     direct_skn_vbm, direct_skn_cbm) = get_gap_info(atoms,
                                                    soc=soc,
                                                    direct=True,
                                                    calc=calc,
                                                    calculator=calculator)

    k_vbm, k_cbm = skn_vbm[1], skn_cbm[1]
    direct_k_vbm, direct_k_cbm = direct_skn_vbm[1], direct_skn_cbm[1]

    get_kc = partial(get_1bz_k, ibzkpts, calc)

    k_vbm_c = get_kc(k_vbm)
    k_cbm_c = get_kc(k_cbm)
    direct_k_vbm_c = get_kc(direct_k_vbm)
    direct_k_cbm_c = get_kc(direct_k_cbm)

    if soc:
        theta, phi = get_spin_axis(atoms, calculator=calculator)
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


def get_gap_info(atoms, soc, direct, calc, calculator):
    from ase.dft.bandgap import bandgap
    from asr.utils.gpw2eigs import calc2eigs
    from asr.magnetic_anisotropy import get_spin_axis
    # e1 is VBM, e2 is CBM
    if soc:
        theta, phi = get_spin_axis(atoms, calculator=calculator)
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
        'dipz': 'Out-of-plane dipole [e * Ang].',
        'evac1': 'Top side vacuum level [eV].',
        'evac2': 'Bottom side vacuum level [eV]',
        'evacmean': 'Average vacuum level [eV].',
        'efermi_nosoc': 'Fermi level without SOC [eV].'}


def vacuumlevels(atoms, calc, n=8):
    """Get the vacuumlevels on both sides of a 2D material.

    Get the vacuumlevels on both sides of a 2D material. Will
    do a dipole corrected dft calculation, if needed (Janus structures).
    Assumes the 2D material periodic directions are x and y.
    Assumes that the 2D material is centered in the z-direction of
    the unit cell.

    Dipole corrected dft calculation -> dipcorrgs.gpw

    Parameters
    ----------
    gpw: str
       name of gpw file to base the dipole corrected calc on
    evacdiffmin: float
        thresshold in eV for doing a dipole moment corrected
        dft calculations if the predicted evac difference is less
        than this value don't do it
    n: int
        number of gridpoints away from the edge to evaluate the vac levels
    """
    import numpy as np

    if not np.sum(atoms.get_pbc()) == 2:
        return VacuumLevelResults.fromdata(
            z_z=None,
            v_z=None,
            evacdiff=None,
            dipz=None,
            evac1=None,
            evac2=None,
            evacmean=None,
            efermi_nosoc=None)

    # Record electrostatic potential as a function of z
    v_z = calc.get_electrostatic_potential().mean(0).mean(0)
    z_z = np.linspace(0, atoms.cell[2, 2], len(v_z), endpoint=False)

    # Store data
    return VacuumLevelResults.fromdata(
        z_z=z_z,
        v_z=v_z,
        dipz=calc.atoms.get_dipole_moment()[2],
        evacdiff=evacdiff(calc.atoms),
        evac1=v_z[n],
        evac2=v_z[-n],
        evacmean=(v_z[n] + v_z[-n]) / 2,
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
        forces='Forces on atoms [eV/Angstrom].',
        stresses='Stress on unit cell [eV/Angstrom^dim].',
        evac='Vacuum level [eV].',
        evacdiff='Vacuum level shift (Vacuum level shift) [eV].',
        dipz='Out-of-plane dipole [e * Ang].',
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


def migrate_main_record(cache, calculateselection, mainselection):
    """Migrate asr.gs::main records to have parameters."""
    calculaterecord = cache.get(**calculateselection)
    original_record = cache.get(**mainselection)
    migrated_record = original_record.copy()
    parameters = calculaterecord.run_specification.parameters
    migrated_record.run_specification.parameters = parameters
    return original_record, migrated_record


def get_gs_main_migrations(cache):
    """Migrate ground state records."""
    from asr.core.migrate import Migration, get_resultfile_migration

    migrations = get_gs_calculate_migrations(cache)

    if migrations:
        calculateselection = {
            'run_specification.name': 'asr.gs::calculate',
            'migration_id': migrations[0].id,
        }
        migration = get_resultfile_migration('results-asr.gs.json')
        mainselection = {
            'run_specification.name': 'asr.gs::main',
            'migration_id': migration.id,
        }
        migrations.append(Migration(
            migrate_main_record,
            name="Add atoms and calculator to gs.main record.",
            args=(cache, calculateselection, mainselection)),
        )
    else:
        print('Found no matches!')
    return migrations


@command(module='asr.gs')
@option('-a', '--atoms', help='Atomic structure.',
        type=AtomsFile(), default='structure.json')
@option('-c', '--calculator', help='Calculator params.', type=DictStr())
def main(atoms: Atoms,
         calculator: dict = {
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
             'charge': 0
         }) -> Result:
    """Extract derived quantities from groundstate in gs.gpw."""
    calculaterecord = calculate(atoms=atoms, calculator=calculator)
    calc = calculaterecord.result.calculation.load(parallel=False)
    calc.atoms.calc = calc

    # Now that some checks are done, we can extract information
    forces = calc.get_property('forces', allow_calculation=False)
    stresses = calc.get_property('stress', allow_calculation=False)
    etot = calc.get_potential_energy()

    gaps_nosoc = gaps(atoms, calc, soc=False, calculator=calculator)
    gaps_soc = gaps(atoms, calc, soc=True, calculator=calculator)
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
