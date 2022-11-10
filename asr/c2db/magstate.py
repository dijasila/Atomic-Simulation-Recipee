"""Module for determining magnetic state."""
import asr
from asr.core import (command, ASRResult, prepare_result, option,
                      AtomsFile)
from ase import Atoms
import typing

atomic_mom_threshold = 0.1
total_mom_threshold = 0.1


def get_magstate(calc):
    """Determine the magstate of calc."""
    magmoms = calc.get_property('magmoms', allow_calculation=False)

    if magmoms is None:
        return 'nm'

    maximum_mom = abs(magmoms).max()
    if maximum_mom < atomic_mom_threshold:
        return 'nm'

    magmom = calc.get_magnetic_moment()

    if abs(magmom) < total_mom_threshold and maximum_mom > atomic_mom_threshold:
        return 'afm'

    return 'fm'


def webpanel(result, context):
    """Webpanel for magnetic state."""
    from asr.database.browser import describe_entry, dl, code, WebPanel

    is_magnetic = describe_entry(
        'Magnetic',
        'Is material magnetic?'
        + dl(
            [
                [
                    'Magnetic',
                    code('if max(abs(atomic_magnetic_moments)) > '
                         f'{atomic_mom_threshold}')
                ],
                [
                    'Not magnetic',
                    code('otherwise'),
                ],
            ]
        )
    )

    rows = [[is_magnetic, context.is_magnetic]]
    summary = {'title': 'Summary',
               'columns': [[{'type': 'table',
                             'header': ['Electronic properties', ''],
                             'rows': rows}]],
               'sort': 0}

    atoms = context.atoms

    if result.magstate == 'NM':
        return [summary]
    else:
        assert len(atoms) == len(result.magmoms)
        magmoms_rows = [[str(a), symbol, f'{magmom:.2f}']
                        for a, (symbol, magmom)
                        in enumerate(zip(atoms.symbols, result.magmoms))]
        magmoms_table = {'type': 'table',
                         'header': ['Atom index', 'Atom type',
                                    'Local magnetic moment (μ<sub>B</sub>)'],
                         'rows': magmoms_rows}

        xcname = context.xcname
        panel = WebPanel(title=f'Basic magnetic properties ({xcname})',
                         columns=[[], [magmoms_table]],
                         sort=11)

        return [summary, panel]


@prepare_result
class Result(ASRResult):

    magstate: str
    is_magnetic: bool
    magmoms: typing.List[float]
    magmom: float
    nspins: int

    key_descriptions = {'magstate': 'Magnetic state.',
                        'is_magnetic': 'Is the material magnetic?',
                        'magmoms': 'Atomic magnetic moments.',
                        'magmom': 'Total magnetic moment.',
                        'nspins': 'Number of spins in system.'}
    formats = {"webpanel2": webpanel}


@command('asr.c2db.magstate')
@option('-a', '--atoms', help='Atomic structure.',
        type=AtomsFile(), default='structure.json')
@asr.calcopt
def main(atoms: Atoms,
         calculator: dict = {
             'name': 'gpaw',
             'mode': {'name': 'pw', 'ecut': 800},
             'xc': 'PBE',
             'kpts': {'density': 12.0, 'gamma': True},
             'occupations': {'name': 'fermi-dirac',
                             'width': 0.05},
             'convergence': {'bands': 'CBM+3.0'},
             'nbands': '200%',
             'txt': 'gs.txt',
             'charge': 0
         }) -> Result:
    """Determine magnetic state."""
    from asr.c2db.gs import calculate as calculategs

    calculateresult = calculategs(atoms=atoms, calculator=calculator)
    calc = calculateresult.calculation.load()
    magstate = get_magstate(calc)
    magmoms = calc.get_property('magmoms', allow_calculation=False)
    magmom = calc.get_property('magmom', allow_calculation=False)
    nspins = calc.get_number_of_spins()
    results = {'magstate': magstate.upper(),
               'is_magnetic': magstate != 'nm',
               'magmoms': magmoms,
               'magmom': magmom,
               'nspins': nspins}

    return Result(data=results)


if __name__ == '__main__':
    main.cli()
