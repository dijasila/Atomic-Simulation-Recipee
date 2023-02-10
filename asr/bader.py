"""Bader charge analysis."""
import numpy as np

from typing import List

from asr.core import command, option, ASRResult, prepare_result

from asr.database.browser import (
    describe_entry,
    entry_parameter_description,
    make_panel_description, href)

panel_description = make_panel_description(
    """The Bader charge analysis ascribes a net charge to an atom
by partitioning the electron density according to its zero-flux surfaces.""",
    articles=[
        href("""W. Tang et al. A grid-based Bader analysis algorithm
without lattice bias. J. Phys.: Condens. Matter 21, 084204 (2009).""",
             'https://doi.org/10.1088/0953-8984/21/8/084204')])


def webpanel(result, row, key_descriptions):
    rows = [[str(a), symbol, f'{charge:.2f}']
            for a, (symbol, charge)
            in enumerate(zip(result.sym_a, result.bader_charges))]
    table = {'type': 'table',
             'header': ['Atom index', 'Atom type', 'Charge (e)'],
             'rows': rows}

    parameter_description = entry_parameter_description(
        row.data,
        'asr.bader')

    title_description = panel_description + parameter_description

    panel = {'title': describe_entry('Bader charges',
                                     description=title_description),
             'columns': [[table]]}

    return [panel]


@prepare_result
class Result(ASRResult):

    bader_charges: np.ndarray
    sym_a: List[str]

    key_descriptions = {'bader_charges': 'Array of charges [\\|e\\|].',
                        'sym_a': 'Chemical symbols.'}

    formats = {"ase_webpanel": webpanel}


@command('asr.bader',
         dependencies=['asr.gs'],
         returns=Result)
@option('--grid-spacing', help='Grid spacing (Å)', type=float)
def main(grid_spacing: float = 0.05) -> Result:
    """Calculate bader charges.

    To make Bader analysis we use another program. Download the executable
    for Bader analysis and put in path (this is for Linux, find the
    appropriate executable for you own OS)

        $ mkdir baderext && cd baderext
        $ wget theory.cm.utexas.edu/henkelman/code/bader/download/
        ...bader_lnx_64.tar.gz
        $ tar -xf bader_lnx_64.tar.gz
        $ echo 'export PATH=~/baderext:$PATH' >> ~/.bashrc
    """
    from pathlib import Path
    import subprocess
    from ase.io import write
    from ase.units import Bohr
    from gpaw.new.ase_interface import GPAW
    from gpaw.mpi import world
    from gpaw.utilities.bader import read_bader_charges

    assert world.size == 1, 'Do not run in parallel!'

    gs = GPAW('gs.gpw')
    dens = gs.calculation.densities()
    n_sR = dens.all_electron_densities(grid_spacing=grid_spacing)
    write('density.cube', gs.atoms, data=n_sR.data.sum(axis=0) * Bohr**3)

    cmd = 'bader density.cube'
    out = Path('bader.out').open('w')
    err = Path('bader.err').open('w')
    subprocess.run(cmd.split(),
                   stdout=out,
                   stderr=err)
    out.close()
    err.close()

    charges = -read_bader_charges('ACF.dat')
    charges += gs.atoms.get_atomic_numbers()
    assert abs(charges.sum()) < 0.01

    sym_a = gs.atoms.get_chemical_symbols()

    return Result(data=dict(bader_charges=charges,
                            sym_a=sym_a))


if __name__ == '__main__':
    main.cli()
