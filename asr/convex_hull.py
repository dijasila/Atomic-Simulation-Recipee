import json
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

import click

from ase.db import connect
from ase.io import read
from ase.phasediagram import PhaseDiagram
from ase.db.row import AtomsRow


@click.command()
@click.option('-r', '--references', type=str,
              help='Reference database.')
@click.option('-d', '--database', type=str,
              help='Database of systems to be included in the figure.')
def main(references: str, database: str):
    atoms = read('gs.gpw')
    energy = atoms.get_potential_energy()
    count = Counter(atoms.get_chemical_symbols())
    formula = atoms.get_chemical_formula()

    refdb = connect(references)
    rows = select_references(refdb, set(count))
    pdrefs = [(row.formula, row.energy / row.natoms) for row in rows]

    results = {'references': pdrefs}

    try:
        pd = PhaseDiagram(pdrefs)
    except ValueError:
        pass
    else:
        N = len(atoms)
        e0, _, _ = pd.decompose(formula)
        results['ehull'] = (energy - e0) / N

    ref_energies = {}
    for row in rows:
        if len(row.count_atoms()) == 1:
            symbol = row.symbols[0]
            assert symbol not in ref_energies
            ref_energies[symbol] = row.energy / row.natoms

    results['hform'] = (energy -
                        sum(n * ref_energies[symbol]
                            for symbol, n in count.items())
                        ) / len(atoms)

    if database:
        db = connect(database)
        rows = select_references(db, set(count))
        links = []
        for row in rows:
            hform = (row.energy -
                     sum(n * ref_energies[symbol]
                         for symbol, n in row.count_atoms().items())
                     ) / len(atoms)
            links.append((hform,
                          row.formula,
                          row.prototype,
                          row.magstate,
                          row.uid))
        results['links'] = links

    Path('convex_hull.json').write_text(json.dumps(results))


def select_references(db, symbols):
    refs: Dict[int, 'AtomsRow'] = {}
    for symbol in symbols:
        for row in db.select(symbol):
            for symb in row.count_atoms():
                if symb not in symbols:
                    break
            else:
                uid = row.get('uid', row.id)
                refs[uid] = row
    return list(refs.values())


def collect_data(atoms):
    path = Path('convex_hull.json')
    if not path.is_file():
        return {}, {}, {}
    dct = json.loads(path.read_text())
    kvp = {'hform': dct.pop('hform')}
    if 'ehull' in dct:
        kvp['ehull'] = dct.pop('ehull')
    return (kvp,
            {'ehull': ('Energy above convex hull', '', 'eV/atom'),
             'hform': ('Heat of formation', '', 'eV/atom')},
            {'convex_hull': dct})


def plot(row, fname):
    from ase.phasediagram import PhaseDiagram, parse_formula
    import matplotlib.pyplot as plt

    data = row.data.convex_hull

    count = row.count_atoms()
    if not (2 <= len(count) <= 3):
        return

    refs = data['references']
    pd = PhaseDiagram(refs, verbose=False)

    fig = plt.figure()
    ax = fig.gca()

    if len(count) == 2:
        x, e, names, hull, simplices, xlabel, ylabel = pd.plot2d2()
        for i, j in simplices:
            ax.plot(x[[i, j]], e[[i, j]], '-', color='lightblue')
        ax.plot(x, e, 's', color='C0', label='Bulk')
        dy = e.ptp() / 30
        for a, b, name in zip(x, e, names):
            ax.text(a, b - dy, name, ha='center', va='top')
        A, B = pd.symbols
        ax.set_xlabel('{}$_{{1-x}}${}$_x$'.format(A, B))
        ax.set_ylabel(r'$\Delta H$ [eV/atom]')
        label = '2D'
        ymin = e.min()
        for y, formula, prot, magstate, uid in data.links:
            count = parse_formula(formula)[0]
            x = count[B] / sum(count.values())
            if uid == row.uid:
                ax.plot([x], [y], 'rv', label=label)
                ax.plot([x], [y], 'ko', ms=15, fillstyle='none')
            else:
                ax.plot([x], [y], 'v', color='C1', label=label)
            label = None
            ax.text(x + 0.03, y, '{}-{}'.format(prot, magstate))
            ymin = min(ymin, y)
        ax.axis(xmin=-0.1, xmax=1.1, ymin=ymin - 2.5 * dy)
    else:
        x, y, names, hull, simplices = pd.plot2d3()
        for i, j, k in simplices:
            ax.plot(x[[i, j, k, i]], y[[i, j, k, i]], '-', color='lightblue')
        ax.plot(x[hull], y[hull], 's', color='C0', label='Bulk (on hull)')
        ax.plot(x[~hull], y[~hull], 's', color='C2', label='Bulk (above hull)')
        for a, b, name in zip(x, y, names):
            ax.text(a - 0.02, b, name, ha='right', va='top')
        A, B, C = pd.symbols
        label = '2D'
        for e, formula, prot, magstate, id, uid in data.links:
            count = parse_formula(formula)[0]
            x = count.get(B, 0) / sum(count.values())
            y = count.get(C, 0) / sum(count.values())
            x += y / 2
            y *= 3**0.5 / 2
            if id == row.id:
                ax.plot([x], [y], 'rv', label=label)
                ax.plot([x], [y], 'ko', ms=15, fillstyle='none')
            else:
                ax.plot([x], [y], 'v', color='C1', label=label)
            label = None
        plt.axis('off')

    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def convex_hull_tables(row: AtomsRow,
                       project: str = 'c2db',
                       ) -> List[Dict[str, Any]]:
    from ase.symbols import string2symbols
    data = row.data.convex_hull

    rows = []
    for e, formula, prot, magstate, uid in sorted(data.links,
                                                  reverse=True):
        name = '{} ({}-{})'.format(formula, prot, magstate)
        if id != row.id:
            name = '<a href="/{}/row/{}">{}</a>'.format(project, uid, name)
        rows.append([name, '{:.3f} eV/atom'.format(e)])

    refs = data.references
    bulkrows = []
    for formula, e in refs:
        e /= len(string2symbols(formula))
        link = '<a href="/oqmd12/row/{formula}">{formula}</a>'.format(
            formula=formula)
        bulkrows.append([link, '{:.3f} eV/atom'.format(e)])

    return [{'type': 'table',
             'header': ['Monolayer formation energies', ''],
             'rows': rows},
            {'type': 'table',
             'header': ['Bulk formation energies', ''],
             'rows': bulkrows}]


def webpanel(row, key_descriptions):
    from asr.custom import fig
    from asr.custom import table

    prefix = key_descriptions.get('prefix', '')
    if 'c2db-' in prefix:  # make sure links to other rows just works!
        projectname = 'c2db'
    else:
        projectname = 'default'

    hulltable1 = table(row,
                       'Property',
                       ['hform', 'ehull', 'minhessianeig'],
                       key_descriptions)
    hulltable2, hulltable3 = convex_hull_tables(row, projectname)

    panel = ('Stability',
             [[fig('convex-hull.png')],
              [hulltable1, hulltable2, hulltable3]])

    things = [(plot, ['convex-hull.png'])]

    return panel, things


group = 'Property'
dependencies = ['asr.gs']


if __name__ == '__main__':
    main()
