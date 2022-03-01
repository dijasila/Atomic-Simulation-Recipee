"""Convex hull stability analysis."""
from collections import Counter
from typing import List, Dict, Any, Optional

import asr
from asr.core import (
    command, argument, ASRResult, prepare_result,
    atomsopt, calcopt,
    File, FileStr,
)
from asr.calculators import set_calculator_hook

import numpy as np
from asr.database.browser import (
    fig, table, describe_entry, dl, br, make_panel_description
)

from ase import Atoms
from ase.phasediagram import PhaseDiagram
from ase.db import connect
from ase.db.row import AtomsRow
from ase.formula import Formula

# from matplotlib.legend_handler import HandlerPatch
from matplotlib import patches
# from matplotlib.legend_handler import HandlerLine2D, HandlerTuple


from asr.c2db.gs import main as groundstate

known_methods = ['DFT', 'DFT+D3']


def get_hull_energies(pd: PhaseDiagram):
    hull_energies = []
    for ref in pd.references:
        count = ref[0]
        refenergy = ref[1]
        natoms = ref[3]
        decomp_energy, indices, coefs = pd.decompose(**count)
        ehull = (refenergy - decomp_energy) / natoms
        hull_energies.append(ehull)

    return hull_energies


panel_description = make_panel_description(
    """The heat of formation (ΔH) is the internal energy of a compound relative to
the standard states of the constituent elements at T=0 K.  The energy above the
convex hull is the internal energy relative to the most stable (possibly mixed)
phase of the constituent elements at T=0 K.""",
    articles=['C2DB'],
)


def webpanel(result, context):
    hulltable1 = table(result,
                       'Stability',
                       ['hform', 'ehull'],
                       context.descriptions)
    hulltables = convex_hull_tables(context.record)
    panel = {
        'title': describe_entry(
            'Thermodynamic stability', panel_description),
        'columns': [[fig('convex-hull.png')],
                    [hulltable1] + hulltables],
        'plot_descriptions': [{'function': plot,
                               'filenames': ['convex-hull.png']}],
        'sort': 1,
    }

    thermostab = result['thermodynamic_stability_level']

    stability_texts = [
        [stability_names[stab], stability_descriptions[stab]]
        for stab in [LOW, MEDIUM, HIGH]
    ]

    thermodynamic = describe_entry(
        'Thermodynamic',
        'Classifier for the thermodynamic stability of a material.'
        + br
        + dl(stability_texts)
    )
    row = [thermodynamic, stability_names[thermostab]]

    summary = {'title': 'Summary',
               'columns': [[{'type': 'table',
                             'header': ['Stability', ''],
                             'rows': [row],
                             'columnwidth': 3}]],
               'sort': 1}
    return [panel, summary]


@prepare_result
class Result(ASRResult):

    ehull: float
    hform: float
    references: List[dict]
    thermodynamic_stability_level: str
    coefs: Optional[List[float]]
    indices: Optional[List[int]]
    key_descriptions = {
        "ehull": "Energy above convex hull [eV/atom].",
        "hform": "Heat of formation [eV/atom].",
        "thermodynamic_stability_level": "Thermodynamic stability level.",
        "references": "List of relevant references.",
        "indices":
        "Indices of references that this structure will decompose into.",
        "coefs": "Fraction of decomposing references (see indices doc).",
    }

    formats = {'webpanel2': webpanel}


def convert_database_strings_to_files(parameters):
    databases = []
    for database in parameters.databases:
        if isinstance(database, str):
            database = File.fromstr(database)
        databases.append(database)
    parameters.databases = databases
    return parameters


def select_records_with_databases_as_string(record):
    if record.name != 'asr.c2db.convex_hull:main':
        return False
    for database in record.parameters.databases:
        if isinstance(database, str):
            return True
    return False


@asr.mutation(selector=select_records_with_databases_as_string)
def convert_database_parameter_to_file(record):
    """Convert databases represented as strings to File objects."""
    try:
        parameters = convert_database_strings_to_files(record.parameters)
    except FileNotFoundError:
        raise asr.NonMigratableRecord
    record.parameters = parameters
    return record


@command('asr.c2db.convex_hull',
         argument_hooks=[set_calculator_hook,
                         convert_database_strings_to_files],
         )
@atomsopt
@calcopt
@argument('databases', nargs=-1, type=FileStr())
def main(
        atoms: Atoms,
        calculator: dict = groundstate.defaults.calculator,
        databases: List[File] = [],
) -> Result:
    """Calculate convex hull energies.

    It is assumed that the first database supplied is the one containing the
    standard references.

    For a database to be a valid reference database each row has to have a
    "uid" key-value-pair. Additionally, it is required that the metadata of
    each database contains following keys:

        - title: Title of the reference database.
        - legend: Collective label for all references in the database to
          put on the convex hull figures.
        - name: f-string from which to derive name for a material.
        - link: f-string from which to derive an url for a material
          (see further information below).
        - label: f-string from which to derive a material specific name to
          put on convex hull figure.
        - method: String denoting the method that was used to calculate
          reference energies. Currently accepted strings: ['DFT', 'DFT+D3'].
          "DFT" means bare DFT references energies. "DFT+D3" indicate that the
          reference also include the D3 dispersion correction.
        - energy_key (optional): Indicates the key-value-pair that represents
          the total energy of a material from. If not specified the
          default value of 'energy' will be used.

    The name and link keys are given as f-strings and can this refer to
    key-value-pairs in the given database. For example, valid metadata looks
    like:

    .. code-block:: javascript

        {
            "title": "Bulk reference phases",
            "legend": "Bulk",
            "name": "{row.formula}",
            "link": "https://cmrdb.fysik.dtu.dk/oqmd12/row/{row.uid}",
            "label": "{row.formula}",
            "method": "DFT",
            "energy_key": "total_energy"
        }

    Parameters
    ----------
    databases : list of str
        List of filenames of databases.

    """
    # XXX Add possibility of D3 correction again
    # TODO: Make separate recipe for calculating vdW correction to total energy
    databases = [connect(database.path) for database in databases]
    result = groundstate(atoms=atoms, calculator=calculator)
    usingd3 = False
    energy = result.etot

    if usingd3:
        mymethod = 'DFT+D3'
    else:
        mymethod = 'DFT'

    formula = atoms.get_chemical_formula()
    count = Counter(atoms.get_chemical_symbols())

    dbdata = {}
    reqkeys = {'title', 'legend', 'name', 'link', 'label', 'method'}
    for refdb in databases:
        # Connect to databases and save relevant rows
        metadata = refdb.metadata
        assert not (reqkeys - set(metadata)), \
            'Missing some essential metadata keys.'

        dbmethod = metadata['method']
        assert dbmethod in known_methods, f'Unknown method: {dbmethod}'
        assert dbmethod == mymethod, \
            ('You are using a reference database with '
             f'inconsistent methods: {mymethod} (this material) != '
             f'{dbmethod} ({refdb.database})')

        rows = []
        # Select only references which contain relevant elements
        rows.extend(select_references(refdb, set(count)))
        dbdata[refdb.filename] = {
            'rows': rows,
            'metadata': metadata,
        }

    ref_database = databases[0]
    ref_metadata = dbdata[ref_database.filename]['metadata']
    ref_energy_key = ref_metadata.get('energy_key', 'energy')
    ref_energies = get_reference_energies(
        atoms,
        ref_database,
        energy_key=ref_energy_key,
    )
    hform = hof(energy,
                count,
                ref_energies)
    # Make a list of the relevant references
    references = []
    for data in dbdata.values():
        metadata = data['metadata']
        energy_key = metadata.get('energy_key', 'energy')
        for row in data['rows']:
            hformref = hof(row[energy_key],
                           row.count_atoms(), ref_energies)
            reference = {'hform': hformref,
                         'formula': row.formula,
                         'uid': row.uid,
                         'natoms': row.natoms}
            reference.update(metadata)
            if 'name' in reference:
                reference['name'] = reference['name'].format(row=row)
            if 'label' in reference:
                reference['label'] = reference['label'].format(row=row)
            if 'link' in reference:
                reference['link'] = reference['link'].format(row=row)
            references.append(reference)

    pdrefs = []
    for reference in references:
        h = reference['natoms'] * reference['hform']
        pdrefs.append((reference['formula'], h))

    results = {'hform': hform,
               'references': references}

    if len(count) == 1:
        ehull = hform
        results['indices'] = None
        results['coefs'] = None
    else:
        pd = PhaseDiagram(pdrefs, verbose=False)
        e0, indices, coefs = pd.decompose(formula)
        ehull = hform - e0 / len(atoms)
        results['indices'] = indices.tolist()
        results['coefs'] = coefs.tolist()

    results['ehull'] = ehull
    results['thermodynamic_stability_level'] = stability_rating(hform, ehull)
    return Result(data=results)


LOW = 1
MEDIUM = 2
HIGH = 3
stability_names = {LOW: 'LOW', MEDIUM: 'MEDIUM', HIGH: 'HIGH'}
stability_descriptions = {
    LOW: 'Heat of formation > 0.2 eV/atom',
    MEDIUM: 'convex hull + 0.2 eV/atom < Heat of formation < 0.2 eV/atom',
    HIGH: 'Heat of formation < convex hull + 0.2 eV/atom'}


def stability_rating(hform, ehull):
    if 0.2 < hform:
        return LOW
    if ehull + 0.2 < hform:
        return MEDIUM
    return HIGH


def get_reference_energies(atoms, refdb, energy_key='energy'):
    count = Counter(atoms.get_chemical_symbols())

    # Get reference energies
    ref_energies = {}
    for row in select_references(refdb, set(count)):
        if len(row.count_atoms()) == 1:
            symbol = row.symbols[0]
            e_ref = row[energy_key] / row.natoms
            assert symbol not in ref_energies
            ref_energies[symbol] = e_ref

    return ref_energies


def hof(energy, count, ref_energies):
    """Heat of formation."""
    energy = energy - sum(n * ref_energies[symbol]
                          for symbol, n in count.items())
    return energy / sum(count.values())


def select_references(db, symbols):
    refs: Dict[int, 'AtomsRow'] = {}

    for symbol in symbols:
        for row in db.select(symbol):
            for symb in row.count_atoms():
                if symb not in symbols:
                    break
            else:
                uid = row.get('uid')
                refs[uid] = row
    return list(refs.values())


class ObjectHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = patches.Polygon(
            [
                [x0, y0],
                [x0, y0 + height],
                [x0 + 3 / 4 * width, y0 + height],
                [x0 + 1 / 4 * width, y0],
            ],
            closed=True, facecolor='C2',
            edgecolor='none', lw=3,
            transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        patch = patches.Polygon(
            [
                [x0 + width, y0],
                [x0 + 1 / 4 * width, y0],
                [x0 + 3 / 4 * width, y0 + height],
                [x0 + width, y0 + height],
            ],
            closed=True, facecolor='C3',
            edgecolor='none', lw=3,
            transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch


def plot(context, fname):
    from ase.phasediagram import PhaseDiagram
    import matplotlib.pyplot as plt

    data = context.result
    atoms = context.atoms

    nspecies = len(atoms.symbols.species())
    if not (2 <= nspecies <= 3):
        return

    references = data['references']
    pdrefs = []
    legends = []
    sizes = []

    for reference in references:
        h = reference['natoms'] * reference['hform']
        pdrefs.append((reference['formula'], h))
        legend = reference.get('legend')
        if legend and legend not in legends:
            legends.append(legend)
        if legend in legends:
            idlegend = legends.index(reference['legend'])
            size = (3 * idlegend + 3)**2
        else:
            size = 2
        sizes.append(size)
    sizes = np.array(sizes)

    pd = PhaseDiagram(pdrefs,
                      verbose=False)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.gca()

    legendhandles = []

    for it, label in enumerate(['On hull', 'off hull']):
        handle = ax.fill_between([], [],
                                 color=f'C{it + 2}', label=label)
        legendhandles.append(handle)

    for it, legend in enumerate(legends):
        handle = ax.scatter([], [], facecolor='none', marker='o',
                            edgecolor='k', label=legend, s=(3 + it * 3)**2)
        legendhandles.append(handle)

    hull_energies = get_hull_energies(pd)

    if nspecies == 2:
        xcoord, energy, _, hull, simplices, xlabel, ylabel = pd.plot2d2()
        hull = np.array(hull_energies) < 0.05
        edgecolors = np.array(['C2' if hull_energy < 0.05 else 'C3'
                               for hull_energy in hull_energies])
        for i, j in simplices:
            ax.plot(xcoord[[i, j]], energy[[i, j]], '-', color='C0')
        names = [ref['label'] for ref in references]

        if data['hform'] < 0:
            mask = energy < 0.05
            energy = energy[mask]
            xcoord = xcoord[mask]
            edgecolors = edgecolors[mask]
            hull = hull[mask]
            sizes = sizes[mask]
            names = [name for name, m in zip(names, mask) if m]

        xcoord0 = xcoord[~hull]
        energy0 = energy[~hull]
        ax.scatter(
            xcoord0, energy0,
            # x[~hull], e[~hull],
            facecolor='none', marker='o',
            edgecolor=np.array(edgecolors)[~hull], s=sizes[~hull],
            zorder=9)

        ax.scatter(
            xcoord[hull], energy[hull],
            facecolor='none', marker='o',
            edgecolor=np.array(edgecolors)[hull], s=sizes[hull],
            zorder=10)

        # ax.scatter(x, e, facecolor='none', marker='o', edgecolor=colors)

        delta = energy.ptp() / 30
        for a, b, name, on_hull in zip(xcoord, energy, names, hull):
            va = 'center'
            ha = 'left'
            dy = 0
            dx = 0.02
            ax.text(a + dx, b + dy, name, ha=ha, va=va)

        A, B = pd.symbols
        ax.set_xlabel('{}$_{{1-x}}${}$_x$'.format(A, B))
        ax.set_ylabel(r'$\Delta H$ [eV/atom]')

        # Circle this material
        ymin = energy.min()
        ax.axis(xmin=-0.1, xmax=1.1, ymin=ymin - 2.5 * delta)
        newlegendhandles = [(legendhandles[0], legendhandles[1]),
                            *legendhandles[2:]]

        plt.legend(
            newlegendhandles,
            [r'$E_\mathrm{h} {^</_>}\, 5 \mathrm{meV}$',
             *legends], loc='lower left', handletextpad=0.5,
            handler_map={tuple: ObjectHandler()},
        )
    else:
        x, y, _, hull, simplices = pd.plot2d3()

        hull = np.array(hull)
        hull = np.array(hull_energies) < 0.05
        names = [ref['label'] for ref in references]
        latexnames = []
        for name in names:
            try:
                pretty = format(
                    Formula(name.split(' ')[0]).reduce()[0],
                    'latex'
                )
            except ValueError:
                pretty = name
            latexnames.append(pretty)

        for i, j, k in simplices:
            ax.plot(x[[i, j, k, i]], y[[i, j, k, i]], '-', color='lightblue')
        edgecolors = ['C2' if hull_energy < 0.05 else 'C3'
                      for hull_energy in hull_energies]
        ax.scatter(
            x[~hull], y[~hull],
            facecolor='none', marker='o',
            edgecolor=np.array(edgecolors)[~hull], s=sizes[~hull],
            zorder=9,
        )

        ax.scatter(
            x[hull], y[hull],
            facecolor='none', marker='o',
            edgecolor=np.array(edgecolors)[hull], s=sizes[hull],
            zorder=10,
        )

        printed_names = set()
        thisformula = atoms.symbols.formula
        thisname = format(thisformula, 'latex')
        comps = thisformula.count().keys()
        for a, b, name, on_hull, hull_energy in zip(
                x, y, latexnames, hull, hull_energies):
            if name in [
                    thisname, *comps,
            ] and name not in printed_names:
                printed_names.add(name)
                ax.text(a - 0.02, b, name, ha='right', va='top')

        newlegendhandles = [(legendhandles[0], legendhandles[1]),
                            *legendhandles[2:]]
        plt.legend(
            newlegendhandles,
            [r'$E_\mathrm{h} {^</_>}\, 5 \mathrm{meV}$',
             *legends], loc='upper right', handletextpad=0.5,
            handler_map={tuple: ObjectHandler()},
        )
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def convex_hull_tables(record) -> List[Dict[str, Any]]:
    data = record.result

    # XXX What about these references?
    references = data['references']
    tables = {}
    for reference in references:
        tables[reference['title']] = []

    for reference in sorted(references, reverse=True,
                            key=lambda x: x['hform']):
        name = reference['name']
        matlink = reference['link']
        if reference['uid'] != record.uid:
            name = f'<a href="{matlink}">{name}</a>'
        e = reference['hform']
        tables[reference['title']].append([name, '{:.2f} eV/atom'.format(e)])

    final_tables = []
    for title, rows in tables.items():
        final_tables.append({'type': 'table',
                             'header': [title, ''],
                             'rows': rows})
    return final_tables
