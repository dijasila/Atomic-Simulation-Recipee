"""Convex hull stability analysis."""
from collections import Counter
from typing import List, Dict, Any, Optional
from pathlib import Path
import functools

import numpy as np

from asr.core import command, argument, ASRResult, prepare_result
from asr.database.browser import (
    fig, table, describe_entry, make_panel_description
)

from ase.db import connect
from ase.io import read
from ase.phasediagram import PhaseDiagram
from ase.db.row import AtomsRow
from ase.formula import Formula

# from matplotlib.legend_handler import HandlerPatch
from matplotlib import patches
# from matplotlib.legend_handler import HandlerLine2D, HandlerTuple


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


eform_description = """\
The heat of formation (ΔH) is the internal energy of a compound relative to
the standard states of the constituent elements at T=0 K."""


ehull_description = """\
The energy above the convex hull is the internal energy relative to the most
stable (possibly mixed) phase of the constituent elements at T=0 K."""

panel_description = make_panel_description(
    '{eform_description}\n\n{ehull_description}',
    articles=['C2DB'],
)


def webpanel(result, row, key_descriptions):
    hulltable1 = table(row,
                       'Stability',
                       ['hform', 'ehull'],
                       key_descriptions)
    hulltables = convex_hull_tables(row)
    panel = {
        'title': describe_entry(
            'Thermodynamic stability', panel_description),
        'columns': [[fig('convex-hull.png')],
                    [hulltable1] + hulltables],
        'plot_descriptions': [{'function':
                               functools.partial(plot, thisrow=row),
                               'filenames': ['convex-hull.png']}],
        'sort': 1,
    }

    return [panel]


# XXX This string is hardcoded also in c2db's search html file in cmr
# repository (with different formatting).
# cmr could probably import the string from here instead.
ehull_long_description = """\
The energy above the convex hull (or the decomposition energy) is the main
descriptor for thermodynamic stability. It represents the energy/atom of the
material relative to the most stable, possibly mixed phase of the material.
The latter is evaluated using a \
<a href="https://cmrdb.fysik.dtu.dk/oqmd123/">reference database of bulk \
materials</a>.
For more information see Sec. 2.3 in \
<a href="https://iopscience.iop.org/article/10.1088/2053-1583/aacfc1"> \
Haastrup <i>et al</i>.</a>
"""


# This is for the c2db Summary panel.  We actually define most of that panel
# in the structureinfo.py
def ehull_table_rows(row, key_descriptions):
    ehull_table = table(row, 'Stability', ['ehull', 'hform'], key_descriptions)

    # We have to magically hack a description into the arbitrarily
    # nested "table" *grumble*:
    rows = ehull_table['rows']
    if len(rows) == 2:
        # ehull and/or hform may be missing if we run tests.
        # Dangerous and hacky, as always.
        rows[0][0] = describe_entry(rows[0][0], ehull_long_description)
        rows[1][0] = describe_entry(rows[1][0], eform_description)
    return ehull_table


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

    formats = {"ase_webpanel": webpanel}


@command('asr.convex_hull',
         requires=['results-asr.structureinfo.json',
                   'results-asr.database.material_fingerprint.json'],
         dependencies=['asr.structureinfo',
                       'asr.database.material_fingerprint'],
         returns=Result)
@argument('databases', nargs=-1, type=str)
def main(databases: List[str]) -> Result:
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
    from asr.relax import main as relax
    from asr.gs import main as groundstate
    from asr.core import read_json
    atoms = read('structure.json')

    if not relax.done:
        if not groundstate.done:
            groundstate()

    # TODO: Make separate recipe for calculating vdW correction to total energy
    for filename in ['results-asr.relax.json', 'results-asr.gs.json']:
        if Path(filename).is_file():
            results = read_json(filename)
            energy = results.get('etot')
            usingd3 = results.metadata.params.get('d3', False)
            break

    if usingd3:
        mymethod = 'DFT+D3'
    else:
        mymethod = 'DFT'

    formula = atoms.get_chemical_formula()
    count = Counter(atoms.get_chemical_symbols())

    dbdata = {}
    reqkeys = {'title', 'legend', 'name', 'link', 'label', 'method'}
    for database in databases:
        # Connect to databases and save relevant rows
        refdb = connect(database)
        metadata = refdb.metadata
        assert not (reqkeys - set(metadata)), \
            'Missing some essential metadata keys.'

        dbmethod = metadata['method']
        assert dbmethod in known_methods, f'Unknown method: {dbmethod}'
        assert dbmethod == mymethod, \
            ('You are using a reference database with '
             f'inconsistent methods: {mymethod} (this material) != '
             f'{dbmethod} ({database})')

        rows = []
        # Select only references which contain relevant elements
        rows.extend(select_references(refdb, set(count)))
        dbdata[database] = {'rows': rows,
                            'metadata': metadata}

    ref_database = databases[0]
    ref_metadata = dbdata[ref_database]['metadata']
    ref_energy_key = ref_metadata.get('energy_key', 'energy')
    ref_energies_per_atom = get_singlespecies_reference_energies_per_atom(
        atoms, ref_database, energy_key=ref_energy_key)

    # Make a list of the relevant references
    references = []
    for data in dbdata.values():
        metadata = data['metadata']
        energy_key = metadata.get('energy_key', 'energy')
        for row in data['rows']:
            hformref = hof(row[energy_key],
                           row.count_atoms(), ref_energies_per_atom)
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

    assert len(atoms) == len(Formula(formula))
    return calculate_hof_and_hull(formula, energy, references,
                                  ref_energies_per_atom)


def calculate_hof_and_hull(
        formula, energy, references, ref_energies_per_atom):
    formula = Formula(formula)

    species_counts = formula.count()

    hform = hof(energy,
                species_counts,
                ref_energies_per_atom)

    pdrefs = []
    for reference in references:
        h = reference['natoms'] * reference['hform']
        pdrefs.append((reference['formula'], h))

    results = {'hform': hform,
               'references': references}

    pd = PhaseDiagram(pdrefs, verbose=False)
    e0, indices, coefs = pd.decompose(str(formula))
    ehull = hform - e0 / len(formula)
    if len(species_counts) == 1:
        assert abs(ehull - hform) < 1e-10

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


def stability_rating(hform, energy_above_hull):
    assert hform <= energy_above_hull
    if 0.2 < hform:
        return LOW
    if 0.2 < energy_above_hull:
        return MEDIUM
    return HIGH


def get_singlespecies_reference_energies_per_atom(
        atoms, references, energy_key='energy'):

    # Get reference energies
    ref_energies_per_atom = {}
    refdb = connect(references)
    for row in select_references(refdb, set(atoms.symbols)):
        if len(row.count_atoms()) == 1:
            symbol = row.symbols[0]
            e_ref = row[energy_key] / row.natoms
            assert symbol not in ref_energies_per_atom
            ref_energies_per_atom[symbol] = e_ref

    return ref_energies_per_atom


def hof(energy, count, ref_energies_per_atom):
    """Heat of formation."""
    energy = energy - sum(n * ref_energies_per_atom[symbol]
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


def plot(row, fname, thisrow):
    from ase.phasediagram import PhaseDiagram
    import matplotlib.pyplot as plt

    data = row.data['results-asr.convex_hull.json']

    count = row.count_atoms()
    if not (2 <= len(count) <= 3):
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

    pd = PhaseDiagram(pdrefs, verbose=False)

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

    if len(count) == 2:
        xcoord, energy, _, hull, simplices, xlabel, ylabel = pd.plot2d2()
        hull = np.array(hull_energies) < 0.05
        edgecolors = np.array(['C2' if hull_energy < 0.05 else 'C3'
                               for hull_energy in hull_energies])
        for i, j in simplices:
            ax.plot(xcoord[[i, j]], energy[[i, j]], '-', color='C0')
        names = [ref['label'] for ref in references]

        if row.hform < 0:
            mask = energy < 0.05
            energy = energy[mask]
            xcoord = xcoord[mask]
            edgecolors = edgecolors[mask]
            hull = hull[mask]
            names = [name for name, m in zip(names, mask) if m]
            sizes = sizes[mask]

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
        latexnames = [
            format(
                Formula(name.split(' ')[0]).reduce()[0],
                'latex'
            )
            for name in names
        ]
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
        thisformula = Formula(thisrow.formula)
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


def plot_html(row, fname, thisrow):
    from ase.phasediagram import PhaseDiagram
    import matplotlib.pyplot as plt
    import plotly
    import plotly.graph_objs as go
    import plotly.express as px
    import pandas 

    colors = px.colors.qualitative.D3

    data = row.data['results-asr.convex_hull.json']

    count = row.count_atoms()
    if not (2 <= len(count) <= 3):
        return

    references = data['references']

    df_ref = pandas.DataFrame(references)

    # Why are we even using thisrow? Isn't it the same as row?
    df_ref['thisrow']= df_ref.apply(lambda x: True if x['uid'] == thisrow.uid else False, axis=1)


    names = [ref['label'] for ref in references]
    latexnames = [
        format(
            Formula(name.split(' ')[0]).reduce()[0],
            'latex'
        )
        for name in names
    ]

    latexnames = [r'{}'.format(s) for s in latexnames]
    df_ref['latexname'] = latexnames

    # Highlight this material by making it bold
    name_coloumn_to_plot = 'latexname'
    df_ref.loc[df_ref['thisrow'], name_coloumn_to_plot] = '<b>' +  df_ref[df_ref['thisrow']][name_coloumn_to_plot].values[0] + '</b>'

    pdrefs = []

    for reference in references:
        h = reference['natoms'] * reference['hform']
        pdrefs.append((reference['formula'], h))
    

    pd = PhaseDiagram(pdrefs, verbose=False)

    hull_energies = get_hull_energies(pd)

    if len(count) == 2:
        xcoord, energy, _, hull, simplices, xlabel, ylabel = pd.plot2d2()

        df_ref['xcoord'] = xcoord
        df_ref['energy'] = energy
        df_ref['hull'] = hull



        figs = []

        for i, j in simplices:
            fig_temp = px.line(x=xcoord[[i, j]], y=energy[[i, j]],color_discrete_sequence=colors)
            figs.append(fig_temp)

        fig_temp = px.scatter(df_ref,x='xcoord',y='energy',color='legend',hover_data=['link'], text='name',
                    labels={
                     "xcoord": xlabel_text,
                     "energy": '\u0394H [eV/atom]',
                     'legend': 'Crystal type',
                    },
                    color_discrete_sequence=colors)
        figs.append(fig_temp)

        delta = energy.ptp() / 30
        ymin = energy.min() - 2.5 * delta
        A, B = pd.symbols
       
        xlabel_text = f'{A}<sub>1-x</sub>{B}<sub>x</sub>'
        fig = go.Figure(data=sum([fig.data for fig in figs], ()), layout_yaxis_range=[ymin,0.1])
        fig.update_traces(textposition='middle right')
        fig.update_layout(
            xaxis_title=xlabel_text,
            yaxis_title='\u0394H [eV/atom]',
            )
        fig.update_layout(legend={"title":"Crystal type"},
                            width=700, height=500) 
        fig.update_layout(
            plot_bgcolor='white'
        )
        fig.update_xaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey',
        )
        fig.update_layout(yaxis=dict(zerolinecolor='lightgrey'),
                          xaxis=dict(zerolinecolor='lightgrey'))
        
        fig.write_html(fname, include_mathjax='cdn')
        fig.show()

    else:
        #x, y, _, hull, simplices = pd.plot2d3()
        x, y, e = pd.points[:, 1:].T
        df_ref['x'] = x
        df_ref['y'] = y
        df_ref['e'] = e

        hull = np.array(pd.hull)
        df_ref['hull'] = hull
        
        figs = []
        for i, j, k in pd.simplices:
            fig_temp = go.Figure(data=[go.Mesh3d(x=x[[i, j, k, i]],
                                                 y=y[[i, j, k, i]],
                                                 z=e[[i, j, k, i]], 
                                                 color=colors[2], opacity=0.5),])
            fig_temp.update_traces(hoverinfo='skip')
            figs.append(fig_temp)

        # Plot materials on the hull with formula and thisrow
        fig_temp = px.scatter_3d(df_ref[df_ref.hull | df_ref.thisrow], x='x', y='y', z='e', hover_data=['uid'], color='legend',
                                text='latexname', color_discrete_sequence=colors,
                                labels={"x": pd.symbols[1],
                                        "y": pd.symbols[2],
                                        "e": '\u0394H [eV/atom]',
                                        'legend': 'Crystal type',
                                    },)
        figs.append(fig_temp)

        # plot the rest of the materials without formula
        fig_temp = px.scatter_3d(df_ref[~df_ref.hull & ~df_ref.thisrow], x='x', y='y', z='e', hover_data=['uid'], color='legend',
                        color_discrete_sequence=colors,
                        labels={"x": pd.symbols[1],
                                "y": pd.symbols[2],
                                "e": '\u0394H [eV/atom]',
                                'legend': 'Crystal type',
                            },)
        figs.append(fig_temp)



        delta = e.ptp() / 30
        ymin = e.min() - 2.5 * delta
        fig = go.Figure(data=sum([fig.data for fig in figs], ()))

        fig.update_layout(
            legend={"title": "Crystal type"},
            width=800, 
            height=800,
            scene=dict(
                xaxis_title=pd.symbols[1],
                yaxis_title=pd.symbols[2],
                zaxis_title='\u0394H [eV/atom]',
                zaxis=dict(range=[ymin, 0.1])
            ),
                )
        
        fig.update_layout(
            plot_bgcolor='white'
        )
        fig.update_xaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey',
        )

        fig.update_layout(yaxis=dict(zerolinecolor='lightgrey'),
                          xaxis=dict(zerolinecolor='lightgrey'),
                          )


    fig.write_html(fname, include_mathjax='cdn')
    fig.show()



def convex_hull_tables(row: AtomsRow) -> List[Dict[str, Any]]:
    data = row.data['results-asr.convex_hull.json']

    references = data.get('references', [])
    tables = {}
    for reference in references:
        tables[reference['title']] = []

    for reference in sorted(references, reverse=True,
                            key=lambda x: x['hform']):
        name = reference['name']
        matlink = reference['link']
        if reference['uid'] != row.uid:
            name = f'<a href="{matlink}">{name}</a>'
        e = reference['hform']
        tables[reference['title']].append([name, '{:.2f} eV/atom'.format(e)])

    final_tables = []
    for title, rows in tables.items():
        final_tables.append({'type': 'table',
                             'header': [title, ''],
                             'rows': rows})
    return final_tables


if __name__ == '__main__':
    main.cli()
