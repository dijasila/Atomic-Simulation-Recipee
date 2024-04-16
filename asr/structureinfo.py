"""Structural information."""
import numpy as np

from asr.core import ASRResult, command, prepare_result
from asr.database.browser import bold, br, code, describe_entry, div, dl, href


def get_reduced_formula(formula, stoichiometry=False):
    """Get reduced formula from formula.

    Returns the reduced formula corresponding to a chemical formula,
    in the same order as the original formula
    E.g. Cu2S4 -> CuS2

    Parameters
    ----------
    formula : str
    stoichiometry : bool
        If True, return the stoichiometry ignoring the
        elements appearing in the formula, so for example "AB2" rather than
        "MoS2"

    Returns
    -------
        A string containing the reduced formula.
    """
    import re
    import string
    from functools import reduce
    from math import gcd
    split = re.findall('[A-Z][^A-Z]*', formula)
    matches = [re.match('([^0-9]*)([0-9]+)', x)
               for x in split]
    numbers = [int(x.group(2)) if x else 1 for x in matches]
    symbols = [matches[i].group(1) if matches[i] else split[i]
               for i in range(len(matches))]
    divisor = reduce(gcd, numbers)
    result = ''
    numbers = [x // divisor for x in numbers]
    numbers = [str(x) if x != 1 else '' for x in numbers]
    if stoichiometry:
        numbers = sorted(numbers)
        symbols = string.ascii_uppercase
    for symbol, number in zip(symbols, numbers):
        result += symbol + number
    return result


def get_spg_href(url):
    return href('SpgLib', url)


def describe_pointgroup_entry(spglib):
    pointgroup = describe_entry(
        'Point group',
        f"Point group determined with {spglib}."
    )

    return pointgroup


def describe_crystaltype_entry(spglib):
    crystal_type = describe_entry(
        'Crystal type',
        "The crystal type is defined as "
        + br
        + div(bold('-'.join([code('stoi'),
                             code('spg no.'),
                             code('occ. wyck. pos.')])), 'well well-sm text-center')
        + 'where'
        + dl(
            [
                [code('stoi'), 'Stoichiometry.'],
                [code('spg no.'), f'The space group calculated with {spglib}.'],
                [code('occ. wyck. pos.'),
                 'Alphabetically sorted list of occupied '
                 f'wyckoff positions determined with {spglib}.'],
            ]
        )
    )

    return crystal_type


# XXX This string is used in CMR also in search.html also.
# It should probably be imported from here instead.
dynstab_description = """\
Dynamically stable materials are stable against small perturbations of
their structure (atom positions and unit cell shape). The structure
thus represents a local minimum on the {PES}.

DS materials are characterised by having only real, non-negative {phonon}
frequencies and positive definite {stiffnesstensor}.
""".format(
    PES=href('potential energy surface',
             'https://en.wikipedia.org/wiki/Potential_energy_surface'),
    phonon=href('phonon', 'https://en.wikipedia.org/wiki/Phonon'),
    stiffnesstensor=href(
        'stiffness tensor',
        'https://en.wikiversity.org/wiki/Elasticity/Constitutive_relations'))


def webpanel(result, row, key_descriptions):
    from asr.database.browser import describe_entry, href, table

    spglib = get_spg_href('https://spglib.github.io/spglib/')
    crystal_type = describe_crystaltype_entry(spglib)

    spg_list_link = href(
        'Space group', 'https://en.wikipedia.org/wiki/List_of_space_groups'
    )

    layergroup_link = href(
        'Layer group', 'https://en.wikipedia.org/wiki/Layer_group')

    spacegroup = describe_entry(
        'spacegroup',
        f"{spg_list_link} determined with {spglib}."
        f"The {spg_list_link} determined with {spglib} by stacking the "
        f"monolayer in A-A configuration."
    )

    spgnum = describe_entry(
        'spgnum',
        f"{spg_list_link} number determined with {spglib}."
        f"{spg_list_link} number determined with {spglib} by stacking the "
        f"monolayer in A-A configuration."
    )

    layergroup = describe_entry(
        'layergroup',
        f'{layergroup_link} determined with {spglib}')
    lgnum = describe_entry(
        'lgnum',
        f'{layergroup_link} number determined with {spglib}')

    pointgroup = describe_pointgroup_entry(spglib)

    icsd_link = href('Inorganic Crystal Structure Database (ICSD)',
                     'https://icsd.products.fiz-karlsruhe.de/')

    icsd_id = describe_entry(
        'icsd_id',
        f"ID of a closely related material in the {icsd_link}."
    )

    cod_link = href(
        'Crystallography Open Database (COD)',
        'http://crystallography.net/cod/browse.html'
    )

    cod_id = describe_entry(
        'cod_id',
        f"ID of a closely related material in the {cod_link}."
    )

    # Here we are hacking the "label" out of a row without knowing
    # whether there is a label, or that the "label" recipe exists.

    tablerows = [
        crystal_type, layergroup, lgnum, spacegroup, spgnum, pointgroup,
        icsd_id, cod_id]

    # The table() function is EXTREMELY illogical.
    # I can't get it to work when appending another row
    # to the tablerows list.  Therefore we append rows afterwards.  WTF.
    basictable = table(row, 'Structure info', tablerows, key_descriptions, 2)
    rows = basictable['rows']

    labelresult = row.data.get('results-asr.c2db.labels.json')
    if labelresult is not None:
        tablerow = labelresult.as_formatted_tablerow()
        rows.append(tablerow)

    codid = row.get('cod_id')
    if codid:
        # Monkey patch to make a link
        for tmprow in rows:
            href = ('<a href="http://www.crystallography.net/cod/'
                    + '{id}.html">{id}</a>'.format(id=codid))
            if 'cod_id' in tmprow[0]:
                tmprow[1] = href

    doi = row.get('doi')
    doistring = describe_entry(
        'Reported DOI',
        'DOI of article reporting the synthesis of the material.'
    )
    if doi:
        rows.append([
            doistring,
            '<a href="https://doi.org/{doi}" target="_blank">{doi}'
            '</a>'.format(doi=doi)
        ])

    # There should be a central place defining "summary" panel to take
    # care of stuff that comes not from an individual "recipe" but
    # from multiple ones.  Such as stability, or listing multiple band gaps
    # next to each other, etc.
    #
    # For now we stick this in structureinfo but that is god-awful.
    phonon_stability = row.get('dynamic_stability_phonons')
    stiffness_stability = row.get('dynamic_stability_stiffness')

    from asr.convex_hull import ehull_table_rows
    ehull_table_rows = ehull_table_rows(row, key_descriptions)['rows']

    if phonon_stability is not None and stiffness_stability is not None:
        # XXX This will easily go wrong if 'high'/'low' strings are changed.
        dynamically_stable = (
            phonon_stability == 'high' and stiffness_stability == 'high')

        yesno = ['No', 'Yes'][dynamically_stable]

        dynstab_row = [describe_entry('Dynamically stable', dynstab_description), yesno]
        dynstab_rows = [dynstab_row]
    else:
        dynstab_rows = []

    panel = {'title': 'Summary',
             'columns': [[basictable,
                          {'type': 'table', 'header': ['Stability', ''],
                           'rows': [*ehull_table_rows, *dynstab_rows]}],
                         [{'type': 'atoms'}, {'type': 'cell'}]],
             'sort': -1}

    return [panel]


tests = [{'description': 'Test SI.',
          'cli': ['asr run "setup.materials -s Si2"',
                  'ase convert materials.json structure.json',
                  'asr run "setup.params asr.gs@calculate:ecut 300 '
                  'asr.gs@calculate:kptdensity 2"',
                  'asr run structureinfo',
                  'asr run database.fromtree',
                  'asr run "database.browser --only-figures"']}]


@prepare_result
class Result(ASRResult):

    cell_area: float
    has_inversion_symmetry: bool
    stoichiometry: str
    spacegroup: str
    spgnum: int
    layergroup: str
    lgnum: int
    pointgroup: str
    crystal_type: str
    spglib_dataset: dict
    formula: str

    key_descriptions = {
        "cell_area": "Area of unit-cell [`Å²`]",
        "has_inversion_symmetry": "Material has inversion symmetry",
        "stoichiometry": "Stoichiometry",
        "spacegroup": "Space group (AA stacking)",
        "spgnum": "Space group number (AA stacking)",
        "layergroup": "Layer group",
        "lgnum": "Layer group number",
        "pointgroup": "Point group",
        "crystal_type": "Crystal type",
        "spglib_dataset": "SPGLib symmetry dataset.",
        "formula": "Chemical formula."
    }

    formats = {"ase_webpanel": webpanel}


def get_layer_group(atoms, symprec):
    try:
        from spglib.spglib import get_symmetry_layerdataset
    except ImportError:
        return None, None

    assert atoms.pbc.sum() == 2
    aperiodic_dir = np.where(~atoms.pbc)[0][0]
    # Prepare for spglib v3 API change to always have the aperiodic_dir == 2
    # See: https://github.com/spglib/spglib/issues/314.
    if aperiodic_dir != 2:
        perm = np.array([0, 1, 2])
        # Swap axes such that aperiodic is always 2
        perm[2], perm[aperiodic_dir] = perm[aperiodic_dir], perm[2]
        atoms = atoms.copy()
        atoms.set_pbc(atoms.get_pbc()[perm])
        # The atoms are stored in cartesian coordinates, therefore, we are
        # free to permute the cell vectors and system remains invariant.
        atoms.set_cell(atoms.get_cell()[perm], scale_atoms=False)
        aperiodic_dir = 2

    assert aperiodic_dir == 2

    lg_dct = get_symmetry_layerdataset(
        (atoms.get_cell(),
         atoms.get_scaled_positions(),
         atoms.get_atomic_numbers()),
        symprec=symprec,
        aperiodic_dir=aperiodic_dir)

    layergroup = lg_dct['number']
    layergroupname = lg_dct['international']

    return layergroupname, layergroup


@command('asr.structureinfo',
         tests=tests,
         requires=['structure.json'],
         returns=Result)
def main() -> Result:
    """Get structural information of atomic structure.

    This recipe produces information such as the space group and magnetic
    state properties that requires only an atomic structure. This recipes read
    the atomic structure in `structure.json`.
    """
    import numpy as np
    from ase.io import read

    from asr.utils.symmetry import c2db_symmetry_angle, c2db_symmetry_eps

    atoms = read('structure.json')
    info = {}

    formula = atoms.get_chemical_formula(mode='metal')
    stoichimetry = get_reduced_formula(formula, stoichiometry=True)
    info['formula'] = formula
    info['stoichiometry'] = stoichimetry

    # Get crystal symmetries
    from asr.utils.symmetry import atoms2symmetry

    symmetry = atoms2symmetry(atoms,
                              tolerance=c2db_symmetry_eps,
                              angle_tolerance=c2db_symmetry_angle)
    info['has_inversion_symmetry'] = symmetry.has_inversion
    dataset = symmetry.dataset
    info['spglib_dataset'] = dataset

    # Get crystal type
    stoi = atoms.symbols.formula.stoichiometry()[0]
    sg = dataset['international']
    number = dataset['number']
    pg = dataset['pointgroup']
    w = ''.join(sorted(set(dataset['wyckoffs'])))
    crystal_type = f'{stoi}-{number}-{w}'

    info['layergroup'] = None
    info['lgnum'] = None
    ndims = sum(atoms.pbc)
    if ndims == 2:
        info['layergroup'], info['lgnum'] = get_layer_group(
            atoms,
            symprec=c2db_symmetry_eps)

    info['crystal_type'] = crystal_type
    info['spacegroup'] = sg
    info['spgnum'] = number
    from ase.db.core import convert_str_to_int_float_bool_or_str, str_represents
    if str_represents(pg):
        info['pointgroup'] = convert_str_to_int_float_bool_or_str(pg)
    else:
        info['pointgroup'] = pg

    if (atoms.pbc == [True, True, False]).all():
        info['cell_area'] = abs(np.linalg.det(atoms.cell[:2, :2]))
    else:
        info['cell_area'] = None

    return Result(data=info)


if __name__ == '__main__':
    main.cli()
