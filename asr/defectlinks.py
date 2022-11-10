from ase.formula import Formula
from ase.io import read

from asr.core import ASRResult, prepare_result
from asr.database.browser import WebPanel, table
from asr.database.material_fingerprint import fingerprint
# Hack: For flexibility/error reasons, we need to circumvent the fact
# that material_fingerprint is a recipe.  Therefore we import _main.
from asr.defect_symmetry import DefectInfo
from pathlib import Path
import typing


def webpanel(result, row, key_description):
    baselink = 'https://cmrdb.fysik.dtu.dk/qpod/row/'

    # initialize table for charged and neutral systems
    charged_table = table(row, 'Other charge states', [])
    neutral_table = table(row, 'Other defects', [])
    # fill in values for the two tables from the result object
    charged_table = extend_table(charged_table, result, 'charged', baselink)
    neutral_table = extend_table(neutral_table, result, 'neutral', baselink)
    neutral_table = extend_table(neutral_table, result, 'pristine', baselink)

    # define webpanel
    panel = WebPanel('Other defects',
                     columns=[[charged_table], [neutral_table]],
                     sort=45)

    return [panel]


def extend_table(table, result, resulttype, baselink):
    if resulttype == 'pristine':
        tmpresult = result.pristinelinks
    elif resulttype == 'neutral':
        tmpresult = result.neutrallinks
    elif resulttype == 'charged':
        tmpresult = result.chargedlinks
    else:
        raise RuntimeError('did not find {resulttype} results!')

    for element in tmpresult:
        if element[1].startswith('V'):
            linkstring = element[1].replace('V', 'v', 1)
        else:
            linkstring = element[1]
        table['rows'].extend(
            [[f'{linkstring}',
              f'<a href="{baselink}{element[0]}">link</a>']])

    return table


@prepare_result
class Result(ASRResult):
    """Container for defectlinks results."""

    chargedlinks: typing.List
    neutrallinks: typing.List
    pristinelinks: typing.List

    key_descriptions = dict(
        chargedlinks='Links tuple for the charged states of the same defect.',
        neutrallinks='Links tuple for other defects within the same material.',
        pristinelinks='Link tuple for pristine material.')

    formats = {'ase_webpanel': webpanel}


# @command(module='asr.defectlinks',
#         requires=['structure.json'],
#         dependencies=['asr.relax'],
#         resources='1:1h',
#         returns=Result)
def main() -> Result:
    """Generate QPOD database links for the defect project.

    This recipe is dependent on the folder structure created by
    asr.setup.defects. Will generate links between different chargestates
    of the same defect, neutral chargestates for different defects of the
    same host material, and pristine structures. For your own project,
    make sure to change the 'basepath' variable in the webpanel function
    according to the location of your database.
    """
    # extract path of current directory
    p = Path('.')

    # First, get charged links for the same defect system
    chargedlinks = []
    chargedlist = list(p.glob('./../charge_*'))
    for charged in chargedlist:
        chargedlinks = get_list_of_links(charged)

    # Second, get the neutral links of systems in the same host
    neutrallinks = []
    neutrallist = list(p.glob('./../../*/charge_0'))
    for neutral in neutrallist:
        neutrallinks = get_list_of_links(neutral)

    # Third, the pristine material
    pristinelinks = []
    pristine = list(p.glob('./../../defects.pristine_sc*'))[0]
    pristinefile = pristine / 'structure.json'
    if pristinefile.is_file():
        atoms = read(pristinefile)
        uid = fingerprint(atoms)['uid']
        pristinelinks.append((uid, 'pristine material'))

    return Result.fromdata(
        chargedlinks=chargedlinks,
        neutrallinks=neutrallinks,
        pristinelinks=pristinelinks)


def get_list_of_links(path):
    links = []
    structurefile = path / 'structure.json'
    atoms = read(structurefile)
    charge = get_charge_from_folder(path)
    if structurefile.is_file() and charge != 0:
        defectinfo = DefectInfo(defectpath=path)
        uid = fingerprint(atoms)['uid']
        hostformula = get_hostformula_from_defectpath(path)
        defectstring = get_defectstring_from_defectinfo(defectinfo, charge)
        links.append((uid, f"{defectstring} in {hostformula:html}"))

    return links


def get_defectstring_from_defectinfo(defectinfo, charge):
    defectstring = ''
    for name in defectinfo.names:
        def_type, def_kind = defectinfo.get_defect_type_and_kind_from_defectname(
            name)
        defectstring += f"{def_type}<sub>{def_kind}</sub>"
    defectstring += f" (charge {charge})"

    return defectstring


def get_hostformula_from_defectpath(path):
    fullpath = path.absolute()
    token = fullpath.parent.name
    hostname = token.split('_')[0].split('defects.')[-1]

    return Formula(hostname)


def get_charge_from_folder(path):
    fullpath = path.absolute()
    chargedstring = fullpath.name
    charge = int(chargedstring.split('charge_')[-1])

    return charge


if __name__ == '__main__':
    main.cli()
