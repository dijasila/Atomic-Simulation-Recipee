from ase.formula import Formula
from asr.core import (command, ASRResult, prepare_result, chdir, read_json)
from asr.database.browser import WebPanel, table
from asr.database.material_fingerprint import main as material_fingerprint
from asr.defect_symmetry import DefectInfo
from pathlib import Path
import typing


def webpanel(result, row, key_description):
    baselink = 'http://sylg:5000/database.db/row/'
    charged_table = table(row, 'Charged systems', [])
    for element in result.chargedlinks:
        charged_table['rows'].extend(
            [[f'{element[1]}',
              f'<a href="{baselink}{element[0]}">link</a>']])

    neutral_table = table(row, 'Within the same material', [])
    for element in result.neutrallinks:
        neutral_table['rows'].extend(
            [[f'{element[1]}',
              f'<a href="{baselink}{element[0]}">link</a>']])
    for element in result.pristinelinks:
        neutral_table['rows'].extend(
            [[f'{element[1]}',
              f'<a href="{baselink}{element[0]}">link</a>']])

    panel = WebPanel('Related materials',
                     columns=[[charged_table], [neutral_table]],
                     sort=45)

    return [panel]


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


@command(module='asr.defectlinks',
         requires=['structure.json'],
         dependencies=['asr.relax'],
         resources='1:1h',
         returns=Result)
def main() -> Result:
    """Generate QPOD database links for the defect project."""
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
    if (Path(pristine / 'structure.json').is_file()):
        uid = get_uid_from_fingerprint(pristine)
        pristinelinks.append((uid, f"pristine material"))

    return Result.fromdata(
        chargedlinks=chargedlinks,
        neutrallinks=neutrallinks,
        pristinelinks=pristinelinks)


def get_list_of_links(path, charge):
    links = []
    structurefile = Path(path / 'structure.json')
    charge = get_charge_from_folder(path)
    if structurefile.is_file() and charge != 0:
        defectinfo = DefectInfo(defectpath=path)
        uid = get_uid_from_fingerprint(path)
        hostformula = get_hostformula_from_defectpath(path)
        defectstring = get_defectstring_from_defectinfo(defectinfo, charge)
        links.append((uid, f"{defectstring} in {hostformula:html}"))

    return links


def get_uid_from_fingerprint(path):
    with chdir(path):
        material_fingerprint()
        res = read_json('results-asr.database.material_fingerprint.json')
        uid = res['uid']

    return uid


def get_defectstring_from_defectinfo(defectinfo, charge):
    defecttype = defectinfo.defecttype
    defectkind = defectinfo.defectkind
    if defecttype == 'v':
        defecttype = 'V'
    defectstring = f"{defecttype}<sub>{defectkind}</sub> (charge {charge})"

    return defectstring


def get_hostformula_from_defectpath(path):
    fullpath = Path(path.absolute())
    token = fullpath.parent.name
    hostname = token.split('_')[0].split('defects.')[-1]

    return Formula(hostname)


def get_charge_from_folder(path):
    fullpath = Path(path.absolute())
    chargedstring = fullpath.name
    charge = int(chargedstring.split('charge_')[-1])

    return charge


if __name__ == '__main__':
    main.cli()
