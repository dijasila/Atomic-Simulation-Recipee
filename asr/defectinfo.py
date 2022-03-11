from ase.db import connect
from ase.io import read
from asr.core import command, ASRResult, prepare_result, read_json, option
from asr.defectlinks import get_charge_from_folder
from asr.defect_symmetry import DefectInfo
from asr.setup.defects import return_distances_cell
from pathlib import Path
import typing
from asr.database.browser import (table, describe_entry, href)
from asr.structureinfo import describe_crystaltype_entry


def webpanel(result, row, key_descriptions):
    spglib = href('SpgLib', 'https://spglib.github.io/spglib/')
    crystal_type = describe_crystaltype_entry(spglib)

    spg_list_link = href(
        'space group', 'https://en.wikipedia.org/wiki/List_of_space_groups')
    spacegroup = describe_entry(
        'Space group',
        f"The {spg_list_link} is determined with {spglib}.")
    pointgroup = describe_entry(
        'Point group',
        f"The point group is determined with {spglib}.")
    host_hof = describe_entry(
        'Heat of formation',
        result.key_descriptions['host_hof'])
    # XXX get correct XC name
    host_gap_pbe = describe_entry(
        'PBE band gap',
        'PBE band gap of the host crystal [eV].')
    host_gap_hse = describe_entry(
        'HSE band gap',
        'HSE band gap of the host crystal [eV].')
    R_nn = describe_entry(
        'Defect-defect distance',
        result.key_descriptions['R_nn'])

    # only show results for the concentration if charge neutrality results present
    try:
        defect_name = row.defect_name
        charge_state = row.charge_state
        q = charge_state.split()[-1].split(')')[0]
        conc_res = row.data['results-asr.charge_neutrality.json']
        for i, element in enumerate(conc_res['defect_concentrations']):
            if element['defect_name'] == defect_name:
                for altel in element['concentrations']:
                    if altel[1] == int(q):
                        concentration = altel[0]
        defect_name = (f'{defect_name.split("_")[0]}<sub>{defect_name.split("_")[1]}'
                       '</sub>')
        conc_row = describe_entry(
            'Eq. concentration',
            'Equilibrium concentration at self-consistent Fermi level.')
        show_conc = True
    except KeyError:
        show_conc = False

    uid = result.host_uid
    uidstring = describe_entry(
        'C2DB link',
        'Link to C2DB entry of the host material.')

    basictable = table(result, 'Pristine crystal', [])
    basictable['rows'].extend(
        [[crystal_type, result.host_crystal]])
    basictable['rows'].extend(
        [[spacegroup, result.host_spacegroup]])
    basictable['rows'].extend(
        [[pointgroup, result.host_pointgroup]])
    basictable['rows'].extend(
        [[host_hof, f'{result.host_hof:.2f} eV/atom']])
    basictable['rows'].extend(
        [[host_gap_pbe, f'{result.host_gap_pbe:.2f} eV']])
    if result.host_gap_hse is not None:
        basictable['rows'].extend(
            [[host_gap_hse, f'{result.host_gap_hse:.2f} eV']])
    defecttable = table(result, 'Defect properties', [])
    if result.R_nn is not None:
        defecttable['rows'].extend(
            [[R_nn, f'{result.R_nn:.2f} Å']])
    if show_conc:
        defecttable['rows'].extend(
            [[conc_row, f'{concentration:.1e} cm<sup>-2</sup>']])

    if uid:
        basictable['rows'].extend(
            [[uidstring,
              '<a href="https://cmrdb.fysik.dtu.dk/c2db/row/{uid}"'
              '>{uid}</a>'.format(uid=uid)]])

    panel = {'title': 'Summary',
             'columns': [[basictable, defecttable], []],
             'sort': -1}

    return [panel]


@prepare_result
class Result(ASRResult):
    """Container for asr.defectinfo results."""

    defect_name: str
    host_name: str
    charge_state: str
    host_pointgroup: str
    host_spacegroup: str
    host_crystal: str
    host_uid: str
    host_hof: float
    host_gap_pbe: float
    host_gap_hse: float
    R_nn: float

    key_descriptions: typing.Dict[str, str] = dict(
        defect_name='Name of the defect({type}_{position}).',
        host_name='Name of the host system.',
        charge_state='Charge state of the defect system.',
        host_pointgroup='Point group of the host crystal.',
        host_spacegroup='Space group of the host crystal.',
        host_crystal='Crystal type of the host crystal.',
        host_uid='UID of the primitive host crystal.',
        host_hof='Heat of formation for the host crystal [eV/atom].',
        host_gap_pbe='PBE bandgap of the host crystal [eV].',
        host_gap_hse='HSE bandgap of the host crystal [eV].',
        R_nn='Nearest neighbor distance of repeated defects [Å].')

    formats = {"ase_webpanel": webpanel}


@command(module='asr.defectinfo',
         resources='1:10m',
         returns=Result)
@option('--structurefile', help='Structure file for the evaluation of '
        'the nearest neighbor distance.', type=str)
@option('--pristine/--no-pristine', help='Flag to treat systems '
        'in the pristine folder of the set up tree structure '
        'from asr.setup.defects.', is_flag=True)
def main(structurefile: str = 'structure.json',
         pristine: bool = False) -> Result:
    """Extract defect, host name, and host crystalproperties.

    Extract defect name and host hame based on the folder structure
    created by asr.setup.defects.
    """
    atoms = read(structurefile)
    # extract path of the current directory
    p = Path('.')

    # collect all relevant paths for defect info extraction
    primitivepath, pristinepath = get_primitive_pristine_folderpaths(
        pristine, p)

    # obtain defectname and chargestate string
    if pristine:
        defectname = 'pristine'
        chargestate = ''
    else:
        defectname, chargestate = get_defectinfo_from_path(p)

    # calculation nearest neighbor distance for supercell structure
    R_nn = get_nearest_neighbor_distance(atoms)

    # extract atomic structure and name of the host crystal
    hostatoms = read(primitivepath / 'unrelaxed.json')
    hostname = hostatoms.get_chemical_formula()

    # collect all relevant results files
    resultsfile = Path(pristinepath / 'results-asr.structureinfo.json')
    if resultsfile.is_file():
        res = read_json(resultsfile)
        host_pointgroup = res['pointgroup']
        host_spacegroup = res['spacegroup']
        host_crystal = res['crystal_type']
    else:
        raise FileNotFoundError(
            'you need to run asr.structureinfo in the pristine folder!')

    resultsfile = Path(primitivepath / 'results-asr.database.material_fingerprint.json')
    if resultsfile.is_file():
        res = read_json(resultsfile)
        host_uid = res['uid']
    else:
        raise ValueError('no asr.database.material_fingerprint ran for '
                         'primitive structure! Make sure to run it before '
                         'running asr.defectinfo.')

    # extract host crystal properties from C2DB
    db = connect('/home/niflheim/fafb/db/c2db_july20.db')
    hof, pbe, hse = get_host_properties_from_db(db, host_uid)

    return Result.fromdata(
        defect_name=defectname,
        host_name=hostname,
        charge_state=chargestate,
        host_pointgroup=host_pointgroup,
        host_spacegroup=host_spacegroup,
        host_crystal=host_crystal,
        host_uid=host_uid,
        host_hof=hof,
        host_gap_pbe=pbe,
        host_gap_hse=hse,
        R_nn=R_nn)


def get_defectinfo_from_path(path):
    defectinfo = DefectInfo(defectpath=path)
    defectname = defectinfo.defectname
    charge = get_charge_from_folder(path)
    chargestate = f'(charge {charge})'

    return defectname, chargestate


def get_primitive_pristine_folderpaths(path, pristine):
    if pristine:
        primitivepath = Path('../')
        pristinepath = path
    else:
        primitivepath = Path('../../')
        pristinepath = list(path.glob('../../defects.pristine_sc*'))[-1]

    return primitivepath, pristinepath


def get_nearest_neighbor_distance(atoms):
    cell = atoms.get_cell()
    distances = return_distances_cell(cell)

    return min(distances)


def get_host_properties_from_db(db, uid):
    """Extract host properties from C2DB.

    The input database needs a uid, hform, and gap keyword.
    """
    for row in db.select(uid=uid):
        hof = row.hform
        gap_pbe = row.gap
        try:
            gap_hse = row.gap_hse
        except AttributeError:
            gap_hse = None

    return hof, gap_pbe, gap_hse


if __name__ == '__main__':
    main.cli()
