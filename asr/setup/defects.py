"""Generate defective atomic structures."""
from typing import List
from pathlib import Path
from asr.core import command, option, ASRResult, atomsopt
import click
import os

from ase import Atoms


@command('asr.setup.defects')
@atomsopt
@option('-q', '--chargestates', type=int,
        help='Charge states included (-q, ..., +q).')
@option('--supercell', nargs=3, type=click.Tuple([int, int, int]),
        help='List of repetitions in lat. vector directions [N_x, N_y, N_z]')
@option('--maxsize', type=float,
        help='Maximum supercell size in Å.')
@option('--intrinsic', type=bool,
        help='Specify whether you want to incorporate anti-site defects.')
@option('--vacancies', type=bool,
        help='Specify whether you want to incorporate vacancies.')
@option('--uniform_vacuum', type=bool,
        help='Pass some float value to choose vacuum for 2D case manually, '
        ' it will be chosen automatically otherwise.')
@option('--nopbc', type=bool,
        help='Keep the periodic boundary conditions as they are. If this '
        'option is not used, pbc will be enforced for correct defect '
        'calculations')
@option('--halfinteger', type=bool,
        help='Sets up half integer folders within one full integer folder. '
        'It has to be launched within the specific charge folders and needs '
        'both a structure.json file as well as a params.json in order to '
        'work properly')
@option('--general_algorithm', type=float,
        help='Sets up general supercells that break the initial symmetry '
        'of the bravais lattice, as well as choosing the most uniform '
        'configuration with least atoms in the supercell.')
def main(
        atoms: Atoms,
        chargestates: int = 3,
        supercell: List[int] = [0, 0, 0],
        maxsize: float = 8,
        intrinsic: bool = True,
        vacancies: bool = True,
        uniform_vacuum: bool = False,
        nopbc: bool = True,
        halfinteger: bool = False,
        general_algorithm: float = None,
) -> ASRResult:
    """Set up defect structures for a given host.

    Recipe setting up all possible defects within a reasonable supercell as well as the
    respective pristine system for a given input structure. Defects include: vacancies,
    intrinsic substitutional defects. For a given primitive input structure this recipe
    will create a directory tree in the following way (for the example of MoS2):

    - There has to be a 'unrelaxed.json' file with the primitive structure
      of the desired system in the folder you run setup.defects. The tree
      structure will then look like this:

    .                                                                                  '
    ├── general_parameters.json                                                        '
    ├── MoS2_331.v_S                                                                   '
    │   ├── charge_0                                                                   '
    │   │   ├── params.json                                                            '
    │   │   └── unrelaxed.json                                                         '
    │   ├── charge_1                                                                   '
    │   │   ├── params.json                                                            '
    │   │   └── unrelaxed.json -> ./../charge_0/structure.json                         '
    │   .                                                                              '
    │                                                                                  '
    ├── MoS2_231.Mo_S                                                                  '
    │   ├── charge_0                                                                   '
    .   .                                                                              '
    .                                                                                  '
    ├── pristine_sc                                                                    '
    │   ├── params.json                                                                '
    │   └── structure.json                                                             '
    ├── results_setup.defects.json                                                     '
    └── unrelaxed.json                                                                 '

    - Here, the notation for the defects is the following:
      'formula_supercellsize.defect_sustitutionposition' where 'v' denotes a vacancy
    - When the general algorithm is used to set up symmetry broken supercells, the
      foldernames will contain '000' instead of the supersize.
    - In the resulting folders you can find the unrelaxed structures, as well as a
      'params.json' file which contains specific parameters as well as the charge states
      of the different defect structures.
    """
    import numpy as np

    # only run SJ setup if halfinteger is True
    if halfinteger:
        setup_halfinteger()
    # otherwise, run complete setup of defect structures
    elif not halfinteger:
        # first, read input atomic structure and store it in ase's atoms object
        structure = atoms
        print('INFO: starting recipe for setting up defect systems of '
              '{} host system.'.format(structure.symbols))
        # check dimensionality of initial parent structure
        nd = int(np.sum(structure.get_pbc()))
        if nd == 3:
            is2d = False
        elif nd == 2:
            is2d = True
        elif nd == 1:
            raise NotImplementedError(
                'Setup defects not implemented for 1D structures')
        # set up the different defect systems and store their properties
        # in a dictionary
        structure_dict = setup_defects(structure=structure, intrinsic=intrinsic,
                                       charge_states=chargestates,
                                       vacancies=vacancies, sc=supercell,
                                       max_lattice=maxsize, is_2D=is2d,
                                       vacuum=uniform_vacuum, nopbc=nopbc,
                                       general_algorithm=general_algorithm)

        # based on this dictionary, create a folder structure for all defects
        # and respective charge states
        create_folder_structure(structure, structure_dict, chargestates,
                                intrinsic=intrinsic, vacancies=vacancies,
                                sc=supercell, max_lattice=maxsize, is_2D=is2d)

    return ASRResult()


def setup_supercell(structure, max_lattice, is_2D):
    """Set up the supercell of a given structure.

    Parameters
    ----------
    structure
        input structure (primitive cell)
    max_lattice : float
        maximum supercell lattice vector length in Å
    is_2D : bool
        choose 2D or 3D supercell (is_2D=False)

    Returns
    -------
    structure_sc
        supercell structure
    """
    for x in range(1, 50):
        struc_temp = structure.repeat((x, 1, 1))
        diff = struc_temp.get_distance(0, -1)
        if diff > max_lattice:
            x_size = x - 1
            break
    for y in range(1, 50):
        struc_temp = structure.repeat((1, y, 1))
        diff = struc_temp.get_distance(0, -1)
        if diff > max_lattice:
            y_size = y - 1
            break
    if not is_2D:
        for z in range(1, 50):
            struc_temp = structure.repeat((1, 1, z))
            diff = struc_temp.get_distance(0, -1)
            if diff > max_lattice:
                z_size = z - 1
                break
    else:
        z_size = 1

    x_size = max(1, x_size)
    y_size = max(1, y_size)
    z_size = max(1, z_size)
    structure_sc = structure.repeat((x_size, y_size, z_size))

    print('INFO: setting up supercell: ({0}, {1}, {2})'.format(
          x_size, y_size, z_size))

    return structure_sc, x_size, y_size, z_size


def apply_vacuum(structure_sc, vacuum, is_2D, nopbc):
    """
    Apply vacuum to 2D structures.

    Either sets the vacuum automatically for the 2D case (in such a way that
    L_z ~ L_xy, sets it accordingly to the given input vacuum value, or just
    passes in case one is dealing with a 3D structure.

    :param structure_sc: supercell structure without defects incorporated
    :param vacuum: either None (automatic adjustment of the vacuum size) or
                   some float value for manual adjustment of the vacuum for
                   2D structure
    :param is_2D: dimensionality of the structure

    :return supercell_final: supercell structure with suitable vacuum size
                             applied
    """
    import numpy as np
    if is_2D:
        cell = structure_sc.get_cell()
        oldvac = cell[2][2]
        pos = structure_sc.get_positions()
        a1 = np.sqrt(cell[0][0]**2 + cell[0][1]**2)
        a2 = np.sqrt(cell[1][0]**2 + cell[1][1]**2)
        a = (a1 + a2) / 2.
        if vacuum is True:
            vacuum = a
            print('INFO: apply vacuum size to the supercell of the 2D structure '
                  'with {} Å.'.format(vacuum))
        elif vacuum is False:
            vacuum = oldvac
            print('INFO: keep vacuum according to the initial 2D structure '
                  'with {} Å.'.format(vacuum))
        cell[2][2] = vacuum
        pos[:, 2] = pos[:, 2] - oldvac / 2. + vacuum / 2.
        structure_sc.set_cell(cell)
        structure_sc.set_positions(pos)
    elif not is_2D:
        print('INFO: no vacuum to be applied since this is a 3D structure.')
    if nopbc is False:
        structure_sc.set_pbc([True, True, True])
        print('INFO: overwrite pbc and apply them in all three directions for'
              ' subsequent defect calculations.')

    return structure_sc


def setup_defects(structure, intrinsic, charge_states, vacancies, sc,
                  max_lattice, is_2D, vacuum, nopbc, general_algorithm):
    """
    Set up defects for a particular input structure.

    Sets up all possible defects (i.e. vacancies, intrinsic anti-sites,
    extrinsic point defects('extrinsic=True')) for a given structure.

    Parameters
    ----------
    structure
        input structure (primitive cell)
    intrinsic : bool
        incorporate intrinsic point defects
    vacancies : bool
        incorporate vacancies

    Returns
    -------
    structure_dict : dict
        dictionary of all possible defect configurations
        of the given structure with different charge
        states. The dictionary is built up in the
        following way: see folder structure in 'main()'.
    """
    import spglib

    # set up artificial array in order to check for equivalent positions later
    cell = (structure.cell.array, structure.get_scaled_positions(),
            structure.numbers)

    # set up a dictionary
    structure_dict = {}
    formula = structure.symbols

    # first, find the desired supercell
    if sc[0] == 0 and sc[1] == 0 and sc[2] == 0 and general_algorithm is False:
        pristine, N_x, N_y, N_z = setup_supercell(
            structure, max_lattice, is_2D)
        pristine = apply_vacuum(pristine, vacuum, is_2D, nopbc)
    elif general_algorithm is not None:
        pristine = create_general_supercell(structure, size=float(general_algorithm))
        pristine = apply_vacuum(pristine, vacuum, is_2D, nopbc)
        N_x = 0
        N_y = 0
        N_z = 0
    else:
        N_x = sc[0]
        N_y = sc[1]
        N_z = sc[2]
        print('INFO: setting up supercell: ({0}, {1}, {2})'.format(
              N_x, N_y, N_z))
        pristine = structure.repeat((N_x, N_y, N_z))
        pristine = apply_vacuum(pristine, vacuum, is_2D, nopbc)
    parameters = {}
    string = 'defects.pristine_sc.{}{}{}'.format(N_x, N_y, N_z)
    calculator_relax = {
        'name': 'gpaw',
        'mode': {
            'name': 'pw',
            'ecut': 800,
            'dedecut': 'estimate'},
        'xc': 'PBE',
        'kpts': {
            'density': 6.0,
            'gamma': True},
        'symmetry': {
            'symmorphic': False},
        'convergence': {
            'forces': 1e-4},
        'txt': 'relax.txt',
        'occupations': {
            'name': 'fermi-dirac',
            'width': 0.02},
        'spinpol': True}
    calculator_gs = {'name': 'gpaw',
                     'mode': {'name': 'pw', 'ecut': 800},
                     'xc': 'PBE',
                     'kpts': {'density': 12.0, 'gamma': True},
                     'occupations': {'name': 'fermi-dirac',
                                     'width': 0.02},
                     'convergence': {'bands': 'CBM+3.0'},
                     'nbands': '200%',
                     'txt': 'gs.txt',
                     'spinpol': True}
    parameters['asr.gs@calculate'] = {
        'calculator': calculator_gs}
    parameters['asr.relax'] = {'calculator': calculator_relax}
    structure_dict[string] = {'structure': pristine, 'parameters': parameters}

    # incorporate the possible vacancies
    dataset = spglib.get_symmetry_dataset(cell)
    eq_pos = dataset.get('equivalent_atoms')

    finished_list = []
    if vacancies:
        temp_dict = {}
        for i in range(len(structure)):
            if not eq_pos[i] in finished_list:
                vacancy = pristine.copy()
                sitename = vacancy.get_chemical_symbols()[i]
                vacancy.pop(i)
                vacancy.rattle()
                string = 'defects.{0}_{1}{2}{3}.v_{4}'.format(
                         formula, N_x, N_y, N_z, sitename)
                charge_dict = {}
                for q in range((-1) * charge_states, charge_states + 1):
                    parameters = {}
                    calculator_relax = {
                        'name': 'gpaw',
                        'mode': {
                            'name': 'pw',
                            'ecut': 800,
                            'dedecut': 'estimate'},
                        'xc': 'PBE',
                        'kpts': {
                            'density': 6.0,
                            'gamma': True},
                        'symmetry': {
                            'symmorphic': False},
                        'convergence': {
                            'forces': 1e-4},
                        'txt': 'relax.txt',
                        'occupations': {
                            'name': 'fermi-dirac',
                            'width': 0.02},
                        'spinpol': True}
                    calculator_gs = {'name': 'gpaw',
                                     'mode': {'name': 'pw', 'ecut': 800},
                                     'xc': 'PBE',
                                     'kpts': {'density': 12.0, 'gamma': True},
                                     'occupations': {'name': 'fermi-dirac',
                                                     'width': 0.02},
                                     'convergence': {'bands': 'CBM+3.0'},
                                     'nbands': '200%',
                                     'txt': 'gs.txt',
                                     'spinpol': True}
                    parameters['asr.gs@calculate'] = {
                        'calculator': calculator_gs}
                    parameters['asr.gs@calculate']['calculator']['charge'] = q
                    parameters['asr.relax'] = {'calculator': calculator_relax}
                    parameters['asr.relax']['calculator']['charge'] = q
                    charge_string = 'charge_{}'.format(q)
                    charge_dict[charge_string] = {
                        'structure': vacancy, 'parameters': parameters}
                temp_dict[string] = charge_dict
            finished_list.append(eq_pos[i])

    # incorporate anti-site defects
    finished_list = []
    if intrinsic:
        defect_list = []
        for i in range(len(structure)):
            symbol = structure[i].symbol
            if symbol not in defect_list:
                defect_list.append(symbol)
        for i in range(len(structure)):
            if not eq_pos[i] in finished_list:
                for element in defect_list:
                    if not structure[i].symbol == element:
                        defect = pristine.copy()
                        sitename = defect.get_chemical_symbols()[i]
                        defect[i].symbol = element
                        defect.rattle()
                        string = 'defects.{0}_{1}{2}{3}.{4}_{5}'.format(
                                 formula, N_x, N_y, N_z, element,
                                 sitename)
                        charge_dict = {}
                        for q in range(
                                (-1) * charge_states,
                                charge_states + 1):
                            parameters = {}
                            calculator_relax = {
                                'name': 'gpaw',
                                'mode': {
                                    'name': 'pw',
                                    'ecut': 800,
                                    'dedecut': 'estimate'},
                                'xc': 'PBE',
                                'kpts': {
                                    'density': 6.0,
                                    'gamma': True},
                                'symmetry': {
                                    'symmorphic': False},
                                'convergence': {
                                    'forces': 1e-4},
                                'txt': 'relax.txt',
                                'occupations': {
                                    'name': 'fermi-dirac',
                                    'width': 0.02},
                                'spinpol': True}
                            calculator_gs = {
                                'name': 'gpaw',
                                'mode': {
                                    'name': 'pw',
                                    'ecut': 800},
                                'xc': 'PBE',
                                'kpts': {
                                    'density': 12.0,
                                    'gamma': True},
                                'occupations': {
                                    'name': 'fermi-dirac',
                                    'width': 0.02},
                                'convergence': {
                                    'bands': 'CBM+3.0'},
                                'nbands': '200%',
                                'txt': 'gs.txt',
                                'spinpol': True}
                            parameters['asr.gs@calculate'] = {
                                'calculator': calculator_gs}
                            parameters['asr.gs@calculate']['calculator'
                                                           ]['charge'] = q
                            parameters['asr.relax'] = {
                                'calculator': calculator_relax}
                            parameters['asr.relax']['calculator']['charge'] = q
                            charge_string = 'charge_{}'.format(q)
                            charge_dict[charge_string] = {
                                'structure': defect, 'parameters': parameters}
                        temp_dict[string] = charge_dict
                finished_list.append(eq_pos[i])

    # put together structure dict
    structure_dict['defects'] = temp_dict

    print('INFO: rattled atoms to make sure defect systems do not get stuck at'
          ' a saddle point.')

    print('INFO: setting up {0} different defect supercell systems in '
          'charge states -{1}, ..., +{1}, as well as the pristine supercell '
          'system.'.format(len(structure_dict['defects']), charge_states))

    return structure_dict


def create_folder_structure(structure, structure_dict, chargestates,
                            intrinsic, vacancies, sc, max_lattice, is_2D):
    """Create folder for all configurations.

    Creates a folder for every configuration of the defect supercell in
    the following way:

    - see example directory tree in 'main()'
    - these each contain two files: 'unrelaxed.json' (the defect
      supercell structure), 'params.json' (the non-general parameters
      of each system)
    - the content of those folders can then be used to do further
      processing (e.g. relax the defect structure)
    """
    from ase.io import write
    from asr.core import write_json

    # create a json file for general parameters that are equivalent for all
    # the different defect systems
    if sc == [0, 0, 0]:
        pristine, N_x, N_y, N_z = setup_supercell(
            structure, max_lattice, is_2D)
    else:
        N_x = sc[0]
        N_y = sc[1]
        N_z = sc[2]
    gen_params = {}
    gen_params['chargestates'] = chargestates
    gen_params['is_2D'] = is_2D
    gen_params['supercell'] = [N_x, N_y, N_z]
    gen_params['intrinsic'] = intrinsic
    gen_params['vacancies'] = vacancies
    write_json('general_parameters.json', gen_params)

    # then, create a seperate folder for each possible defect
    # configuration of this parent folder, as well as the pristine
    # supercell system
    for element in structure_dict:
        folder_name = element
        try:
            if not folder_name == 'defects':
                Path(folder_name).mkdir()
        except FileExistsError:
            print('WARNING: folder ("{0}") already exists in this '
                  'directory. Skip creating it.'.format(folder_name))
        if structure_dict[element].get('structure') is not None:
            struc = structure_dict[element].get('structure')
            params = structure_dict[element].get('parameters')
            try:
                write(folder_name + '/structure.json', struc)
                write_json(folder_name + '/params.json', params)
            except FileExistsError:
                print('WARNING: files already exist inside this folder.')
        else:
            sub_dict = structure_dict[element]
            j = 0
            for sub_element in sub_dict:
                defect_name = [key for key in sub_dict.keys()]
                defect_folder_name = defect_name[j]
                j = j + 1
                try:
                    Path(defect_folder_name).mkdir()
                except FileExistsError:
                    print(
                        'WARNING: folder ("{0}") already exists in this '
                        'directory. Skip creating '
                        'it.'.format(defect_folder_name))
                for i in range((-1) * chargestates, chargestates + 1):
                    charge_name = 'charge_{}'.format(i)
                    charge_folder_name = defect_folder_name + '/' + charge_name
                    try:
                        Path(charge_folder_name).mkdir()
                    except FileExistsError:
                        print(
                            'WARNING: folder ("{0}") already exists in this '
                            'directory. Skip creating '
                            'it.'.format(charge_folder_name))
                    struc = sub_dict[sub_element].get(
                        charge_name).get('structure')
                    params = sub_dict[sub_element].get(
                        charge_name).get('parameters')
                    write_json(charge_folder_name + '/params.json', params)
                    if i == 0:
                        write(charge_folder_name + '/unrelaxed.json', struc)
                    elif i < 0:
                        os.system(
                            'ln -s ../charge_{}/structure.json {}'
                            '/unrelaxed.json'.format(i + 1, charge_folder_name))
                    elif i > 0:
                        os.system(
                            'ln -s ../charge_{}/structure.json {}'
                            '/unrelaxed.json'.format(i - 1, charge_folder_name))

    return None


def setup_halfinteger():
    """
    Set up folders for SJ calculations.

    Sets up halfinteger folder which copies params.json and changes the q
    keyword as well as copying the relaxed structure into those folders.
    """
    from asr.core import read_json, write_json
    import shutil

    charge = int(str(Path('.').absolute()).split('/')[-1].split('_')[-1])

    folderpath = Path('.')
    foldername = str(folderpath)
    print('INFO: set up half integer folders and parameter sets for '
          'a subsequent Slater-Janach calculation')
    if charge <= 0:
        print('INFO: charge = {} -> set up negative half integer folder.'.format(
            charge))
        Path('sj_-0.5').mkdir()
        params = read_json('params.json')
        params_m05 = params.copy()
        params_m05['asr.gs@calculate']['calculator']['charge'] = charge - 0.5
        params_m05['asr.relax']['calculator']['charge'] = charge - 0.5
        write_json('sj_-0.5/params.json', params_m05)
        print('INFO: changed parameters m: {}'.format(params_m05))
        shutil.copyfile(foldername + '/structure.json',
                        foldername + '/sj_-0.5/structure.json')
    if charge >= 0:
        print('INFO: charge = {} -> set up positive half integer folder.'.format(
            charge))
        Path('sj_+0.5').mkdir()
        params = read_json('params.json')
        params_p05 = params.copy()
        charge = params.get('asr.gs@calculate').get('calculator').get('charge')
        print('INFO: initial charge {}'.format(charge))
        params_p05['asr.gs@calculate']['calculator']['charge'] = charge + 0.5
        params_p05['asr.relax']['calculator']['charge'] = charge + 0.5
        write_json('sj_+0.5/params.json', params_p05)
        print('INFO: changed parameters p: {}'.format(params_p05))
        shutil.copyfile(foldername + '/structure.json',
                        foldername + '/sj_+0.5/structure.json')

    return None


def create_general_supercell(structure, size=12.5):
    """
    Use algorithm to generate general supercell.

    Creates supercell of a form that breaks initial bravais lattice symmetry
    as well as tries to find the most uniform configuration containing the
    least number of atoms. Only works in 2D so far!
    """
    from ase.build import make_supercell
    import numpy as np
    # b1 = n1*a1 + m1*a2
    # b2 = n2*a1 + m2*a2
    # we restrict ourselves such that m1=0
    # the respective new cell is then:
    # P = [[n1, 0, 0], [n2, m2, 0], [0, 0, 1]]
    print('INFO: set up general supercell.')
    sc_structuredict = {}
    for n1 in range(1, 10):
        for n2 in range(0, 10):
            for m2 in range(1, 10):
                # set up transformation, only for symmetry broken setup
                if not (n1 == m2 and n2 == 0):
                    P = np.array([[n1, 0, 0], [n2, m2, 0], [0, 0, 1]])
                    sc_structure = make_supercell(prim=structure, P=P)
                    # now implement the postprocessing
                    sc_structuredict[str(n1) + str(0) + str(n2) + str(m2)] = {
                        'structure': sc_structure}
    print('INFO: created all possible linear combinations of the old lattice.')

    # check physical distance between defects
    indexlist = []
    structurelist = []
    numatoms_list = []
    stdev_list = []
    for i, element in enumerate(sc_structuredict):
        cell = sc_structuredict[element]['structure'].get_cell()
        distance_xx = np.sqrt(cell[0][0]**2 + cell[0][1]**2 + cell[0][2]**2)
        distance_yy = np.sqrt(cell[1][0]**2 + cell[1][1]**2 + cell[1][2]**2)
        distance_xy = np.sqrt((
            cell[0][0] + cell[1][0])**2 + (
            cell[0][1] + cell[1][1])**2 + (
            cell[0][2] + cell[1][2])**2)
        distance_mxy = np.sqrt((
            -cell[0][0] + cell[1][0])**2 + (
            -cell[0][1] + cell[1][1])**2 + (
            -cell[0][2] + cell[1][2])**2)
        distances = [distance_xx, distance_yy, distance_xy, distance_mxy]
        stdev = np.std(distances)
        sc_structuredict[element]['distances'] = distances
        sc_structuredict[element]['stdev'] = stdev
        if min(distances) <= size:
            sc_structuredict[element]['suitable'] = False
        elif min(distances) > size:
            sc_structuredict[element]['suitable'] = True
            indexlist.append(i)
            stdev_list.append(stdev)
            numatoms_list.append(len(sc_structuredict[element]['structure']))
            structurelist.append(sc_structuredict[element]['structure'])

    lowlist = []
    lowstruclist = []
    for j, structure in enumerate(structurelist):
        if len(structure) == min(numatoms_list):
            lowlist.append(stdev_list[j])
            lowstruclist.append(structure)
    for k, structure in enumerate(lowstruclist):
        if lowlist[k] == min(lowlist):
            print('INFO: optimal structure found: {}'.format(structure))
            finalstruc = structure

    return finalstruc


if __name__ == '__main__':
    main.cli()
