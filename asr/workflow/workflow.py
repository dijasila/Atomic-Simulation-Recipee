from typing import List
from pathlib import Path
import numpy as np
import os

from ase.db import connect
from ase.io import read
from asr.core import read_json

from asr.dynstab import dynstab_manager
from asr.workflow.bilayerutils import task, silentremove, listdirs, recently_updated, \
    select_resources_based_on_size, all_tasks_done, cleanup_gs, copy_gs_results, \
    is_magnetic, has_tm3d, has_atom_at_origin, get_vacuum, verify_monolayer, \
    starting_magmoms, analyse_w_wo_U, hubbardu_needed, select_stables, gs_subfolders, \
    create_fullrelax_folder, select_largest_ebs, is_zscan_valid, many_invalid_zscans

db_home = '/home/niflheim2/cmr/C2DB-ASR/collected-databases/'
c2db = connect(f'{db_home}/c2db-first-class-20220908.db')


###########################
def getresources() -> str:
    """
    Randomly generate a resource.
    Use this to distribute jobs evenly across multiple partitions.

    Note: Use with supervision based on availablity of nodes and jobs being submitted.

    Note: To distribute resources based on the number of atoms, use the alternative
          function select_resources_based_on_size from the bilayerutils.
    """
    r = np.random.rand()
    if r < 0.5:
        resources = "40:50h"
    elif r < 0.9:
        resources = "56:50h"
    else:
        resources = "48:2d"

    return resources


###########################
# Bilayer tasks ###########
###########################

###########################
def calc_vdw(folder: str) -> None:
    """Calculates D3 energy correction for the structure in the folder."""
    from ase.calculators.calculator import get_calculator_class

    result_file = f"{folder}/vdw_e.npy"

    # When the result is saved, the extra files are removed to cleanup the folder.
    if os.path.isfile(result_file):
        silentremove(f"{folder}/vdwtask.*.*", check_other_file=result_file)
        return

    # This calculation is done while submmiting the workflow.
    Calculator = get_calculator_class("dftd3")
    calc = Calculator()

    atoms = read(f"{folder}/structure.json")
    atoms.set_calculator(calc)

    e = atoms.get_potential_energy()
    np.save(result_file, e)


###########################
def create_bilayers(folder: str,
                    structure_file: str = "structure_initial.json",
                    hundrule: bool = False):
    """
    Submits stack_bilayer task.
    - Creates monolayer "structure_adjusted.json". An atom at origin and adjusted vacuum
    - Generates the inital pool of bilayers.
    - The interlayer distance is not known at this point & a fixed value (6A) is used.
    - The bilayer structure created here is called "bilayerprototype.json"
      - By default we use the magmoms of the structure_file to set up bilayerprototypes
      - If hundrule, we initialize magmoms with hundrule and we donot need calc on atoms
    - The stacking information is stored in: 'translation.json' and 'transformdata.json'

    Raises:
        ValueError: if you use more than 1 core in resources.
        AttributeError: if not hundrule and structure does not have a calc attached.
    """
    # In rare cases of TIMEOUT, resubmit with longer time not more cores.
    resources = "1:10m"

    # stack_bilayer recipe works with one core.
    if resources[0:2] != "1:":
        raise ValueError('stack_bilayer recipe works with one core.')

    # If we do not use hundrule for initial magnetic moments,
    # structures should have calculator on them to initialize magmoms.
    atoms = read(f'{folder}/{structure_file}')
    if not hundrule and not atoms.calc:
        raise AttributeError('Structure does not have calc: {folder}/{structure_file}')

    # Add options to task if hundrule
    options = f' --hund-rule {hundrule}' if hundrule else ''

    return [task(f"asr.stack_bilayer{options}",
                 creates=['structure_adjusted.json',
                          'results-asr.stack_bilayer.json'],
                 resources=resources, folder=folder)]


###########################
def get_descriptors(blfolders: List[str]):
    """Get bilayer descriptor in html and latex formats"""
    tasks = []

    for bl in blfolders:
        tasks += [task("asr.bilayerdescriptor", resources="1:5m", folder=bl)]

    return tasks


###########################
def monolayer_treatment(folder: str,
                        structure_file: str,
                        initial_magmoms: List[float],
                        e_threshold: float = 1e-6,
                        hubbardu: float = 0.0):
    """
    Submits gs calculation and creates "structure.json" in the monolayer folder.

    Note: Monolayer gs has to be done with same parameters as bilayers for binding
          energy. Do not use monolayer gs from c2db.

    - If monolayer has tm3d:
        - Creates and does calculations in subfolders: gs-withU and gs-withoutU.
        - Modifies params.json with: mixer sum and beta = 0.02 (helps convergence).
        - Checks the +U gap and decided if +U id needed.

    - If monolayer does not have tm3d:
        - Creates and does calculations in a subfolder: gs-withoutU. This calculation
          could be done in the main folder but for consistency it is done in a subfolder
        - Modifies params.json with: mixer sum and beta = 0.05

    - In both cases:
        - Create structure.json file in the main folder with calc & magmoms on it.
        - Relevant (with/without U) gs result files are copied in the main folder.
        - gs.gpw files are removed from the subfolders.

    Project Notes:
    - There are 3 structure files in the monolayer folder.
    - We preserved the initial monolayer structure as "structure_initial.json".
    - Before this stack_bilayer recipe is done and created "structure_adjusted" with:
      - an atom at the origin
      - same vacuum as bilayers
    - In this function, gs calculation is done and "structure.json" will be created
      with calculator on it.

    Raises:
        ValueError: If there is not an atom in the origin
        ValueError: If the vacuum of structure_file != vaccum of bilayerprototpyes.
    """
    tasks = []
    res = getresources()
    tm3d = has_tm3d(folder, structure_file)

    # The workflow can skip the steps of this function if the result file exists
    if os.path.isfile(f'{folder}/results-asr.gs.json'):
        return tasks

    if not has_atom_at_origin(f'{folder}/{structure_file}'):
        raise ValueError(f'No atom at the origin: {f"{folder}/{structure_file}"}')

    ml_vacuum = get_vacuum(f'{folder}/{structure_file}')
    blfolder = listdirs(folder)[0]
    bl_vacuum = get_vacuum(f'{blfolder}/bilayerprototype.json')
    if abs(bl_vacuum - ml_vacuum) > 0.01:
        raise ValueError(f'Monolayer and bilayers have different vacuum: {folder}')

    if not tm3d:
        mixer = {'type': 'sum', 'beta': 0.05}
    else:
        mixer = {'type': 'sum', 'beta': 0.02}

    subfolder_noU, subfolder_U = gs_subfolders(top_folder_path=folder,
                                               subfolder_noU='gs-withoutU',
                                               subfolder_U='gs-withU' if tm3d else '',
                                               structure_path=structure_file,
                                               initial_magmoms=initial_magmoms,
                                               hubbardu=hubbardu,
                                               e_threshold=e_threshold,
                                               mixer=mixer)

    for subf in [subfolder_noU, subfolder_U]:
        # Checks that folder is not empty string. It happens for subfolder_U if not tm3d
        if subf:
            tasks += [task("asr.gs", resources=res, folder=subf, restart=1)]
            calc_vdw(subf)

    # Checks if we need HubbardU and brings the chosen gs results in the main folder
    analyse_w_wo_U(tm3d, folder, subfolder_noU, subfolder_U)

    return tasks


###########################
def submit_zscan(blfolders: List[str], magnetic: bool):
    """
    Submits zscan tasks:
    - If not magnetic: submits with 'optimize' method.
    - If magnetic or non-magnetic failed once: submits with 'zscan' method.

    Note:
    - U correction is not applied in the z-scan level of the project.
    - It starts from bilayerprototype with final magmoms of 'structure_initial' as we
      set in stack_bilayer recipe.

    Special conditions:
    - If "shifted_from" and non-magnetic: Restart from previous interlayer.
    - For non-magnetics: If there is a zscan.txt but not the result file of the recipe
        - The recipe has failed: we resubmit with 'zscan' method.
        - The calculation is running now: we leave it to finish.
    """
    res = getresources()
    tasks = []

    for blfolder in blfolders:
        distancearg = ""
        method = ""

        if not magnetic:

            # Starting "shifted_from" and non-magnetic with the previous interlayer.
            translation = read_json(f'{blfolder}/translation.json')
            if 'Shifted_from' in translation:
                original = translation['Shifted_from']
                resultsfname = f"{blfolder}/../{original}/results-asr.zscan.json"
                if os.path.exists(resultsfname):
                    h = read_json(resultsfname)['interlayer_distance']
                    distancearg = f' --distance {h+0.2}'

            # First submission of non-magnetic systems
            if not os.path.isfile(f'{blfolder}/zscan.txt'):
                method = 'optimize'

            # If there is a zscan.txt file and running now: let it continue.
            elif recently_updated(blfolder, 'zscan.txt'):
                continue

            # If "optimize" failed: resubmit with "zscan" and restart the distance.
            else:
                method, distancearg = 'zscan', ''

        else:
            method, distancearg = 'zscan', ''

        tasks += [task(f"asr.zscan{distancearg} --method {method}", resources=res,
                       creates=["results-asr.zscan.json"], restart=2, folder=blfolder)]

    return tasks


#######################
def bilayer_gs(blfolders: List[str],
               structure_file: str,
               monolayer_folder: str,
               e_threshold: float = 1e-6,
               hubbardu: float = 0.0,
               cleanup: bool = False):
    """
    Calculate bilayer gs inside subfolders with or without U.
     - Eb will be calculated without U. We do this calculation inside a subfolder.
     - If U is needed:
        - If the structure is not magnetic, make a gs-withU folder
        - If the structure is magnetic, we will not create a gs-withU because the gs
          will be done by the interlayer magnetic recipe in the main folder.
    """
    res = getresources()
    tasks = []

    magnetic = is_magnetic(monolayer_folder, c2db)
    mixer = {'type': 'sum', 'beta': 0.02} if magnetic else {'type': 'sum', 'beta': 0.05}

    subfolder_noU = 'gs-withoutU' if abs(hubbardu) < 0.01 else ''
    subfolder_U = 'gs-withU' if abs(hubbardu) >= 0.01 else ''

    for bilayer_folder in blfolders:
        subfolder_noU, subfolder_U = gs_subfolders(top_folder_path=bilayer_folder,
                                                   subfolder_noU=subfolder_noU,
                                                   subfolder_U=subfolder_U,
                                                   structure_path=structure_file,
                                                   initial_magmoms=[],
                                                   hubbardu=hubbardu,
                                                   e_threshold=e_threshold,
                                                   mixer=mixer)

        for subf in [subfolder_noU, subfolder_U]:
            if subf:
                tasks += [task("asr.gs", resources=res, folder=subf, restart=2)]
                calc_vdw(subf)

        if cleanup:
            cleanup_gs(subf)

    return tasks


###########################
def create_final_structure(blfolder: str, hubbardu: float, mlfolder: str = ''):
    """
    Creates the final strucutre.json file in the bilayer folder.
      - If the material is not magnetic: we copy the gs results and strucutre file from
        the relevant subfolder with or without U.
      - If the material is magnetic: the interlayer_magnetic_exchange will create the
        structure file (FM or AFM) and the gs.gpw in the main bilayer folder.
    """
    tasks = []

    if os.path.isfile(f'{blfolder}/structure.json') and \
       os.path.isfile(f'{blfolder}/gs.gpw'):
        return tasks

    mlfolder = f'{blfolder}/../' if not mlfolder else mlfolder

    # Selecting resources based on number of atoms in the bilayer.
    requested_resources = {"small": ["15", "56:50h"],
                           "large": ["21", "112:50h"]}

    res = select_resources_based_on_size(f'{blfolder}/structure_zscan.json',
                                         selected_resources=requested_resources)

    if not is_magnetic(mlfolder, c2db):
        if hubbardu_needed(mlfolder):
            copy_gs_results(origin=f'{blfolder}/gs-withU', destination=blfolder)
        else:
            copy_gs_results(origin=f'{blfolder}/gs-withoutU', destination=blfolder)

    else:
        uvalue = hubbardu if hubbardu_needed(mlfolder) else 0.0

        tasks += [task(f"asr.interlayer_magnetic_exchange -u {uvalue}",
                  resources=res, folder=blfolder, restart=2,
                  creates="results-asr.interlayer_magnetic_exchange.json")]

    return tasks


###########################
def get_binding_energies(blfolders: List[str]):
    tasks = []

    for bl in blfolders:
        tasks += [task("asr.bilayerbinding_overview", resources="1:10m", folder=bl)]

    return tasks


###########################
# Properties tasks ########
###########################
def bandstructure(folder: str):
    """
    Submits band structure task inside the input folder.
    - gs is done in a subfolder so we cannot use the deps of mq to check it is done
      but the result file of gs exists.
    """
    res = getresources()
    if not os.path.isfile(f'{folder}/results-asr.gs.json'):
        return []

    return [task("asr.bandstructure", resources=res, folder=folder, restart=2)]


##########################
def pdos(folder: str):
    """
    Submits PDOS task inside the input folder.
    - gs is done in a subfolder so we cannot use the deps of mq to check it is done
      but the result file of gs exists.
    """
    res = getresources()
    if not os.path.isfile(f'{folder}/results-asr.gs.json'):
        return []

    return [task("asr.pdos", resources=res, folder=folder)]


##########################
def projected_bandstructure(folder: str):
    """ Submited projected band structure task inside the input folder"""

    return [task("asr.projected_bandstructure", resources='1:10m',
            folder=folder, restart=2, deps=['asr.bandstructure'])]


##########################
def fermisurface(folder: str):
    """
    Submits fermi surface task inside the input folder if the material is metalic.
    - gs is done in a subfolder so we cannot use the deps of mq to check it is done
      but the result file of gs exists.
    """
    if not os.path.isfile(f'{folder}/results-asr.gs.json'):
        return []

    gap = read_json(f"{folder}/results-asr.gs.json").get("gap")

    if gap == 0.0:
        return [task("asr.fermisurface", resources="1:30m", folder=folder)]
    else:
        return []


##########################
def emasses(folder: str):
    """
    Submit emasses task inside the input folder if the material has a gap
    - gs is done in a subfolder so we cannot use the deps of mq to check it is done
      but the result file of gs exists.
    """
    res = getresources()
    if not os.path.isfile(f'{folder}/results-asr.gs.json'):
        return []

    gap = read_json(f"{folder}/results-asr.gs.json").get("gap")

    if gap > 0.01:
        return [task("asr.emasses", resources=res, folder=folder, restart=2)]
    else:
        return []


##########################
def hse(folder: str):
    """ Submits HSE task inside the input folder"""
    res = getresources()
    if not os.path.isfile(f"{folder}/results-asr.gs.json") or \
       not os.path.isfile(f'{folder}/results-asr.bandstructure.json'):
        return []
    gap = read_json(f"{folder}/results-asr.gs.json").get("gap")

    if gap > 0.01:
        return [task("asr.hse", resources=res, folder=folder,
                restart=2, deps=['asr.bandstructure', 'asr.gs'])]
    else:
        return []


##########################
def raman(folder: str, magnetic: bool, structure_file: str = 'structure.json'):
    """ Submits Raman tasks for non-magnetic materials with upto 16 atoms/cell"""
    atoms = read(f'{folder}/{structure_file}')

    # Selecting resources based on number of atoms in the bilayer.
    requested_resources = {"small": ["13", "56:50h"],
                           "medium": ["15", "112:50h"],
                           "large": ["21", "224:50h"]}

    res = select_resources_based_on_size(structure_file=f'{folder}/{structure_file}',
                                         selected_resources=requested_resources)

    if not magnetic and len(atoms) < 17:
        return [task("asr.ramanpol", resources=res, folder=folder)]
    else:
        return []


##########################
def full_relax(mlfolder: str, blfolders: List[str]):
    """
    Creates subfolders and submits fullrelax without U for monolayer and bilayers.
    Note: In this project we don't use U for structure relaxations.
    """
    res = getresources()
    tasks = []
    calculation_subfolder = 'fullrelax'

    # Monolayer full relaxation
    create_fullrelax_folder(top_folder=mlfolder, subfolder_name=calculation_subfolder)

    tasks += [task("asr.relax --d3", creates=["results-asr.relax.json"],
              restart=2, resources=res, folder=f'{mlfolder}/{calculation_subfolder}')]

    # Bilayer full relax for all the bilayers in the 3meV window
    for blfolder in blfolders:
        create_fullrelax_folder(top_folder=blfolder,
                                subfolder_name=calculation_subfolder)

        tasks += [task("asr.relax --d3", creates=["results-asr.relax.json"],
                  restart=2, resources=res,
                  folder=f'{blfolder}/{calculation_subfolder}')]

    return tasks


###########################
# Create tasks ############
###########################

class WorkflowParams:
    """
    This class contains all the control parameters of the workflow.

    Note: In this project we use the magnetic moment of the converged monolayer for the
    initial magmoms. If magnetic info of the monolayer is not available use hund_rule.
    """

    # Do not remove any item. Set it to None if you want to ignore it
    monolayer_criteria = {'max_natom': 10,
                          'c2db': {'in_c2db' : True,
                                   'dynamic_stability_phonons': 'high',
                                   'dynamic_stability_stiffness': 'high',
                                   'thermodynamic_stability_level': 3,
                                   'ehull': 0.1,
                                   },
                          'has_tm3d': None,
                          }

    # List of monolayers that don't have your criteria but you want to include them.
    forced_monolayers = ['P4-276f0a298324']

    hund_rule = False
    hubbardu = 4.0
    e_threshold = 1e-6
    e_threshold_dynstab = 1e-5
    deltaE = 3
    ignore_failed_zscan = True
    ignore_failed_percentage = 20 / 100 if ignore_failed_zscan else 0.0


###########################
def create_tasks():
    """
    To start the bilayer specific and the whole workflow there should be monolayer
    folders with following conditions:

    - Mandatory: "structure_initial.json" file in the folder.
      Raises:
          FileNotFoundError: if "structure_initial.json" file in the monolayer folder.

    - Suggested: The structure should have calculator on it.
        - This will allow using converged magmoms.
        - If mangnetic info is not available use hund_rule or an error will be raised.

    - Suggested: Use c2db uid as monolayer folder name.
        - This will help accessing c2db data and cross-referencing the two databases.
        - If a different name is used, the monolayer can be removed for not having our
          selection criteria (i. e. stability info is not available). In such cases add
          the folder name to excpetions list.

    To start the properties part of the workflow, the final bilayer structure.json and
    gs.gpw must be created in the main folder:

    - For magnetics: since interlayer_magnetic creates the files and it is done for
      stables only, the other magnetic bilayers can not calculate properties.

    - For non-magnetics: properties can be calculated for any bilayer in the zscan
      selected window (slide stability is not needed to create the necessary files).
      Increase the the zscan_selcted window add properties for more unstable bilayers.
    """
    tasks = []

    mlfolder = Path(".").absolute()
    print('\n\n>>>>> Monolayer: ', mlfolder)

    monolayer_criteria = WorkflowParams.monolayer_criteria
    hubbardu = WorkflowParams.hubbardu
    e_threshold = WorkflowParams.e_threshold
    ignore_failed_zscan = WorkflowParams.ignore_failed_zscan
    ignore_failed_percentage = WorkflowParams.ignore_failed_percentage
    ignore_failed_gs = WorkflowParams.ignore_failed_gs
    deltaE = WorkflowParams.deltaE
    hund_rule = WorkflowParams.hund_rule

    if not os.path.isfile(f'{mlfolder}/structure_initial.json'):
        raise FileNotFoundError('Monolayer "structure_initial.json" file not found >',
                                mlfolder)

    tm3d = has_tm3d(mlfolder, structure_file="structure_initial.json")

    # If hund_rule = False: structure_initial should have calculator on it.
    ml_magmoms = starting_magmoms(structure_path=f'{mlfolder}/structure_initial.json',
                                  hundrule=hund_rule)

    # Verify if the monolayer has the monolayer_criteria:
    # - To add the monolayer despite the selection, add it's uid to exceptions list.
    # - To verify the material has some selection criteria according to c2db, the
    #   monolayer folder name should be the c2db uid.
    mlfolder = verify_monolayer(mlfolder,
                                criteria=monolayer_criteria,
                                c2db=c2db,
                                structure_file="structure_initial.json",
                                exceptions=WorkflowParams.forced_monolayers)

    if not mlfolder:
        return tasks

    # Before this we only have structure_initial.json file for monolayer
    tasks += create_bilayers(mlfolder,
                             structure_file="structure_initial.json",
                             hundrule=hund_rule)

    # We need the stacking to be done before going to zscan
    if not all_tasks_done(tasks):
        return tasks

    # Ensure the subfolders in mlfolder have the signature of the bilayer folder names
    all_blfolders = [bl for bl in listdirs(mlfolder) if '-2-' in bl]

    tasks += get_descriptors(all_blfolders)

    # At this point we have "structure_adjusted.json" for monolayer.
    tasks += monolayer_treatment(mlfolder,
                                 structure_file="structure_adjusted.json",
                                 initial_magmoms=ml_magmoms,
                                 e_threshold=e_threshold,
                                 hubbardu=hubbardu)

    # monolayer_treatment should be done before zscan so that gs.gpw is created to check
    # if monolayer is magnetic.
    if not all_tasks_done(tasks):
        print('Monolayer gs with and without not finished yet', mlfolder)
        return tasks

    # Use c2db uid or monolayer gs.gpw to verify if magnetic.
    # If it neither finds a gs.gpw nor finds monolayer on c2db return None.
    magnetic = is_magnetic(mlfolder=mlfolder, c2db=c2db)
    if magnetic is None:
        print("Monolayer gs is not done and we can not verify if magnetic")
        return tasks

    tasks += submit_zscan(all_blfolders, magnetic)

    if not all_tasks_done(tasks):
        return tasks

    # If zscan fails for some bilayers, decide to continue or stop the workflow
    if many_invalid_zscans(all_blfolders, ignore_failed_zscan,
                          ignore_failed_percentage):
        return tasks

    # If zscan is not valid or failed don't continue with that bilayer
    valid_blfolders = [bl for bl in all_blfolders if is_zscan_valid(bl)]

    # Decide based on zscan for which bilayers gs calculation is needed.
    zscan_selected = select_largest_ebs(valid_blfolders, tm3d, deltaE=10,
                                        hubbardU=False, source='zscan',
                                        ignore_failed=ignore_failed_zscan)
    print(f">>> From {len(all_blfolders)} bilayers generated, ",
          f"{len(zscan_selected)} are selected by zscan.")

    if not zscan_selected:
        print('Stopped because no bilayer was selected from zscan step.')
        return tasks

    # Eb will be calculated without U: all bilayers need gs-withoutU.
    tasks += bilayer_gs(blfolders=zscan_selected,
                        structure_file="structure_zscan.json",
                        monolayer_folder=mlfolder,
                        e_threshold=e_threshold,
                        hubbardu=0.0)

    # Eb is calculated from gs and if in the 3meV/A2 window we go for DynStab.
    # Monolayer gs should be done before this step.
    gs_selected = select_largest_ebs(zscan_selected, tm3d, deltaE=deltaE,
                                     hubbardU=False, source='gs',
                                     ignore_failed=ignore_failed_gs)

    print(f">>> From {len(all_blfolders)} bilayers generated, ",
          f"{len(zscan_selected)} are selected by zscan and",
          f"among those {len(gs_selected)} selected by gs")

    # Slide stability subworkflow:
    for blfolder in gs_selected:
        tasks += dynstab_manager(blfolder,
                                 start_structure='structure_zscan.json',
                                 dynstab_path='DynStab',
                                 step_num=2,
                                 resources=getresources())

    stable_selected = select_stables(gs_selected)

    # Up to this point of the workflow all the steps where without U.
    # If the bilayer is not magnetic and needs U: we should calculate the gs with U
    # If it is magnetic, the interlayer_magnetic_exchange recipe will do the gs of FM
    # and AFM and uses the most stable
    if hubbardu_needed(mlfolder) and not magnetic:
        tasks += bilayer_gs(blfolders=zscan_selected,
                            structure_file="structure_zscan.json",
                            monolayer_folder=mlfolder,
                            e_threshold=e_threshold,
                            hubbardu=hubbardu)

    # Bilayer folders that have gs.gpw and can move on to properties
    blfolders_with_gs = stable_selected if magnetic else zscan_selected

    for blfolder in blfolders_with_gs:
        tasks += create_final_structure(blfolder, magnetic, hubbardu)

    tasks += get_binding_energies(mlfolder, blfolders_with_gs)

    # In the first step we calculate properties for stable bilayers.
    # We can later expand this to blfolders_with_gs for all except raman
    for blfolder in stable_selected:
        print("Bilayer reached properties: ", blfolder)

        if not os.path.isfile(f'{blfolder}/structure.json') or \
           not os.path.isfile(f'{blfolder}/gs.gpw'):
            continue

        tasks += bandstructure(blfolder)
        tasks += pdos(blfolder)
        tasks += raman(blfolder, magnetic)
        tasks += projected_bandstructure(blfolder)
        tasks += fermisurface(blfolder)

    print()
    return tasks


if __name__ == "__main__":
    tasks = create_tasks()

    for ptask in tasks:
        print(ptask, ptask.is_done())
