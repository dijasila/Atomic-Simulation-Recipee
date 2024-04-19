from typing import Dict, List, Tuple, Any
from asr.core import read_json, write_json
from ase.io import read
from pathlib import Path
import numpy as np
import os
import shutil
import glob
import time

###########################
"""
Bilayer workflow does a lot of gs calculations.
This params dict is our reference and we apply changes when needed.

Note1: The hubbard U values used in this workflow is U=4 for all TM3d elements.
    Here we provide an alternative of element dependent U values if needed.
Note2: We need reproducible total energies to calculate the bilayer binding energies.
    Total energies of GPAW can differ from one run to another with same calculation
    parameters. So we use stronger convergence criteria on etot & density.
    The energy convergence in this worflow is in most cases on etot rather
    than etot/electron.
Note3: Because we have a hard convergence criteria, the maxiter needs to be large
Note4: The default of mixer is set to 'sum' with smaller than default beta value. This
    has shown better chance of convergence for magnetic bilayers.
"""
PARAMS = {
    "asr.gs@calculate": {
        "calculator": {
            "mixer": {
                "method": "sum",
                "beta": 0.02,
                "history": 3,
                "weight": 50
            },
            "setups": {
                "V": ":d,3.1",
                "Cr": ":d,3.5",
                "Mn": ":d,3.8",
                "Fe": ":d,4.0",
                "Co": ":d,3.3",
                "Ni": ":d,6.4",
                "Cu": ":d,4.0"
            },
            "name": "gpaw",
            "mode": {
                "name": "pw",
                "ecut": 800
            },
            "xc": "PBE",
            "basis": "dzp",
            "kpts": {
                "density": 12.0,
                "gamma": True
            },
            "occupations": {
                "name": "fermi-dirac",
                "width": 0.05
            },
            "convergence": {
                "energy": {
                    "name": "energy",
                    "tol": 0.00001,
                    "relative": False,
                    "n_old": 4
                },
                "bands": "CBM+3.0",
                "density": 0.000001
            },
            "symmetry": {},
            "nbands": "200%",
            "txt": "gs.txt",
            "maxiter": 7000,
            "charge": 0
        }
    }
}


###########################
def listdirs(path: str) -> List[str]:
    """ List the subdirectories of a directory"""
    if not isinstance(path, str):
        path = str(path.absolute())

    if not os.path.isdir(path):
        raise ValueError(f"'{path}' is not a valid directory.")

    return [f"{path}/" + x for x in os.listdir(path) if os.path.isdir(f"{path}/{x}")]


###########################
def parse_folders() -> List[Path]:
    """
    Parse the provided folder paths/pattern in the input into list of absolute paths.

    Raises:
        ValueError: If no folder path is provided in the input.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("folders", nargs="*", help="Folders to analyse.")
    args = parser.parse_args()

    if len(args.folders) == 0:
        raise ValueError(f"No folder paths provided in the input")

    return [Path(x).absolute() for x in args.folders]


###########################
def select_resources_based_on_size(structure_file: str,
                                   selected_resources: Dict[str, List[str]] = {}
                                   ) -> str:
    """
    Select resources for a task based on the number of atoms in the structure.
     - gets an optional input dictionary for the resources to assign to structures
       below certain number of atoms.
    """
    natoms = len(read(structure_file))

    if not selected_resources:
        available_resources = {"small": ["13", "40:50h"],
                               "medium": ["17", "56:50h"],
                               "large": ["21", "112:50h"]}
    else:
        available_resources = selected_resources

    for size in available_resources:
        if natoms < int(available_resources[size][0]):
            resources = available_resources[size][1]

    return resources


###########################
def task(*args, **kwargs):
    """Get MyQueue task instance."""
    from myqueue.task import task as mqtask

    name = kwargs.get("name") or args[0]
    if "creates" not in kwargs:
        kwargs["creates"] = [f"results-{name}.json"]
    return mqtask(*args, **kwargs)


###########################
def all_tasks_done(list_of_tasks):
    """Determine if all tasks in list_of_tasks are done."""
    return all([task.check_creates_files()
                for task in list_of_tasks])


###########################
def silentremove(filepath_pattern: str, check_other_file: str = '') -> None:
    """
    Remove files matching the filepath pattern if they exist.
    Optionally you can remove the file only if another file exists.
    """

    if check_other_file and not os.path.exists(check_other_file):
        raise FileNotFoundError(f"File does not exist: '{check_other_file}'")

    for file_path in glob.glob(filepath_pattern):
        if not check_other_file or os.path.isfile(check_other_file):
            if os.path.isfile(file_path):
                os.remove(file_path)


###########################
def cleanup_gs(folder: str) -> None:
    """ Remove extra files created with a gs recipe if the results are saved """

    # We will removes the files in each "value" list if the file in the "key" exists
    conditional_files = {
        f'{folder}/results-asr.gs.json': [
            'results-asr.magnetic_anisotropy.json',
            'gs.gpw',
            'asr.gs.*.err',
            'asr.gs.*.out'],
        f'{folder}/vdw_e.npy': [
            f"{folder}/vdwtask.*.out",
            f"{folder}/vdwtask.*.err"]
    }

    for conditional_file, files in conditional_files.items():
        for file_pattern in files:
            silentremove(file_pattern, check_other_file=conditional_file)


###########################
def add2file(filename: str, item: str) -> None:
    """
    Append an item to a file if the file exists and the item is not already present.
    If the file does not exist, create it and write the item.

    filename (str): can be either the name of a file in current dir or path to the file.
    """

    if os.path.isfile(filename):
        with open(filename, "r") as file:
            items = {line.strip() for line in file}

        if item not in items:
            with open(filename, "a") as file:
                file.write(item + "\n")

    else:
        with open(filename, "w") as file:
            file.write(item + "\n")


###########################
def recently_updated(folder: str,
                     filename: str,
                     time_in_seconds: int = 3600) -> bool:
    """
    Check if a specific file has been updated within a given time frame (in seconds).

    Note: Some time bilayer workflow resubmits jobs that failed with new parameters
        So we need to make sure that the previous job has been submitted but is not
        running at the moment.

    Args:
        folder (str): The folder path to search for recently updated files.
        filename (str): The name of the file to check.
        time_in_seconds (int): The time threshold in seconds for recent updates.

    Returns:
        bool: True if the file has been recently updated; otherwise, False.
    """

    # Calculate the cutoff time using the current time and the specified time_in_seconds
    cutoff_time = time.time() - time_in_seconds

    # Iterate over the folder and its subdirectories to find recently modified files
    for root, _, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)

            # Check if the file was modified within the specified time frame
            if os.stat(file_path).st_mtime > cutoff_time:

                # If the file matches the specified filename, return True
                if file == filename:
                    return True

    # Return False if the file was not found or not recently updated
    return False


###########################
def set_info(folder: str,
             key: str,
             value: Any,
             update_key: bool = False,
             filename: str = 'info.json') -> None:
    """
    Append a key-value pair to a JSON file if the file exists.
    If the file does not exist, create it and add the key-value pair.

    Args:
        folder (str): The folder path where the JSON file is located/should be created.
        key (str): The key to add to the JSON file.
        value (Any): The value associated with the key to add to the JSON file.
        update_key (bool, optional): If True and the key exists, update the value
        filename (str, optional): The name of the JSON file. Defaults to 'info.json'.

    Raises:
        FileNotFoundError: If the specified folder does not exist.
        ValueError: If the key already exists in the JSON file and update_key =! True.
    """

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"The specified folder '{folder}' does not exist.")

    infofile_path = os.path.join(folder, filename)

    info_data = read_json(infofile_path) if os.path.isfile(infofile_path) else {}

    if not update_key and key in info_data:
        raise ValueError(f"The key '{key}' already exists in the '{filename}' file.")

    info_data[key] = value

    write_json(infofile_path, info_data)


###########################
def has_atom_at_origin(structure_path: str) -> bool:
    """ Checks if the structure has an atom in the in-plane origin"""
    atoms = read(structure_path)
    for atom in atoms:
        if np.linalg.norm(atom.position[0, 0:2]) < 1e-3:
            return True
    print('>>>>>>>>>> WARNING: There was no atom at the origin of monolayer')
    return False


###########################
def get_vacuum(structure_path: str) -> float:
    """ Returns the vacuum of 2D strucutr"""
    atoms = read(structure_path)

    positions = atoms.positions[:, 2]
    width = max(positions) - min(positions)

    return atoms.cell[2, 2] - width


#########################
def create_gs_params(folder: str,
                     replace_calc_dict: Dict[str, Any]) -> None:
    """
    Replace values in the global PARAMS dict based on the keys in replace_calc_dict
    The updated content is then written to a params.json file.

    If there is a params file in the folder:
        - If it has gs@calculate: avoid overwriting and raise an error
        - If it contains parameters for other recipes, add gs@calculate

    Args:
        folder (str): The path that the params.json file should be created.
        replace_calc_dict (Dict[str, Any]): Dictionary containing new calc parameters.

    Raises:
        KeyError: If a key in replace_calc_dict does not exist in global PARAMS.
        RuntimeError: If there is already as params file with gs params in the folder
    """
    params_file = f'{folder}/params.json'

    if os.path.isfile(params_file) and "asr.gs@calculate" in read_json(params_file):
        raise RuntimeError(f"Params file for gs@calculate already exists in: {folder}")

    # If there is a params file but with parameters for other recipes not gs
    # then we want to add the gs params
    current_params = read_json(params_file) if os.path.isfile(params_file) else {}

    gs_params = PARAMS.copy()["asr.gs@calculate"]["calculator"]

    # Check if all keys in replace_dict exist in global PARAMS
    missing_keys = set(replace_calc_dict.keys()) - set(gs_params.keys())
    if missing_keys:
        raise KeyError(
            f"Attempting to modify keys missing in the default PARAMS:{missing_keys}"
        )

    # Updating the calculator dictionary for gs
    gs_params.update(replace_calc_dict)

    current_params.update({"asr.gs@calculate": {"calculator": gs_params}})

    # write the updated parameters in a params.json file
    write_json(params_file, current_params)


#########################
def modify_params(folder: str,
                  etot_threshold: float = -1.0,
                  uvalue: float = 0,
                  mixer: Dict[str, Any] = {},
                  symmetry: bool = True) -> None:
    """
    Create a dict of updated gs calculator params and create a params file with them.

    For consistency, we hard coded the convergence on etot rather than etot/electron
        - etot_threshold >= 0: we set this value for converence on etot
        - etot_threshold < 0: we use the defaults of PARAMS for converence on etot

    The Hubbard U correction can be applied in three ways:
        - U > 0: Set a constant U value for all TM3D elements.
        - U < 0: Set specific U values for each TM3D element.
        - U = 0: Exclude the Hubbard U correction (default).
    """
    replace_calc_params = {}

    # correct energy convergence
    if etot_threshold >= 0:
        replace_calc_params["convergence"] = {
            "energy": {
                "name": "energy",
                "tol": etot_threshold,
                "relative": False,
                "n_old": 3}
        }

    # change mixer type and/or beta
    if mixer:
        replace_calc_params["mixer"] = {
            "method": mixer["type"],
            "beta": mixer["beta"]
        }

    # change symmetry: sometimes you may want to turn symmetries off
    if not symmetry:
        replace_calc_params["symmetry"] = "off"

    # Choose Hubbard U.This can be written more compact but it is more readable this way
    if uvalue == 0:
        replace_calc_params["setups"] = {}

    elif uvalue > 0:
        replace_calc_params["setups"] = {
            "V": f":d,{uvalue}",
            "Cr": f":d,{uvalue}",
            "Mn": f":d,{uvalue}",
            "Fe": f":d,{uvalue}",
            "Co": f":d,{uvalue}",
            "Ni": f":d,{uvalue}",
            "Cu": f":d,{uvalue}"
        }

    elif uvalue < 0:
        replace_calc_params["setups"] = {
            "V": ":d,3.1",
            "Cr": ":d,3.5",
            "Mn": ":d,3.8",
            "Fe": ":d,4.0",
            "Co": ":d,3.3",
            "Ni": ":d,6.4",
            "Cu": ":d,4.0"
        }

    # Create a params file with modified calc params
    create_gs_params(folder, replace_calc_params)


###########################
def _find_structure_file(folder: str,
                         structure_file: str,
                         structure_file_pattern: str = '*structure*.json') -> str:
    """
    Locate the structure file within a folder using the given filename or pattern.

    Args:
        folder (str): Directory path to search for the structure file.
        structure_file (str): The structure file name to search for.
        structure_file_pattern (str): The structure file pattern to search for.

    Returns:
        str: Full path to the located structure file.

    Raises:
        FileNotFoundError: If no suitable structure file is found in the folder.
    """

    if os.path.isfile(os.path.join(folder, structure_file)):
        return os.path.join(folder, structure_file)

    structure_files = glob.glob(os.path.join(folder, structure_file_pattern))
    if structure_files:
        return structure_files[0]

    raise FileNotFoundError(f"Structure file not found in folder: {folder}")


###########################
def has_tm3d(folder: str, structure_file: str = "structure.json") -> bool:
    """
    Check if the structure contains TM3d elements.
    Note: If the structure_file is missing, we seach for *structure*.json pattern

    Args:
        folder (str): Directory path containing the structure file(s).
        structure_file (str, optional): Name of the structure file to search for.
            Defaults to "structure.json".

    Returns:
        bool: True if TM3d elements are present in the structure, otherwise False.

    Raises:
        FileNotFoundError: If no suitable structure file is found in the folder.
    """
    TM3d_atoms = ['V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu']

    # tries to find structure_file, if missing looks for *structure*.json pattern
    file_path = _find_structure_file(folder, structure_file)
    atoms = read(file_path)

    # This line is to ensure mypy that an atoms always has symbols
    assert hasattr(atoms, 'symbols'), "Expected 'symbols' attribute in atoms"

    return any(atom in atoms.symbols for atom in TM3d_atoms)


###########################
def is_magnetic(mlfolder: str, c2db=None) -> (Any):
    """
    Checks if monolayer is magnetic.
    If you use this function with a bilayer folder, it needs gs.gpw file.

    - If gs.gpw exists in the mlfolder, checks magmoms.
      Note1: if all magmoms are less than 0.01 we consider the materials NM.
      Note2: if you want to use only with gs.gpw, you don't need to provide c2db.

    - If c2db is provided in the input, checks if the monolayer folder name exists as a
      uid in c2db. Note: if the uid of the monolayer changes in c2db this is not useful

    - If it can neither find a gs.gpw nor find monolayer on c2db return None.
    """
    # checks if the inserted mlfolder ends with '/', and removes it
    mlfolder = mlfolder.rstrip('/')

    gs_path = os.path.join(mlfolder, 'gs.gpw')

    if os.path.isfile(gs_path):
        magmoms = read(gs_path).get_magnetic_moments()
        return any(abs(m) >= 0.01 for m in magmoms)

    if c2db:
        monolayer_uid = os.path.basename(mlfolder)

        for row in c2db.select(f"uid={monolayer_uid}"):
            return row.magstate != 'NM'

    return None


###########################
def verify_monolayer(mlfolder: str,
                     criteria: Dict[str, Any] = {},
                     c2db=None,
                     structure_file: str = "structure_initial.json",
                     exceptions: List[str] = []) -> str:
    """
    Check if monolayer passes selection criteria.

    Args:
        exceptions (List[str]): list of uids that we accept regardless of the criteria
    """

    structure_path = os.path.join(mlfolder, structure_file)
    atoms = read(structure_path)

    # If we want to add some monolayers ignoring the criteria we let them here
    monolayer_uid = os.path.basename(mlfolder.rstrip('/'))
    if monolayer_uid in exceptions:
        return str(mlfolder)

    # Number of atoms condition
    if 'max_natom' in criteria and len(atoms) > criteria['max_natom']:
        print(f'Monolayer NOT accepted: atomnum > {criteria["max_natom"]}', mlfolder)
        return ''

    # Conditions applied if the matterial has to be in c2db and have certain properties
    c2db_criteria = criteria.get('c2db', {})
    if c2db_criteria.get('in_c2db'):

        if c2db is None:
            raise ValueError('Not connected to c2db')

        exists_in_c2db = False
        for c2db_row in c2db.select(f"uid={monolayer_uid}"):
            exists_in_c2db = True

            # dynamic_stability_phonons of a c2db_row can be 'high' or 'low'
            if c2db_criteria.get('dynamic_stability_phonons', '').lower() == 'high':
                if c2db_row.get('dynamic_stability_phonons', 'low').lower() != 'high':
                    print(f'Monolayer NOT accepted: Dyn Stab phonons > {mlfolder}')
                    return ''

            # dynamic_stability_stiffness of a c2db_row can be 'high' or 'low'
            if c2db_criteria.get('dynamic_stability_stiffness', '').lower() == 'high':
                if c2db_row.get('dynamic_stability_stiffness', 'low').lower() != 'high':
                    print(f'Monolayer NOT accepted: Dyn Stab stiffness > {mlfolder}')
                    return ''

            # thermodynamic_stability_level of a c2db_row can be 1, 2, 3
            thermo_level = c2db_criteria.get('thermodynamic_stability_level', 0)
            if c2db_row.get('thermodynamic_stability_leve', 0) < thermo_level:
                print(f'Monolayer NOT accepted: Thermodyn Stab level > {mlfolder}')
                return ''

            # Apart from thermodynamic_stability we might want check ehull value
            if c2db_criteria.get('ehull'):
                if c2db_row.get('ehull', 1000) > c2db_criteria['ehull']:
                    print(f'Monolayer NOT accepted: ehull > {c2db_criteria["ehull"]}: ',
                          mlfolder)
                    return ''

        if not exists_in_c2db:
            print(f'Monolayer NOT accepted: Not found in the c2db > {mlfolder}')
            return ''

    if criteria.get('has_tm3d') and not has_tm3d(mlfolder):
        print(f'Monolayer NOT accepted: has_tm3d = {has_tm3d(mlfolder)} > {mlfolder}')
        return ''

    return str(mlfolder)


###########################
def starting_magmoms(structure_path: str, hundrule: bool = False) -> List[float]:
    """
    Creates a list of magnetic momemnts for initializing a calculation.
    - If hundrule = True: uses the magmoms for each element from a dictionary.
    - If hundrule = True but the structure does not have TM3d elements return []
    - If hundrule = Flase: reads the magmoms from the structure file provided
    - If hundrule = False but strcuture does not have magmoms returns []
    - If hundrule = False but structure does not have calc on it, raise ValueError.

    Raises:
        ValueError: if hundrule is not used and the atoms don't have clac on it.
    """
    # values used in case of hundrule
    TM3d = {'V': 1, 'Cr': 3, 'Mn': 5, 'Fe': 4, 'Co': 3, 'Ni': 2, 'Cu': 1}

    atoms = read(structure_path)

    if hundrule:
        magmoms = [TM3d[atom.symbol] if atom.symbol in TM3d else 0.0 for atom in atoms]
        return magmoms if any(m > 0.01 for m in magmoms) else []

    elif not hundrule:

        if not atoms.calc:
            raise ValueError('Structure does not have calc to access magmoms: ',
                             structure_path)

        elif 'magmoms' in read_json(structure_path)[1]:  # !!!
            return atoms.get_magnetic_moments()

        else:
            return []


###########################
def gs_subfolders(top_folder_path: str,
                  subfolder_noU: str,
                  subfolder_U: str,
                  structure_path: str,
                  initial_magmoms: List[float],
                  hubbardu: float,
                  e_threshold: float = -1.0,
                  mixer={}) -> Tuple[str, str]:
    """
    Creates subfolders for gs calculations with/without Hubbard U.
      - Skips the subfolder if the name is ''.
      - Puts a structure file with given initial_magmoms inside each folder.
      - Puts a params file with suitable calculator parameters inside each folder.

    The default of e_threshold<0 and mixer={} will leave the params unchanged.

    Returns:
        The address of the created subfolders.
    """
    subfolder_noU = f'{top_folder_path}/{subfolder_noU}' if subfolder_noU else ''
    subfolder_U = f'{top_folder_path}/{subfolder_U}' if subfolder_U else ''

    for subfolder, uval in zip([subfolder_noU, subfolder_U], [0, hubbardu]):
        if subfolder:
            # We want to do this only once, if the subfolders exist we don't repeat
            if not os.path.isdir(subfolder):
                os.mkdir(subfolder)

                # Creating the structure file with given inital magnetic moments
                atoms = read(f'{top_folder_path}/{structure_path}')

                if initial_magmoms:
                    atoms.set_initial_magnetic_moments(initial_magmoms)
                atoms.write(f'{subfolder}/structure.json')

                # Creating a params file to:
                #  (1) Set the gs convergenve energy
                #  (2) Set the appropriate hubbardU correction
                #  (3) Set the mixer
                modify_params(subfolder,
                              etot_threshold=e_threshold,
                              uvalue=uval,
                              mixer=mixer)

        return subfolder_noU, subfolder_U


###########################
def analyse_w_wo_U(tm3d: bool,
                   mlfolder: str,
                   subfolder_noU: str,
                   subfolder_U: str) -> None:
    """
    - Checks if we need +U for bilayers based on:
        - Monolayer having TM3d elmenets or not.
        - If monolayer has TM3d, does it have a gap with U.
    - Copies the results of the chosen gs in the main mlfolder.
    - Removes the gs.gpw files from the subfolders.

    Note: we don't raise errors if the files don't exist because this function is
        called repeatedly by the workflow and acts when it is ready.
    """
    # If there is a gs.gpw in the main folder, this analysis is done.
    if os.path.isfile(f'{mlfolder}/gs.gpw'):
        return

    # If tm3d, we need gs results withU be done inorder to check if it has a gap
    if tm3d and not os.path.isfile(f'{subfolder_U}/results-asr.gs.json'):
        return

    # If the structure does not have tm3d or no gap with U, U is not needed.
    chosen_gs = subfolder_U if tm3d and hubbardu_needed(subfolder_U) else subfolder_noU

    # If the chosen_gs is withoutU then we need the gs calculation for it to be done
    if chosen_gs == subfolder_noU:
        if not os.path.isfile(f'{subfolder_noU}/results-asr.gs.json'):
            return

    # Copy the gs results of the chosen_gs in the main mlfolder
    copy_gs_results(origin=chosen_gs, destination=mlfolder)

    # Clean up subfolders if gs.gpw of the chosen_gs has been copied to the mlfolder
    silentremove(f'{subfolder_noU}/gs.gpw', check_other_file=f'{mlfolder}/gs.gpw')
    if subfolder_U:
        silentremove(f'{subfolder_U}/gs.gpw', check_other_file=f'{mlfolder}/gs.gpw')


###########################
def copy_gs_results(origin: str, destination: str) -> None:
    """Copies the result files of gs calculation from an origin to a destination."""

    # Checks if a gs.gpw file exists is the origin to avoid repeating after cleanup.
    if os.path.isfile(f"{origin}/gs.gpw"):

        # We don't copy structure file because we want the calculator on the structure
        new_structure = read(f'{origin}/gs.gpw')
        new_structure.write(f'{destination}/structure.json')

        files_to_copy = ["results-asr.gs.json", "gs.txt", "gs.gpw",
                         "results-asr.structureinfo.json", "params.json",
                         "results-asr.magnetic_anisotropy.json",
                         "results-asr.magstate.json", "vdw_e.npy"]

        for filename in files_to_copy:
            shutil.copy(f"{origin}/{filename}", f"{destination}/{filename}")

        silentremove(f'{origin}/gs.gpw', check_other_file=f'{destination}/gs.gpw')


###########################
def hubbardu_needed(folder: str) -> bool:
    """
    Checks if monolayer has a gap with Hubbard U correciton.
    - We set a 100 meV lower limit for gap so that the screening is not much

    Raises:
        FileNotFoundError: if gs result file not found in folder
        FileNotFoundError: if params file not found in the folder
        ValueError: if params file exists but setups is empty
    """
    if not os.path.isfile(f'{folder}/results-asr.gs.json'):
        raise FileNotFoundError(f'gs results not found in: {folder}')

    # We apply Hubbard U with a params file: if it does not exist U is not applied
    params_file = f'{folder}/params.json'
    if not os.path.isfile(params_file):
        raise FileNotFoundError(f'params file not found in: {folder}')

    # "Setups" in the params file should not be empty to ensure the gs is done with U.
    elif not read_json(params_file)["asr.gs@calculate"]["calculator"].get("setups"):
        raise ValueError(f'Hubbard U is not applied in the calculation: {folder}')

    monolayer_gap_withU = read_json(f"{folder}/results-asr.gs.json").get("gap")

    return monolayer_gap_withU > 0.100


###########################
def is_zscan_done(folder: str,
                  result_filename: str = 'results-asr.zscan.json') -> bool:
    """ Checks if the result file of the zscan recipe exists"""
    zscan_result = os.path.join(folder, result_filename)
    return os.path.isfile(zscan_result)


def is_zscan_running(folder: str, 
                     time_in_seconds: int = 1200,
                     text_filename:str = 'zscan.txt') -> bool:
    """
    Check if 'text_file' file has been updated inside the folder in the past
    "time_in_seconds".
    """
    zscan_txt = os.path.join(folder, text_filename)
    return os.path.isfile(zscan_txt) and \
        recently_updated(folder, text_filename, time_in_seconds=1200)


def is_zscan_attempted(folder: str, 
                       text_filename: str = "zscan.txt",
                       result_filename: str = 'results-asr.zscan.json') -> bool:
    """
    Checks if the zscan recipe has ((started or finished) and (is not running)):
      - started: 'text_file' is created in the folder
      - finished: 'result_file' is created in the folder
      - is not running now: 'text_file' has not updated recently.
    """
    zscan_txt = os.path.join(folder, text_filename)
    return (os.path.isfile(zscan_txt) or is_zscan_done(folder, result_filename)) and \
        not is_zscan_running(folder, text_filename=text_filename)


def is_zscan_valid(blfolder: str,
                   result_filename: str = 'results-asr.zscan.json') -> bool:
    """Checks if the zscan calculation of a bilayers is valid"""
    from asr.zscan import min_quality

    zscan_resultfile = f'{blfolder}/{result_filename}'
    if os.path.isfile(zscan_resultfile):
        data = read_json(zscan_resultfile)
        hh = data['heights']
        ee = data['energies']
        if not min_quality(energies=ee, heights=hh):
            print(f'>>> Problem in zscan: {blfolder}')
            return False
        else:
            return True
    else:
        return False


def many_invalid_zscans(blfolders: List[str],
                        ignore_failed_percentage: float = 0.0) -> bool:
    """
    Reports if some zscans are failed or invalid and one of the following
      - not all the bilayers have attempted zscan
      - the percentage of the failed cases is more than limit
    """
    valid_bilayers = [bl for bl in blfolders if is_zscan_valid(bl)]
    all_zscans_valid = len(valid_bilayers) == len(blfolders)

    if not all_zscans_valid:

        if ignore_failed_percentage < 0.01:
            print('Not all zscan calculations done: ')
            print(*list(set(blfolders) - set(valid_bilayers)), sep='\n')
            return True

        else:
            zscan_attempted = [bl for bl in blfolders if is_zscan_attempted(bl)]
            zscan_not_valid = [bl for bl in blfolders if not is_zscan_valid(bl)]

            if len(blfolders) != len(zscan_attempted):
                print('Not all bilayers attempted zscan: ')
                print(*list(set(blfolders) - set(zscan_attempted)), sep='\n')
                return True

            if len(blfolders) * ignore_failed_percentage <= len(zscan_not_valid):
                print('Too many bilayers failed "zscan": ')
                print(*zscan_not_valid, sep='\n')
                return True

    return False


###########################
def _read_gs_etot(folder: str) -> float:
    """ Reads ground state total energy if the gs result file exists in the folder"""
    file_path = os.path.join(folder, 'results-asr.gs.json')

    if os.path.isfile(file_path):
        return read_json(file_path)['etot']
    else:
        raise FileNotFoundError(f"gs result file not found in '{folder}'")


def _read_vdw_corr(folder: str, filename: str = 'vdw_e.npy') -> float:
    """ Reads the vdw energy correction if vdw_e.npy file exists in the folder"""
    file_path = os.path.join(folder, filename)

    if os.path.isfile(file_path):
        return np.load(file_path)
    else:
        raise FileNotFoundError(f"file '{filename}' not found in '{folder}'")


def get_vdw_corrected_etot(folder: str) -> float:
    """ Calculates vdw corrected energy from PBE energy and D3 correction"""
    return _read_gs_etot(folder) + _read_vdw_corr(folder)


def get_eb_interlayer(blfolder: str,
                      hubbardU: float,
                      source: str,
                      tm3d: bool,
                      mlfolder: str = '') -> Tuple[Any, Any]:
    """
    Returns the binding energy and interlayer distance.
    Note: If any file is missing, return None.

    Args:
       blfolder (str): Bilayer folder. Should be in the project's folder structure
           because it needs to access several files in paths relative to the blfolder.
       hubbardU (float): If the source of the calculation is zscan, hubbardU is ignored.
       source (str): 'zscan' or 'gs', 'gs' is more accurate but we have 'zscan' for all
       tm3d (bool): True if the structure has a TM3D element.
       mlfolder (str): if not provided, the top folder of blfolder is used as default.

    Returns (If the necessary data is not available returns None.)
        binding energy (float/None): the unit eV/A2 and positive.
        interlayer distance (float/None): distance between closest atoms.
    """

    if not mlfolder:
        mlfolder = f"{blfolder}/../"

    cell = read(f"{mlfolder}/structure.json").cell
    cellarea = np.linalg.norm(np.cross(cell[0], cell[1]))

    zscan_result = f"{blfolder}/results-asr.zscan.json"
    zscan_data = read_json(zscan_result) if os.path.isfile(zscan_result) else {}
    zscan_eb = zscan_data.get("zscan_binding_energy", None)
    zscan_il = zscan_data.get("interlayer_distance", None)

    if source == 'zscan':
        return zscan_eb, zscan_il

    elif source == 'gs':
        try:
            if not tm3d or (tm3d and abs(hubbardU) < 0.01):
                ebl = get_vdw_corrected_etot(f"{blfolder}/gs-withoutU/")
                eml = get_vdw_corrected_etot(f"{mlfolder}/gs-withoutU/")

            elif tm3d and abs(hubbardU) >= 0.01:
                ebl = get_vdw_corrected_etot(f"{blfolder}/gs-withU/")
                eml = get_vdw_corrected_etot(f"{mlfolder}/gs-withU/")

            eb = -1000 * (ebl - 2 * eml) / cellarea
            return eb, zscan_il

        except OSError:
            # return a string rather than None because we can not collect None in the database
            return '-', zscan_il

    else:
        raise ValueError(f"Binding energy source should be 'zscan' or 'gs'.")


###########################
def select_largest_ebs(blfolders: List[str],
                       tm3d: bool,
                       deltaE: float,
                       hubbardU: bool = False,
                       source: str = 'gs',
                       ignore_failed: bool = False,
                       eb_cutoff: float = 1000.0) -> List[str]:
    """
    Selects the bilayers with Ebs within a window of deltaE below Eb_max.

    Args:
        - deltaE (float): The energy range in meV/A2 to keep bilayers below Eb_max.

        - source (str): can be 'gs' or 'zscan':
           - 'zscan' we calculate it for all bilayers but convergence less strict.
           - 'gs' is better converged but avilable for 10meV/A2 below Eb_max.

        - ignore_failed (bool):
           - If True: ignores the fialed bilayers.
           - If False: even if one bilayer fails, returns no bilayer for the monolayer.
           - Note: Keep False until you try all possible ways to converge zscan.

        - eb_cutoff (float): unit meV/A2
           If Eb_max is larger than eb_cutoff, remove the monolayer as unexfoliable
           The default is large (1000) so it does not remove any monolayer.

    Returns:
        A list of selected bilayer folder paths. The list is sorted with decreasing Ebs
        All bilayers must have attempted zscan once otherwise it returns []
    """

    def is_valid_result(eb):
        return eb != '-'

    def is_unexfoliable(eb_max, eb_cutoff):
        return eb_max > eb_cutoff

    energies = []
    for subf in blfolders:

        if not is_zscan_attempted(subf):
            return []

        if not is_zscan_done(subf) and ignore_failed:
            continue

        if not is_zscan_done(subf) and not ignore_failed:
            return []

        eb, length = get_eb_interlayer(subf, hubbardU, source, tm3d)

        if not is_valid_result(eb) and ignore_failed:
            continue

        if not is_valid_result(eb) and not ignore_failed:
            return []

        energies.append((eb, subf))

    if not energies:
        monolayer_uid = blfolders[0].split('/')[-2]
        print(f'>>> The Ebs of no bilayers found for: {monolayer_uid}')
        return []

    eb_max = max([e for e, s in energies])

    if is_unexfoliable(eb_max, eb_cutoff):
        return []

    # Select materials within window of Eb_max
    selected = [(e, s) for e, s in energies if abs(e - eb_max) <= deltaE]

    selected = sorted(selected, key=lambda t: -t[0])[:]
    return [bl[1] for bl in selected]


###########################
def stability_status(bl: str) -> str:
    """
    Checks the slide stability of a bilayer given the bilayer folder address.
    - If the dynstab calculation is done, checks the results-asr.dynstab.json file.
    - If the calculations are not done, checks dynstab.json file.

    Note: In general using dynstab.json file is always possible and it holds the latest
        info. However, since it might be removed from the folders during collecting and
        recreating the tree from the database file, we check for the result file of the
        recipe first. This keeps the function more general.
    """
    # This result file is created when at least 7 of the slided points has gs results.
    result_file1 = f"{bl}/results-asr.dynstab.json"

    # This file is created during workflow submission after slided folders are created.
    result_file2 = f"{bl}/dynstab.json"

    if os.path.isfile(result_file1):
        return read_json(result_file1)["status"]

    elif os.path.isfile(result_file2):
        return read_json(result_file2)["DynStab_Result"]

    else:
        return ''


def select_stables(bilayer_folders: List[str]) -> List[str]:
    """Collects a list of slide stable bilayers from a list of bilayers"""
    return [bl for bl in bilayer_folders if 'Stable' in stability_status(bl)]


###########################
def create_fullrelax_folder(top_folder: str,
                            subfolder_name: str = 'fullrelax') -> None:
    """Creates a subfolder for full relaxation. Puts a 'unrelaxed.json' file in it."""
    fullrelax_folder = f"{top_folder}/{subfolder_name}"

    if not os.path.isdir(fullrelax_folder):
        os.mkdir(fullrelax_folder)

    if not os.path.isfile(f'{fullrelax_folder}/unrelaxed.json'):
        atoms = read(f"{top_folder}/structure.json")
        atoms.write(f'{fullrelax_folder}/unrelaxed.json')
