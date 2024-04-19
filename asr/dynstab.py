from asr.core import command, option, AtomsFile, ASRResult, prepare_result, DictStr
from ase import Atoms
from ase.io import read, write
from asr.core import read_json, write_json
import os
from pathlib import Path
from asr.interlayer_magnetic_exchange import CheckMagmoms
from asr.stack_bilayer import LatticeSymmetries, BilayerMatcher, BuildBilayer
import json
import numpy as np
import shutil
from asr.workflow.bilayerutils import listdirs, task, recently_updated, set_info


##############################################
##### Generating slide stability folders #####
##############################################

def generate_slided_folders(blfolder, start_structure, folders_path, step_size=0.05, step_num=2):

    if os.path.isdir(folders_path):    
        return
    else:
        os.mkdir(folders_path)

    dvec = step_size*step_num

    offsets = {'original':[0, 0, 0], 
               'yp':[0, +dvec, 0], 
               'ym':[0, -dvec, 0], 
               'xp':[+dvec, 0, 0], 
               'xm':[-dvec, 0, 0], 
               'xpyp':[+dvec, +dvec, 0], 
               'xpym':[+dvec, -dvec, 0], 
               'xmyp':[-dvec, +dvec, 0], 
               'xmym':[-dvec, -dvec, 0]}

    for key, shift_vector in offsets.items():
        if 'original' not in key:
            create_slided_folder(blfolder, start_structure, shift_vector, folders_path, f'StabCheck_{key}_{step_num}')
        else:
            create_slided_folder(blfolder, start_structure, shift_vector, folders_path, f'StabCheck_{key}')


# Note: the initial magmom is set on bilayerprototye in the stacking recipe
def create_slided_folder(blfolder, start_structure, vector, path, subfolder_name):
    slided_bilayer = shift_top_layer(blfolder, start_structure, vector)

    if not os.path.isdir(f"{path}/{subfolder_name}"):
       os.mkdir(f"{path}/{subfolder_name}")
       slided_bilayer.write(f'{path}/{subfolder_name}/structure.json')


# Note: bilayer atoms are tagged in the zscan recipe and we use them
def shift_top_layer(blfolder, start_structure, vector):
    ''' 
    Slide stability should be done before the interlayer magnetic recipe 
    But we make sure the magnetic state is FM for slide stability because 
    interlayer magnetic recipe over writes the magmoms of the structure file
    '''
    bilayer = read(f"{blfolder}/{start_structure}")

    magmoms = read(f"{blfolder}/bilayerprototype.json").get_initial_magnetic_moments()

    atoms_new = bilayer.copy()
    atoms_new.set_initial_magnetic_moments(magmoms)
    tags = atoms_new.get_tags()
    for ii, _ in enumerate(atoms_new):
        if tags[ii]>0:
           atoms_new.positions[ii,:] += vector

    return  atoms_new
   

##############################################
##### Submit the gs claculations #############
##############################################

@prepare_result
class GSResult(ASRResult):
    edft: float
    vdw_corr: float
    etot: float

    key_descriptions = dict(
        edft = 'DFT total energy [eV]',
        vdw_corr = 'D3 correction energy [eV]',
        etot = 'DFT+D3 total energy [eV]')
  
@command(module='asr.dynstab')

@option('-f', '--folder', help='folder path to run the calculations inside', type=str)
@option('-s', '--settings', help='Calculator settings', type=DictStr())
@option('-d3', '--dftd3', help='add D3 correction', type=bool)
def gs_minimal(folder = '.', 
              settings = {}, 
              d3=True) -> GSResult:

    """
    I want to call and run this by the workflow and submit them as separate tasks
    This will make the troubleshooting easier if some of the calculations fail
    """
    from asr.calculators import get_calculator
    from ase.calculators.calculator import get_calculator_class

    settings: dict = {
        'name': 'gpaw',
        'mode': {'name': 'pw', 'ecut': 800},
        'xc': 'PBE',
        'basis': 'dzp',
        'kpts': {'density': 12.0, 'gamma': True},
        'occupations': {'name': 'fermi-dirac',
                        'width': 0.05},
        'convergence': {'bands': 'CBM+3.0', 
                        'density': 0.000001,
                        'energy': {"name": "energy", 
                                   "tol": 1e-5, 
                                   "relative": False,  
                                   "n_old": 3} },
        'mixer': {'method': 'sum',
                  'beta': 0.03,
                  'history': 5,
                  'weight': 50},
        'nbands': '200%',
        'charge': 0,
        'symmetry': {},
        'maxiter': 5000,
        'poissonsolver': {'dipolelayer': 'xy'},
        'txt': f'{folder}/gs_minimal.txt',
        **(settings or {})}

    name = settings.pop('name')

    if 'original' in os.path.abspath(folder):
        settings['symmetry'] = "off"

    if not os.path.isfile(f'{folder}/results-asr.gs_minimal.json'):
        atoms = read(f'{folder}/structure.json')

        calc_dft = get_calculator_class(name)(**settings)

        atoms.calc = calc_dft
        atoms.get_forces()
        edft = atoms.get_potential_energy()
        atoms.calc.write(f'{folder}/gs_minimal.gpw')
        atoms.write(f'{folder}/structure.json')

        if d3: 
            D3Calculator = get_calculator_class("dftd3")
            calc_d3 = D3Calculator()
            atoms = read(f'{folder}/structure.json')
            atoms.calc = calc_d3
            vdw_corr = atoms.get_potential_energy()
        else:
            vdw_corr = 0

    return GSResult.fromdata(edft = edft,
                             vdw_corr = vdw_corr,
                             etot = edft + vdw_corr)


##############################################
##### Collect the gs claculations ############
##############################################

def collect_energies(path,
                     step_size=0.05, 
                     result_file = 'results-asr.dynstab@gs_minimal.json'):

    etots, xs, ys, magmoms = [], [], [], []

    # I just want the middle point to come first in the report
    folders = listdirs(path)
    for folder in folders:
        if 'StabCheck_original' in folder:
            folders.remove(folder)
            folders.insert(0, folder)

            if not os.path.isfile(f'{folder}/{result_file}'):       
                #print('The middle strucutre energy not converged')
                return etots, xs, ys, magmoms

    for folder in folders:
        if os.path.isfile(f'{folder}/{result_file}'):
            if check_magmoms(folder):
                data = read_json(f'{folder}/{result_file}')
                etots.append(data.etot)
                xnew, ynew = toplayer_pos(folder.split('/')[-1], step_size=step_size)
                xs.append(xnew)
                ys.append(ynew)
                magmoms.append(read(f'{folder}/structure.json').get_magnetic_moment())
                #print("%%%%%%%%%%%%%%%%%%%%", xnew, ynew, data.etot, read(f'{folder}/structure.json').get_magnetic_moment())
    return etots, xs, ys, magmoms


def toplayer_pos(folder, step_size=0.05):
    """this function finds the position of the top layer from the folder name"""
    if 'original' not in folder:
        step_num = int(folder.split('_')[-1])

    if 'xm' in folder:
       xnew = -step_size*step_num
    elif 'xp' in folder:
       xnew = +step_size*step_num
    else:
       xnew = 0.0
    if 'ym' in folder:
       ynew = -step_size*step_num
    elif 'yp' in folder:
       ynew = +step_size*step_num
    else:
       ynew = 0.0

    return xnew, ynew


def check_magmoms(folder=None, atoms=None):
    """Checks the magnetic moments are still valid after the calculation"""
    if atoms is None:
        # the structure file in the slide stability folder have the calculator info over written on them
        atoms = read(f'{folder}/structure.json')

    bilayer_magmom = atoms.get_magnetic_moment()
    bilayer_magmoms = atoms.get_magnetic_moments()
    bl_initial_magmoms = atoms.get_initial_magnetic_moments()

    if abs(max(bilayer_magmoms))>0.05:
       checkMagmoms = CheckMagmoms(bilayer=atoms.copy(),
                                    state='FM',
                                    initial_magmoms=bl_initial_magmoms,
                                    magmoms=bilayer_magmoms,
                                    magmom=bilayer_magmom)
       return checkMagmoms.magmstate_healthy()
    else:
       return True


##############################################
##### Checking the slide stablity ############
##############################################

def stability_analysis(blfolder, total_energies, xdisp, ydisp, magmoms):
    R2 = 0
    min_positions = []
    report_params = None

    unfinished_jobs = 9-len(total_energies)
    if len(total_energies)<7:
        result_curvature = "unfinished"        
        set_info(blfolder, key="DynStab_Result", value=result_curvature, filename="dynstab.json")
        return result_curvature

    if unfinished_jobs<3:
        ZZ = list((np.array(total_energies)-total_energies[0])*1000)
        if np.average(np.abs(ZZ))<0.1:
            result_curvature = "Stable-flat"
            R2 = 1
        else:
            result_curvature, params, [xnew, ynew, Enew], R2 = fit_parabola(xdisp,ydisp,ZZ)
            min_positions = [xnew, ynew, Enew]
            report_params = params.x.tolist() 

    Data={"X":xdisp, 
          "Y":ydisp, 
          "E":ZZ, 
          "Magmoms":magmoms}

    Fit={"Function": "(a*x)+(b*y)+(0.5*c*(x**2))+(0.5*d*(y**2))+e*x*y+f", 
         "FitParam": report_params, 
         "Fit_Quality (R2)":R2, 
         "CurveMin":min_positions}

    set_info(blfolder, key="DynStab_Result", value=result_curvature, filename="dynstab.json")
    set_info(blfolder, key="DynStab_Data", value=Data, filename="dynstab.json")
    set_info(blfolder, key="DynStab_Fit", value=Fit, filename="dynstab.json")

    return result_curvature


def eparabola(X,a,b,c,d,e,f): # elliptic paraboloid
    '''parabolic fit function'''
    x, y = X
    x = np.array(x)
    y = np.array(y)
    return (a*x)+(b*y)+(0.5*c*(x**2))+(0.5*d*(y**2))+e*x*y+f


def fit_quality(XX,YY,ZZ, res):
    residuals = ZZ- eparabola((XX, YY), *res)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ZZ-np.mean(ZZ))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def fit_parabola(XX, YY, ZZ):
    import scipy
    from scipy.optimize import curve_fit, least_squares

    # The fit was very sensitive to initial guess so we start with different ones and use the best fit
    rangemin = [100,-100,100,-100,10,-10,10,-10,10,-10,10,-10,100,-100,100,-100,0]
    rangemax = [100,100,-100,-100,10,10,-10,-10,100,100,-100,-100,10,10,-10,-10,0]

    fits = []
    values = []
    for ii in range(len(rangemin)):
        fits.append(least_squares(lambda args: (eparabola((XX,YY), *args)-ZZ).ravel(), (0, 0, rangemin[ii], rangemax[ii], 0, 0),jac='3-point',ftol=1e-13, xtol=1e-13, gtol=1e-13))
        # find the best quality fit and continue with that.
        values.append(fit_quality(XX, YY, ZZ, fits[ii].x))

    goodfit_index =values.index(max(values))
    p_opt3 = fits[goodfit_index]
    R2= fit_quality(XX, YY, ZZ, p_opt3.x)

    # here I diagonalize the matrix to find the curvature in the correct direction.
    #print('Fit: a={}, b={}, c={}, d={}, e={}, f={}'.format(*p_opt3.x))
    aa, bb, cc, dd, ee, ff = p_opt3.x
    mat = [[cc, ee],[ee, dd]]
    w, v = np.linalg.eig(mat)

    # here I find the position of the min of the fitted function, 
    from scipy.optimize import minimize
    xmin = minimize(eparabola, args=(aa, bb, cc, dd, ee, ff) , x0=[0,0])
    xnew = xmin.x[0]
    ynew = xmin.x[1]
    Enew = eparabola([xnew, ynew], aa, bb, cc, dd, ee, ff)

    # there are some thresholds here that I had to decide when I call the linear part of the fit small and when the minimun of the function is not far from our stacking.
    if R2<0.8:
       res = "Badfit"
    elif (w[0]>=0 and w[1]>=0) and np.abs(aa)<0.2*np.abs(cc) and np.abs(bb)<0.2*np.abs(dd) and np.abs(xnew)<=0.03 and np.abs(ynew)<=0.03:
       res = "Stable"
    elif w[0]<0 and w[1]<0:
       res = 'Unstable-Max'
    elif (w[0]<0 or w[1]<0) or np.abs(aa)>0.2*np.abs(cc):
       res = "Unstable-Saddle"
    elif np.abs(bb)>0.2*np.abs(dd) or np.abs(xnew)>0.03 or np.abs(ynew)>0.03:
       res = "Unstable-shifted"
    else:
       res = "unfinished"

    return res,  p_opt3, [xnew, ynew, Enew], R2


##############################################
##### Constrained relaxation task ############
##############################################

@prepare_result
class RelaxResult(ASRResult):
    atoms: Atoms

    key_descriptions = dict(
        atoms = 'atoms after constrained relaxation of the top layer')

@command(module='asr.dynstab')

@option('-d3', '--dftd3', help='add D3 correction', type=bool)
@option('-st', '--start-structure', help='Name of the structure file to apply the relaxation on', type=str)

def constrained_relax(d3: bool = True,
                      start_structure: str = "structure_zscan.json") -> RelaxResult:
    from ase.optimize import BFGS
    from ase.constraints import FixAtoms
    from gpaw import GPAW, PW

    from ase.calculators.calculator import get_calculator_class
    from ase.calculators.dftd3 import DFTD3

    class RigidBodyConstraint:
        def adjust_positions(self, atoms, new):
            pass

        def adjust_forces(self, atoms, forces):
            n = len(atoms) // 2
            forces[n:] = forces[n:].mean(axis=0)

    data = read_json('dynstab.json')
    energies = data["DynStab_Data"]["E"]
    xdisp = data["DynStab_Data"]["X"]
    ydisp = data["DynStab_Data"]["Y"]

    start_structure = get_start_structure(blfolder='.',
                                          start_structure=start_structure, 
                                          energies=energies, 
                                          xdisp=xdisp, 
                                          ydisp=ydisp)

    # For convergence on forces: We checked 1e-4 and relaxation sometimes stop at the starting point
    settings: dict = {
        'name': 'gpaw',
        'mode': {'name': 'pw', 'ecut': 800},
        'xc': 'PBE',
        'basis': 'dzp',
        'kpts': {'density': 10.0, 'gamma': True},
        'occupations': {'name': 'fermi-dirac',
                        'width': 0.05},
        'convergence': {'forces': 1e-4,
                        'energy': {"name": "energy",
                                   "tol": 1e-5,
                                   "relative": False,
                                   "n_old": 3} },
        'mixer': {'method': 'sum',
                  'beta': 0.03,
                  'history': 5,
                  'weight': 50},
        'nbands': '200%',
        'charge': 0,
        'symmetry': {},
        'maxiter': 5000,
        'poissonsolver': {'dipolelayer': 'xy'},
        'txt': 'relax_constrained.txt'}

    name = settings.pop('name')
    calc_dft = get_calculator_class(name)(**settings)
    calc_d3 = DFTD3(dft=calc_dft)

    atoms = start_structure.copy()
    atoms.calc = calc_d3

    tags = atoms.get_tags()
    fixed_atom_indices = [iatom for iatom, tag in enumerate(tags) if tag ==0 ]
    atoms.constraints = [FixAtoms(indices=fixed_atom_indices),
                         RigidBodyConstraint()]

    BFGS(atoms).run(fmax=0.001)

    assert check_magmoms(atoms=atoms)

    relaxed_structure = start_structure.copy()
    relaxed_structure.positions = atoms.positions

    return RelaxResult.fromdata(atoms=relaxed_structure.copy())


def get_start_structure(blfolder, start_structure, energies, xdisp, ydisp):
    """ Start from the structure that slide stability suggest as lowest energy"""
    Emin = min(energies)
    index = energies.index(Emin)
    vector = [xdisp[index], ydisp[index], 0]
    return shift_top_layer(blfolder, start_structure, vector)


##############################################
##### Check if relaxed bilayer is new ########
##############################################

def unique_bilayer(blfolder, start_structure):
    """ Check if the generated bilayer is not the same as any of the other bilayers"""
    relaxed = read_json(f'{blfolder}/results-asr.dynstab@constrained_relax.json')["atoms"]

    new_atoms = zposition_adjust_to_prototype(atoms=relaxed, blfolder=blfolder) 

    # We compare II / IF / FI groups only between themselves
    topfolder = f'{blfolder}/../'
    bilayers_to_compare = [folder for folder in listdirs(topfolder) if '-2-' in folder.split('/')[-1] and folder.split('-')[0] == blfolder.split('-')[0]]

    # We compare this new bilayer with each of previously generated bilayers
    unique = True
    similar_to = ''
    for folder in bilayers_to_compare:

        bilayer_pair = [new_atoms, read(f'{folder}/bilayerprototype.json')]

        braveSym = LatticeSymmetries(mlatoms=read(f'{blfolder}/{start_structure}')).braveSym
 
        # The matcher_tol here is 0.1 and the steps of the dynstab are 0.15 this means it the relaxation stops at the first step it can be called new
        matcher = BilayerMatcher(atoms_list = bilayer_pair, 
                                 auxs_list = [[], []], # We can do this because we don't need the auxs in the output 
                                 transform_layer = BuildBilayer.transform_layer, 
                                 brave_symmetries = braveSym,
                                 matcher_tol=0.1)

        unique_bilayers, _ = matcher.unique_materials()

        if len(unique_bilayers)<2:
            unique = False
            similar_to  = folder.split('/')[-1]
            break
           
    return relaxed, unique, similar_to
     

def zposition_adjust_to_prototype(atoms, blfolder):
    stacked_bilayer = read(f'{blfolder}/bilayerprototype.json')
    relaxed_bilayer = atoms.copy()

    tags = stacked_bilayer.get_tags()
    for iatom, atom in enumerate(stacked_bilayer):
        if tags[iatom] == 1:
            zpos_stacked = atom.position[2]
            zpos_relaxed = relaxed_bilayer[iatom].position[2]
            break

    new_zpositions = relaxed_bilayer.positions
    for iatom, atom in enumerate(relaxed_bilayer):
        if tags[iatom] == 1:
            new_zpositions[iatom,2] += (zpos_stacked-zpos_relaxed)

    # we want the interlayer to be the same as the prototype for comparison
    relaxed_bilayer.set_positions(new_zpositions)

    # we want the vacuum of the new layer to be the same as the prototype
    relaxed_bilayer.cell[2] = stacked_bilayer.cell[2]  

    return relaxed_bilayer


##############################################
##### Creare new bilayers ####################
##############################################

def create_new_bilayer(bilayer, start_structure, relaxed_structure, is_new, similar_to):

    set_info(bilayer, key="Relaxed_is_new", value=is_new, filename="dynstab.json")

    if is_new:
       
        old_atoms = read(f'{bilayer}/{start_structure}')
        new_atoms = relaxed_structure.copy()
        vector = displacement(old_atoms, new_atoms)
        print(f'displacement vector: {vector}')
        ml = f'{bilayer}/../'
        tvec = read_json(f'{bilayer}/translation.json')["translation_vector"]
        translation_vector = list(np.array([tvec[0], tvec[1], 0]) + np.array(vector))[0]
        transform_rotation = read_json(f'{bilayer}/transformdata.json')["rotation"]
        transform_translation = read_json(f'{bilayer}/transformdata.json')["translation"]        

        # We have to generate the name of the new bilayer folder
        t = transform_translation + \
            old_atoms.cell.scaled_positions(np.array([translation_vector[0], translation_vector[1], 0.0]))
   
        # For consistency we use the function in the stacking recipe
        build_bilayer = BuildBilayer(mlatoms = read(f'{ml}/structure.json'), 
                                     mlfolder = ml, 
                                     spglib_tol = 0.1, 
                                     remove_unphysical = True, 
                                     vacuum = 15, 
                                     matcher_tol = 0.3, 
                                     use_rmsd=False, 
				     prefix = (bilayer.split('/')[-1]).split('-')[0])  #+'-'

        name = build_bilayer.bilayer_folder_name(formula = os.path.abspath(ml).split('/')[-1].split('-')[0], 
                                                 nlayers = 2, 
                                                 U_cc = transform_rotation, 
                                                 t_c = t)

        # We want to write in the original translation file that this is shifted to another bilayer
        set_info(folder = bilayer,
                 key = 'Shifted_to', 
                 value = name, 
                 filename=f'translation.json')

        # Now we want to create the new folder and the needed files for zscan
        if not os.path.isdir(f'{ml}/{name}'):
            os.mkdir(f'{ml}/{name}')

            # we don't write the relaxed structure because we want the structure to be created by zscan 
            translation_vector[2] = 0
            dct = {'translation_vector': translation_vector, 'Shifted_from': bilayer.split('/')[-1]}
            write_json(f'{ml}/{name}/translation.json', dct)

            # We just copy transformdata file because shifting does not affect that
            shutil.copy2(f'{bilayer}/transformdata.json', f'{ml}/{name}/transformdata.json')            
       
            # We also copy info.json for now
            shutil.copy2(f'{bilayer}/info.json', f'{ml}/{name}/info.json')
 
            # For the prototype we want the interlayer to be what used for all prototypes
            bilayer_prototype = zposition_adjust_to_prototype(new_atoms, bilayer)
            bilayer_prototype.write(f'{ml}/{name}/bilayerprototype.json')
    
    else:
        # We want to write in the original translation file that the relaxation did not end in a new bilayer
        set_info(folder = bilayer,
                 key = 'Shifted_to',
                 value = f'Not New (Similar to: {similar_to})',
                 filename=f'translation.json')


def displacement(old, new):
    from ase.geometry import get_distances

    tags = list(old.get_tags())
    index = tags.index(1)

    _temp = Atoms('C2', positions=[old.positions[index,:], new.positions[index,:]])
    _temp.set_cell(old.cell)
    pos = _temp.positions
    d1, d2= get_distances([pos[0]],[pos[1]], cell=_temp.cell, pbc=[True, True, False])
    return  d1[0]


##############################################
##### Save slide stablity results ############
##############################################

@prepare_result
class DynStabResult(ASRResult):
    origin: str
    status: str
    slide_data: dict 
    fit_data: dict
    shifted_from: str
    shifted_to: str

    key_descriptions = dict(
        origin = 'Origin of the bilayer from stacking recipe',
        status = 'Slide stability status',
        slide_data = 'positions and energies and total magnetinc moments of the slided structures',
        fit_data = 'Fitting parameters to a parabola',
        shifted_from = 'Shows the bilayer is generated by shifting the top layer of another bilayer',
        shifted_to = 'Shows the bilayer was not stable and generated another bilayer')

@command(module = 'asr.dynstab',
         requires = ['structure_zscan.json',
                     'results-asr.zscan.json'])

@option('--blfolder', type=str,
        help='bilayer folder to check slide stability for')

@option('--done-jobs', type=int,
        help='Number of slide stability gs tasks done')
def main(blfolder: str = '.',
         done_jobs: int = 9) -> DynStabResult:
    """ 
    The information we save on this is also saved on dynstab.json
    But we want that file to be created and updated as the jobs get
    submitted and run. This function will run once in the end to 
    collect the information for the database. 
    If some at most 2 gs fails the recipe will be submitted but if you
    manage to converge them the recipe will be submitted agin to update the result file 
    """
    blfolder = str(Path(blfolder).absolute())

    info = read_json(f'{blfolder}/info.json')
    slide_info = read_json(f'{blfolder}/dynstab.json')
    translation = read_json(f'{blfolder}/translation.json')

    results = {'origin': info["original_bilayer_folder"],
               'status': slide_info["DynStab_Result"],
               'slide_data': slide_info["DynStab_Data"],
               'fit_data': slide_info["DynStab_Fit"],
               'shifted_from': translation["shfited_from"] if "shfited_from" in translation else "",
               'shifted_to': translation["shfited_to"] if "shfited_to" in translation else ""}

    return DynStabResult.fromdata(**results)


def slide_stability_subworkflow_done(blfolder, DynStabPath):
    final_result_file = f'{blfolder}/results-asr.dynstab.json'

    # If the result file does not exist the workflow is not done
    if not os.path.isfile(final_result_file):
        return False

    else:
        data = read_json(final_result_file)
        print(">>>>>>>>>>>>>> Stability status [r]: ", data['status'], f">>>>>{blfolder}")

        # If more calculations are done we have to check the bilayer again and resubmit dynstab so it is not done
        dynstab_done_now    = [folder for folder in listdirs(DynStabPath) if os.path.isfile(f'{folder}/results-asr.dynstab@gs_minimal.json')]
        dynstab_done_before = read_json(f'{blfolder}/results-asr.dynstab.json').get("slide_data")["X"] if os.path.isfile(f'{blfolder}/dynstab.json') else []
        if len(dynstab_done_now)>len(dynstab_done_before):
            return False


        # If the bilayer is stable or unstable-max we are done with it
        elif data['status'] in ["Stable", "Unstable-Max"]:
            return True

        else:
            # If we get here the bilayer is not stable so we should relax it
            if not os.path.isfile(f"{blfolder}/results-asr.dynstab@constrained_relax.json"):
                return False           

            # When relaxation is done we have decided where the new bilayer is. If the new folder is created we are done
            else:
                new_folder = read_json(f"{blfolder}/translation.json")
                if "Shifted_to" in new_folder.keys():
                    if "Not" in new_folder['Shifted_to'] or os.path.isdir(f"{blfolder}/../{new_folder}['Shifted_to']"):
                        return True

                #if "Not" in new_folder or os.path.isdir(f"{blfolder}/../{new_folder}"):
                #    return True

    return False
          

##############################################
##### Subworkflow manager ####################
##############################################

def dynstab_manager(blfolder, start_structure, dynstab_path='DynStab', step_num=2, resources="40:12h"):

    DynStabPath = f"{blfolder}/{dynstab_path}"

    if slide_stability_subworkflow_done(blfolder, DynStabPath):
        return []    

    step_size=0.075 #0.05

    if not os.path.isdir(DynStabPath):
        generate_slided_folders(blfolder, start_structure, folders_path=DynStabPath, step_size=step_size, step_num=step_num)

    tasks = []
    for folder in listdirs(DynStabPath):
        tasks += [task("asr.dynstab@gs_minimal", resources=resources, folder=folder, restart=1)]

    etots, xs, ys, magmoms = collect_energies(path=DynStabPath,
                                              step_size=step_size,
                                              result_file = 'results-asr.dynstab@gs_minimal.json')

    stability = stability_analysis(blfolder,
                                   total_energies=etots,
                                   xdisp=xs,
                                   ydisp=ys,
                                   magmoms=magmoms)  
    print(">>>>>>>>>>>>>> Stability status [c]: ", stability, f">>>>>{blfolder}")


    if stability == "unfinished":
        return tasks

    if stability in ['Badfit', 'Unstable-Saddle', 'Unstable-shifted' ]:
        tasks += [task("asr.dynstab@constrained_relax", resources=resources, folder=blfolder, restart=1)]  
  
    if os.path.isfile(f'{blfolder}/results-asr.dynstab@constrained_relax.json'):
        relaxed_structure, is_new, similar_to = unique_bilayer(blfolder, start_structure) 
        print(f">>> Relaxation resulted in new bilayer: {is_new}") 

        create_new_bilayer(blfolder, start_structure, relaxed_structure, is_new, similar_to)

    dynstab_done = [folder for folder in listdirs(DynStabPath) if os.path.isfile(f'{folder}/results-asr.dynstab@gs_minimal.json')]
    dynstab_attempt = [folder for folder in listdirs(DynStabPath) if os.path.isfile(f'{folder}/gs_minimal.txt')]# and not recently_updated(folder=folder, filename='gs_minimal.txt')]

    if len(dynstab_attempt)==9 and len(dynstab_done)>7 and stability!="unfinished":
        #tasks += [task(f"asr.dynstab --done-jobs {len(dynstab_done)}", creates = ['results-asr.dynstab.json'], resources="1:10m", folder=blfolder, restart=1)]  
        tasks += [task(f"asr.dynstab --done-jobs {len(dynstab_done)}", resources="1:10m", folder=blfolder, restart=1)]  

    return tasks


if __name__ == '__main__':
    main.cli()

