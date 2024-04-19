from asr.core import read_json, command, option, AtomsFile, DictStr, prepare_result, ASRResult
import numpy as np
from ase import Atoms
import os
from asr.utils.bilayerutils import translation
from ase.db import connect
import shutil
from asr.utils.symmetry import atoms2symmetry
from spglib import get_symmetry_dataset
from asr.stack_bilayer import BuildBilayer
from asr.interlayer_magnetic_exchange import CheckMagmoms
from ase.io import read
from gpaw import mpi


def webpanel():
    """Return a webpanel showing the binding energy curve.
    Also show the fit made to determine the stiffness.
    """
    raise NotImplementedError()


def min_quality(energies, heights):
    '''
    Makes sure that at least one interlayer checked below and above the min
    Also makes sure that the energies of the closest step before and after 
    the min is not more than 2meV different (so there was not a jump)
    '''
    ee = energies
    hh = heights
    sorted_index = sorted(range(len(ee)), key=lambda k: ee[k])

    e_left=1000
    e_right=1000
    e_min = ee[sorted_index[0]]
    h_min = hh[sorted_index[0]]

    if e_min>0:
       return False, e_min, h_min

    if hh[sorted_index[1]]>hh[sorted_index[0]] and e_min<0:
       e_right = ee[sorted_index[1]]
       h_right = hh[sorted_index[1]]
       for ii in range(len(ee)):
           if ii>1:
              if hh[sorted_index[ii]]<hh[sorted_index[0]] and e_left==1000:
                 e_left = ee[sorted_index[ii]]
                 h_left = hh[sorted_index[ii]]
       if e_left==1000:
          return False, e_min, h_min

    if hh[sorted_index[1]]<hh[sorted_index[0]] and e_min<0:
       e_left = ee[sorted_index[1]]
       h_left = hh[sorted_index[1]]
       for ii in range(len(ee)):
           if ii>1:
              if hh[sorted_index[ii]]>hh[sorted_index[0]] and e_right==1000:
                 e_right = ee[sorted_index[ii]]
                 h_right = hh[sorted_index[ii]]
       if e_right==1000:
          return False, e_min, h_min

    return e_min<0 and e_right<0 and e_left<0 and abs(e_min-e_left)<0.001 and abs(e_min-e_right)<0.001


def bilayer_stiffness(energy_curve, window=0.01):
    """Calculate bilayer stiffness.

    We define the bilayer stiffness as the curvature
    of the binding energy curve at the minimum.
    That is, we are calculating an effective spring
    constant.

    For now we include this property calculation here
    since we can get it almost for free.
    """
    # We do a second order fit using points within
    # 0.01 eV of minimum
    ds = energy_curve[:, 0]
    es = energy_curve[:, 1]
    mine = np.min(es)

    X = ds[np.abs(es - mine) < window]
    Y = es[np.abs(es - mine) < window]

    I = np.array([X**0, X**1, X**2]).T

    P, residuals, rank, singulars = np.linalg.lstsq(I, Y, rcond=None)

    return P


def calc_setup(settings, txt=None, hubbard_U=False):
    from gpaw import MixerDif, Mixer, MixerSum, PW, GPAW
    from ase.calculators.dftd3 import DFTD3
    # Note: I don't want to pop data from stetting and then update it because I want to reuse this function with original setting

    setting_keyes = ['xc', 'maxiter', 'kpts', 'convergence']
    calcsettings = {key: settings[key] for key in setting_keyes}    

    try:
       mixersettings = settings['mixer'].copy()
    except:
       mixersettings = None
    
    mixertype = mixersettings.pop('type', 'mixersum')
    if type(mixersettings) != dict:
        mixersettings = {'beta': 0.02, 'nmaxold': 5,
                         'weight': 50} 
        calcsettings['mixer'] = MixerSum(**mixersettings)
    elif mixertype=='mixersum':
        calcsettings['mixer'] = MixerSum(**mixersettings)
    elif mixertype=='mixer':
        calcsettings['mixer'] = Mixer(**mixersettings)
    elif mixertype=='default' or mixertype=='mixerdif':
        calcsettings['mixer'] = MixerDif(**mixersettings)  
    

    calcsettings['mode'] = PW(settings['PWE'])
    calcsettings['symmetry'] = {'symmorphic': False}
    calcsettings['occupations'] = {'name': 'fermi-dirac',
                                   'width': 0.05}
    calcsettings['poissonsolver'] = {'dipolelayer': 'xy'}
    calcsettings['nbands'] = '200%'

    if txt is None: 
        calcsettings['txt'] = 'zscan.txt'
    else:
        calcsettings['txt'] = txt

    # 3d TM atoms which need a Hubbard U correction
    if hubbard_U:
        TM3d_atoms = {'V':3.1, 'Cr':3.5, 'Mn':3.8, 'Fe':4.0, 'Co':3.3, 'Ni':6.4, 'Cu':4.0}
        atom_ucorr = set([atom.symbol for atom in atoms if atom.symbol in TM3d_atoms])
        U_corrections_dct = {symbol: f':d, {TM3d_atoms[symbol]}' for symbol in atom_ucorr}
        calcsettings.update(setups=U_corrections_dct)

    calc = GPAW(**calcsettings)

    # This calculation has to have d3 so I want an error if it is not in the setting
    d3 = settings['d3']
    if d3: calc = DFTD3(dft=calc, cutoff=60)

    return calc


def get_monolayer_energy(mlfolder, settings):
    mlatoms = read(f'{mlfolder}/structure_adjusted.json')
    calc_setting = settings.copy() 
    mlatoms.calc = calc_setup(calc_setting, txt='zscan_ml.txt')
    e = mlatoms.get_potential_energy()
    return e


def get_binding_energy(monolayer_energy, bilayer_energy, cell):
    cellarea = np.linalg.norm(np.cross(cell[0], cell[1]))
    return -1000*(bilayer_energy-2*monolayer_energy)/cellarea


def get_interlayer(bilayer):
    ''' Min interlayer distance
        Bilayer atoms are tagged, bot=0, top=1     
    '''
    bmax = max([atom.position[2] for atom in bilayer if atom.tag==0])   
    tmin = min([atom.position[2] for atom in bilayer if atom.tag==1])
    return tmin-bmax


def get_optimal_height(bilayer, index1=0, index2=None):
    index2 = int(len(bilayer)/2) if index2 is None else index2
    return bilayer.positions[index2,2]-bilayer.positions[index1,2]


def build_bilayer(bilayer, d):
    bl = bilayer.copy()
    d0 = get_interlayer(bilayer)
   
    for iatom, atom in enumerate(bilayer): 
         if atom.tag==1:
             bl.positions[iatom, 2] += (d-d0) 

    return BuildBilayer.adjust_bilayer_vacuum(vacuum=15, bilayer=bl)


def get_energy(bilayer, h, calc0, memo):
    # First check if height h has already been calculated
    # memo = [(height1, energy1, magmom1), (height2, energy2, magmom2), ...] 
    try:
        h0, e0, mag0 = next(t for t in memo if abs(t[0]-h)<0.001)
        return e0
    except StopIteration:
        pass

    atoms = build_bilayer(bilayer, h)
    atoms.calc = calc0

    e = atoms.get_potential_energy()   
 
    # We want to check the magnetic moment is not changing much if the material has +U or if the monolayer was magnetic
    # The initial magmom of prototpye is either the FM stacked final magmom of the monolayer or the hund law
    if "initial_magmoms" in read_json("bilayerprototype.json")[1]:   
       #bilayer_magmom = atoms.get_magnetic_moment() #this does not work because the calculator is D3 (relax recipe problem)
       bilayer_magmom = atoms.calc.dft.get_magnetic_moment()
       bilayer_magmoms = atoms.calc.dft.get_magnetic_moments() 
       bl_initial_magmoms = atoms.get_initial_magnetic_moments()
       check_magmoms = CheckMagmoms(bilayer, 
                                    state='FM', 
                                    initial_magmoms=bl_initial_magmoms, 
                                    magmoms=bilayer_magmoms, 
                                    magmom=bilayer_magmom)
       assert check_magmoms.magmstate_healthy(), '>>> Final magnetic moments are not accepted'
 
    else:
       bilayer_magmom = 0
 
    callback_fn(memo, h, e, bilayer_magmom)

    return e


def callback_fn(energy_curve, h, energy, totmag):
    from gpaw import mpi
    energy_curve.append((h, energy, totmag))
    if mpi.rank == 0:
       np.save('energy_curve.npy', np.array(energy_curve))


def zscan_interlayer(calc, settings, blatoms, energy_curve, h, step, energy_tol, distance_tol, maxiter=12):

    step0 = step
    eold = 10.0
    enew = eold
    h += 2*step
    icount = 0
    h_restart = True

    enew_list = []
    while (abs(step)>distance_tol or icount==0 or np.abs(enew-eold)>energy_tol) and icount<maxiter and enew!=None:
       print('Interlayer distance: ', h, 'Restart mode: ', h_restart)

       icount += 1
       if enew>eold: 
           step *= (-0.5)
       h -= step
       eold = enew

       try: 
           enew = get_energy(blatoms.copy(), h, calc, energy_curve)
       except:
           if h_restart:
              print("Restart zscan with larger interlayer distances.")
              shutil.copy('./zscan.txt','./zscan-before-restart.txt')    
              h += 0.5
              step = step0
              h_restart = False
              calc = calc_setup(settings)
              try:
                 enew = get_energy(blatoms.copy(), h, calc, energy_curve)
              except:
                 # If calculation does not converge enew remains the same
                 pass

       # If the last 5 steps did not converge we want to stop
       enew_list.append(enew)
       if len(enew_list)>=5:
          if len(set(enew_list[-5:-1]))==1:
             icount = 1000
       
       print('inside', abs(step)>distance_tol , icount<maxiter , enew!=None , (icount==0 or np.abs(enew-eold)>energy_tol))

    # Here I want to make sure at least one distance smaller than min is checked
    curve = np.array(energy_curve)
    assert np.size(curve)>1, 'No zscan step converged'

    if abs(h-min(curve[:,0]))<0.0001:
       if enew>eold:
          step *= (-0.5)
       h -= step
       eold = enew
       enew = get_energy(blatoms.copy(), h, calc, energy_curve)

    if enew is None:
       status = False
       reason = f'None energy - Magmom issue - {icount} Step'
    elif abs(step)>0.03:#distance_tol:
       status = False
       reason = f'Interlayer distance tolerenece not acheived - {icount} Step'
    else:
       status = True
       reason = f'Successful - {icount} Step'

    return status, reason, energy_curve


def optimize_interlayer(calc, settings, blatoms, energy_curve, d0, tol):
    import scipy.optimize as sciop 

    def energy_fn(h):
        return get_energy(blatoms.copy(), h[0], calc, energy_curve)

    opt_result = sciop.minimize(energy_fn, x0=d0, method="Nelder-Mead", tol=tol)

    if not opt_result.success:
        status = False
        reason = 'Relaxation failed in optimizer'
    else:
        status = True
        reason = 'Relaxation succeeded in optimizer'

    return status, reason, energy_curve


def calculate(settings, atoms, method, start_structure, distance, zscan_step, zscan_energy_tol, distance_tol, remove_history=False):
    from gpaw import mpi
    calc = calc_setup(settings.copy())

    if os.path.isfile('energy_curve.npy') and remove_history:
        try:
            os.remove('energy_curve.npy')
        except:
            pass

    if os.path.exists('energy_curve.npy'):
        energy_curve = np.load('energy_curve.npy', allow_pickle=True)
        energy_curve = [(d, e, mag) for d, e, mag in energy_curve]
    else:
        energy_curve = []

    bottom_layer = atoms.copy()
    if method == 'zscan':
       status, reason, energy_curve = zscan_interlayer(calc, settings, start_structure.copy(), energy_curve, distance, zscan_step, zscan_energy_tol, distance_tol, maxiter=200)

    elif method == 'optimize':
       status, reason, energy_curve = optimize_interlayer(calc, settings, start_structure.copy(), energy_curve, distance, distance_tol)

    curve = np.array(energy_curve)
    # We do this because maybe the last calculation is not the min energy
    ee=curve[:, 1]
    hh=curve[:, 0]
    sorted_index = sorted(range(len(ee)), key=lambda k: ee[k])
    emin = ee[sorted_index[0]]
    hmin = hh[sorted_index[0]]

    if mpi.rank == 0:
        np.save('energy_curve.npy', np.array(energy_curve))

    return status, ee, hh, emin, hmin, curve, reason



@prepare_result
class RelaxBilayerResult(ASRResult):
    method: str
    heights: np.ndarray
    energies: np.ndarray
    total_magmoms: np.ndarray
    optimal_height: float
    interlayer_distance: float
    bilayer_energy: float
    monolayer_energy: float
    zscan_binding_energy: float
    curvature: float
    FittingParams: np.ndarray

    key_descriptions = dict(
        method='Calculation method (zscan or optimise)',
        heights='Interlayer distances calculated during optimization (closest atoms)',
        energies='Energies calculated during optimization (including D3 correction)',
        total_magmoms='Total magnetic moment in each interlayer distance',
        optimal_height='The z-distance between two specific atoms (by default first atom of each layer)',
        interlayer_distance='Distance between highest atom in the bottom layer from lowest atom in the top layer',
        bilayer_energy='Energy at the optimized interlayer distance [eV]',
        monolayer_energy = 'Energy of the monolayer calculated with same settings [eV]',
        zscan_binding_energy = 'Binding energy from zscan calculation in [meV/A2] (convergence can be improved for final Eb)',
        curvature='Curvature at the optimized interlayer distance',
        FittingParams='Parameters for second order fit')

@command(module='asr.zscan',
         creates = ['structure_zscan.json'],
         requires = ['bilayerprototype.json',
                     'transformdata.json',
                     'translation.json',
                     '../structure_adjusted.json']) #The structure that we move to the origin and the vacuum adjusted for monolayer etot calculation

@option('-a', '--atoms', help='Bilayer structure',
        type=AtomsFile(), default='bilayerprototype.json')
@option('--bottom-layer-indices', help='Indices of bottom layer atoms 0,1,2,3,4 or 0-5 (start from 0, in 0-5 format last number does not count)',
        type=str)
@option('--indices-of-atoms-to-be-compared', help='Cama seperated indices of the atoms that optimal height is their z-distance (start from 0)',
        type=str)
@option('-s', '--settings', help='Relaxation settings',
        type=DictStr())
@option('--distance-tol', help='Interlayer convergence threshold',
        type=float)
@option('-d', '--distance', help='Initial interlayer distance',
        type=float)
@option('-v', '--vacuum', help='Extra vacuum',
        type=float)
@option('--method', help='Interlayer calculation method',
        type=str, default='optimize')
@option('-zs', '--zscan-step', help='Zscan initial step',
        type=float)
@option('-ze', '--zscan-energy-tol', help='Zscan energy tol',
        type=float)
@option('--restart/--norestart', help='Delete memo and start relaxation from scratch',
        is_flag=True, type=bool)
@option('--outputname', help='Name of output file', type=str)
def main(atoms: Atoms,
         bottom_layer_indices: str = '', 
         indices_of_atoms_to_be_compared: str = '',
         settings = None,
         distance_tol: float = 1e-2,
         distance: float = 5,
         vacuum: float = 6,
         method: str = 'optimize',
         zscan_step: float = 0.079,
         zscan_energy_tol: float = 0.000001,
         restart: bool = False,
         outputname: str = 'structure_zscan.json') -> ASRResult:

    from asr.core import read_json
    from ase.io import read
    from gpaw import mpi
    import scipy.optimize as sciop

    settings: dict = {'d3': True,
                      'xc': 'PBE',
                      'PWE': 700, # 'mode': {'name': 'pw', 'ecut': 800}
                      'maxiter': 800,
                      'kpts': {'density': 8.0, 'gamma': False}, #'kpts': {'density': 6.0, 'gamma': True},
                      'convergence': {'bands': 'CBM+3.0', 'energy': {"name": "energy", "tol": 1e-4, "relative": False,  "n_old": 3} },
                      'mixer': {'type': 'mixersum', # used for the first trouble shooting of magntic systems
                                'beta': 0.02, #0.02,
                                'nmaxold': 5, #None,
                                'weight': 50}, #None},
                      **(settings or {})}

    if restart:
        if mpi.rank == 0:
            if os.path.exists('energy_curve.npy'):
                os.remove('energy_curve.npy')
        mpi.world.barrier()

    # Here we get the list of indices of top and bottom layer atoms
    if bottom_layer_indices == '':
        bot_indices = range(0,int(len(atoms)/2))
    elif '-' in bottom_layer_indices:
        start, end = bottom_layer_indices.split('-')
        bot_indices = range(int(start), int(end))
    elif ',' in bottom_layer_indices:
        bot_indices = list(map(int, bottom_layer_indices.split(',')))     
    top_indices = [index for index in range(0,int(len(atoms))) if index not in bot_indices]


    # tag the atoms of top and bottom layer  
    tags = [0 if iatom in bot_indices else 1 for iatom, atom in enumerate(atoms)]
    atoms.set_tags(tags)

    # We will later overwrite this on the bilayerprototpye 
    tagged_bilayerprototype = atoms.copy()

    # optimal height will be the distance between these two atoms
    if indices_of_atoms_to_be_compared == '':
        index1, index2 = 0, int(len(atoms)/2)
    else:
        index1, index2 = map(int, indices_of_atoms_to_be_compared.split(','))

    # We can start the interlayer distances from any given distance
    start_structure = build_bilayer(atoms.copy(), distance)
   
    status, ee, hh, e_min, h_min, curve, reason = calculate(settings, atoms.copy(), method, start_structure.copy(), distance, zscan_step, zscan_energy_tol, distance_tol, remove_history=False)
 
    # This is one step trouble shooting if the energies where not converged well
    if not status or not min_quality(energies=ee, heights=hh):
        print("Restarted with stronger convergence", f'zscan_step = {zscan_step}')
        reset_settings = {**settings, 'convergence': {'bands': 'CBM+3.0', 'energy': {"name": "energy", "tol": 1e-5, "relative": False,  "n_old": 3} }}
        status, ee, hh, e_min, h_min, curve, reason = calculate(reset_settings, atoms.copy(), method, start_structure.copy(), distance, zscan_step, zscan_energy_tol, distance_tol, remove_history=True)

    # We want to check the quality of the zscan calculation 
    if not min_quality(energies=ee, heights=hh):
        raise ValueError("The zscan is not trsutable. Jumps near the min")

    if not status:
        print(reason)
        raise ValueError(f'Relaxation failed - {reason}')

    #e_ml = get_monolayer_energy(mlfolder='./..', calc=calc_setup(settings.copy()))
    e_ml = get_monolayer_energy(mlfolder='./..', settings = settings.copy())

    eb = get_binding_energy(monolayer_energy=e_ml, bilayer_energy=e_min, cell=atoms.cell)

    P = bilayer_stiffness(curve)

    # We tag the top and bottom layers for this recipe, we save those tags on the bilayerprototype
    tagged_bilayerprototype.write('bilayerprototype.json')   
 
    # This file has to be written at the end after we check that zscan has been successful
    final_atoms = build_bilayer(atoms.copy(), h_min) 
    final_atoms.write(outputname)

    results = {'method': method,
               'heights': curve[:, 0],
               'energies': curve[:, 1],
               'total_magmoms': curve[:, 2],
               'optimal_height': get_optimal_height(final_atoms, index1, index2),
               'interlayer_distance': h_min,
               'bilayer_energy': e_min,
               'monolayer_energy': e_ml,
               'zscan_binding_energy': eb,
               'curvature': P[2],
               'FittingParams': P}

    return RelaxBilayerResult.fromdata(**results)


if __name__ == '__main__':
    main.cli()
