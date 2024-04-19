from ase.io import read, write
from asr.core import read_json, command, option, DictStr, ASRResult, prepare_result, AtomsFile
from ase import Atoms
import numpy as np
from ase.calculators.dftd3 import DFTD3
import os
from asr.utils.symmetry import atoms2symmetry
from spglib import get_symmetry_dataset
from functools import cached_property
from gpaw import GPAW
import shutil
from asr.workflow.bilayerutils import silentremove

class CheckMagmoms:
    mag_elements = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                    'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
                    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl']

    def __init__(self, bilayer, state, initial_magmoms, magmoms, magmom=None):
        self.bilayer = bilayer.copy()
        self.initial_magmoms = initial_magmoms
        self.magmoms = magmoms
        self.magmom = magmom if magmom is not None else sum(magmoms)
        self.state = state    

    @cached_property
    def equivalent_atoms(self):
        from asr.utils.symmetry import atoms2symmetry
        symmetry = atoms2symmetry(self.bilayer, tolerance=0.01, angle_tolerance=0.1)
        eq_atoms =  symmetry.dataset['equivalent_atoms']
        return eq_atoms


    def get_magnetic_atoms(self):
        mag_atoms = []
        mag_atoms_type = []
        mag_eq_atoms = []
        for iatom, atom in enumerate(self.bilayer):
            if atom.symbol in CheckMagmoms.mag_elements:
               mag_atoms.append(iatom)
               mag_atoms_type.append(atom.symbol)
               mag_eq_atoms.append(self.equivalent_atoms[iatom])

        return mag_atoms, mag_atoms_type, mag_eq_atoms


    def get_deviation_matrix(self):
        mag_atoms, mag_atoms_type, mag_eq_atoms = self.get_magnetic_atoms()
        
        deviation_matrix = []
        for x in mag_atoms:
            deviation_matrix_row = []
            for y in mag_atoms:
                if abs(self.magmoms[x])>=0.05 or abs(self.magmoms[y])>=0.05:
                    deviation_matrix_row.append((abs(self.magmoms[x]) - abs(self.magmoms[y]))/abs(self.magmoms[x]))
                else:
                    deviation_matrix_row.append(0)
            deviation_matrix.append(deviation_matrix_row)
        return np.array(deviation_matrix)


    def check_magmoms_signs(self):
        '''When we check deviations we don't check sign changes so there could be 
           overall sign changes but we don't want that to happen'''
        same_sign = True
        symbols = self.bilayer.get_chemical_symbols()
        for x in [i for i, e in enumerate(symbols) if e in CheckMagmoms.mag_elements]:
            mag_max = max(abs(np.array(self.initial_magmoms)))
            if abs(self.initial_magmoms[x])>(0.1*mag_max):
                if not np.sign(self.magmoms[x]) == np.sign(self.initial_magmoms[x]):
                    same_sign = False
                    break
        return same_sign 


    def magmstate_healthy(self, dipz=None, dipz_threshold=None):
        '''the information about out of plane dipole is only needed for AFM 
           note that we compare atoms with same wykoff positions so if there is out 
           of plane dipole the atoms are not compared and deviation will not be reported'''
        mag_atoms, mag_atoms_type, mag_eq_atoms = self.get_magnetic_atoms()
        deviation_matrix = self.get_deviation_matrix()

        check_values = []
        
        for m, x, w1 in zip(deviation_matrix, mag_atoms_type, mag_eq_atoms):
            for n, y, w2 in zip(m, mag_atoms_type, mag_eq_atoms):
                if abs(n) > 0.1 and x == y and w1==w2:
                    check_values.append(n)
 
        # total magmom being for AFM state 0 is only checked when there is not out of plabe dipole
        if self.state == 'AFM':
            if dipz<dipz_threshold and not np.allclose((self.magmom/len(mag_atoms)),0, atol=0.01):
                check_values.append('illdefined')

        # We check the sign of magmoms not changing during the gs calculation
        consistent_sign = self.check_magmoms_signs()
        if not consistent_sign:
            return False

        mag_state = True if (len(check_values) == 0) else False
        return mag_state


def set_magnetic_config(bilayer_atoms, mlmagmoms, config=''):
    bilayer = bilayer_atoms.copy()

    if config.lower() == 'fm':
        magmoms_bilayer = list(mlmagmoms) + list(mlmagmoms)
        bilayer.set_initial_magnetic_moments(magmoms_bilayer)
    
    elif config.lower() == 'afm':
        magmoms_bilayer = list(mlmagmoms) + list(-mlmagmoms)
        bilayer.set_initial_magnetic_moments(magmoms_bilayer)
    else:
        raise ValueError(f'Configuration ({config}) not recognized. Must be FM or AFM')

    return bilayer


def calc_setup(blatoms, settings, mixertype, hubbardu, output_text, U_dict=None):
    from ase.calculators.calculator import get_calculator_class

    settings: dict = {
        'name': 'gpaw',
        'mode': {'name': 'pw', 'ecut': 800},
        'xc': 'PBE',
        'basis': 'dzp',
        'kpts': {'density': 12.0, 'gamma': True},
        'occupations': {'name': 'fermi-dirac',
                        'width': 0.05},
        'maxiter': 5000,
        'mixer': {"method": mixertype,
                  "beta": 0.01,
                  "history": 5,
                  "weight": 50},
        'poissonsolver' : {'dipolelayer': 'xy'},
        'convergence': None, # I will set the convergence later because I want to hard code it
        'nbands': '200%',
        **(settings or {})}

    # Don't change or allow the user to change the convergence criteria for consistancy and GPAW limitations.
    convergence_energy = {'bands': 'CBM+3.0', "density": 1e-6, 'energy': {"name": "energy", "tol": 1e-6, "relative": False,  "n_old": 3} }
    calculator = {**settings, 'convergence': convergence_energy}

    if hubbardu < 0:
        # 3d TM atoms which need a Hubbard U correction
        TM3d_atoms = {'V':3.1, 'Cr':3.5, 'Mn':3.8, 'Fe':4.0, 'Co':3.3, 'Ni':6.4, 'Cu':4.0} if U_dict is None else U_dict
        atom_ucorr = set([atom.symbol for atom in blatoms if atom.symbol in TM3d_atoms])
        U_corrections_dct = {symbol: f':d, {TM3d_atoms[symbol]}' for symbol in atom_ucorr}
    elif hubbardu > 0:
        TM3d_atoms = ['V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu']
        atom_ucorr = set([atom.symbol for atom in blatoms
                      if atom.symbol in TM3d_atoms])
        U_corrections_dct = {symbol: f':d, {hubbardu}' for symbol in atom_ucorr}

    if abs(hubbardu)>0.001:
       calculator.update(setups=U_corrections_dct)

    name = calculator.pop('name')
    calc = get_calculator_class(name)(**calculator, txt=output_text)
 
    calc_d3 = DFTD3(dft=calc, cutoff=60)

    return calc_d3


def calc_gs_old(calc, bilayer_atoms, state, monolayer_dipole, mlmagmoms):
    bilayer = bilayer_atoms.copy()
    bilayer = set_magnetic_config(bilayer, mlmagmoms, config=state)

    bilayer.set_calculator(calc)
    energy = bilayer.get_potential_energy()
    bilayer.get_forces()

    magmom  = bilayer.calc.dft.get_magnetic_moment()
    magmoms = bilayer.calc.dft.get_magnetic_moments()
    initial_magmoms = bilayer.get_initial_magnetic_moments()

    check_magmoms = CheckMagmoms(bilayer,
                                 state,
                                 initial_magmoms,
                                 magmoms,
                                 magmom)
    if check_magmoms.magmstate_healthy(dipz=monolayer_dipole, dipz_threshold=1e-3):
        mag_state = state
    else:
        mag_state = 'illdefined'

    return bilayer, energy, mag_state, magmoms, magmom 


def calc_gs_new(calc, bilayer_atoms, state, monolayer_dipole, mlmagmoms):

    result_file = f"gs_{state}.gpw"
    
    if os.path.isfile(result_file):
        calc = GPAW(result_file)
        bilayer = calc.atoms
        bilayer.calc = calc
    else:
        bilayer = bilayer_atoms.copy()
        bilayer = set_magnetic_config(bilayer, mlmagmoms, config=state)
        bilayer.set_calculator(calc)

    energy = bilayer.get_potential_energy()
    bilayer.get_forces()

    magmom  = bilayer.calc.dft.get_magnetic_moment()
    magmoms = bilayer.calc.dft.get_magnetic_moments()
    initial_magmoms = bilayer.get_initial_magnetic_moments()

    check_magmoms = CheckMagmoms(bilayer,
                                 state,
                                 initial_magmoms,
                                 magmoms,
                                 magmom)
    if check_magmoms.magmstate_healthy(dipz=monolayer_dipole, dipz_threshold=1e-3):
        mag_state = state
    else:
        mag_state = 'illdefined'

    if not os.path.isfile(result_file):
        bilayer.calc.dft.write(result_file)

    return bilayer, energy, mag_state, magmoms, magmom


def calc_gs(calc, bilayer_atoms, state, monolayer_dipole, mlmagmoms):

    result_file = f"gs_{state}.gpw"

    if os.path.isfile(result_file):
        calc_done = GPAW(result_file)
        bilayer = calc_done.atoms
        bilayer.calc = calc_done
    else:
        bilayer = bilayer_atoms.copy()
        bilayer = set_magnetic_config(bilayer, mlmagmoms, config=state)
        bilayer.set_calculator(calc)

    energy = bilayer.get_potential_energy()
    bilayer.get_forces()

    magmom  = bilayer.get_magnetic_moment()
    magmoms = bilayer.get_magnetic_moments()
    initial_magmoms = bilayer.get_initial_magnetic_moments()

    check_magmoms = CheckMagmoms(bilayer,
                                 state,
                                 initial_magmoms,
                                 magmoms,
                                 magmom)
    if check_magmoms.magmstate_healthy(dipz=monolayer_dipole, dipz_threshold=1e-3):
        mag_state = state
    else:
        mag_state = 'illdefined'

    if not os.path.isfile(result_file):
        bilayer.calc.dft.write(result_file)

    return bilayer, energy, mag_state, magmoms, magmom


@prepare_result
class MagResult(ASRResult):
     eAFM: float 
     eFM: float
     eDIFF: float           
     M_AFM: float 
     M_FM: float 
     magmoms_afm: np.ndarray
     magmoms_fm: np.ndarray
     AFM_state: str
     FM_state: str

     key_descriptions = dict(
         eAFM = 'Total energy of the AFM state',
         eFM = 'Total energy of the FM state',
         eDIFF = 'Inerlayer magnetic exchange energy in eV',
         M_AFM = 'Total magnetic moment of the AFM state',
         M_FM = 'Total magnetic moment of the FM state',
         magmoms_afm = 'magnetic moments of the AFM state',
         magmoms_fm = 'magnetic moments of the FM state',
         AFM_state = 'Final magnetic state of the calculation started as AFM', 
         FM_state = 'Final magnetic state of the calculation started as FM')


@command(module='asr.interlayer_magnetic_exchange',
         requires=['structure_zscan.json', 
                   '../structure_initial.json',
                   'results-asr.zscan.json'])

@option('-a', '--bilayer-atoms', help='Bilayer structure',
        type=AtomsFile(), default='structure_zscan.json')
@option('-calc', '--settings', help='Calculator params.', type=DictStr())
@option('-u', '--hubbardu', type=float, help="Hubbard U correction")
@option('-ml', '--mlfolder', type=str, help="monolayer folder used to set the initial magmoms")
@option('-mix', '--mixertype', type=str, help="mixer: sum or difference")

def main(bilayer_atoms: Atoms,
         settings = None,
         hubbardu: float = 0.0, # if negative uses the hard coded list with different U for each atom
         mlfolder: str = '..',
         mixertype: str = "sum") -> ASRResult:

    """Calculate the energy difference between FM and AFM configurations.

    Returns the energy difference between the FM and
    AFM configurturations for a bilayer, where FM is
    defined as all spins pointing in the same directions,
    and AFM is defined as spins in the two layers
    pointing in opposite directions.

    If U is non-zero, Hubbard-U correction is used for 3d TM
    atoms, if there are any.

    """
    atoms = read(f'{mlfolder}/structure.json')

    # Read the magnetic moments from the monolayers.
    # This is used as the starting point for the bilayer magnetic moments.
    ml_magmoms = read_json(f"{mlfolder}/structure_initial.json")[1]["magmoms"]
    monolayer_dipole = read_json(f'{mlfolder}/results-asr.gs.json')['dipz']

    # FM calculation
    calc_fm_D3 = calc_setup(bilayer_atoms, settings, mixertype, hubbardu, output_text='fm.txt')

    bilayer_FM, eFM, FM_state, fm_magmoms, fm_magmom  = calc_gs(calc=calc_fm_D3,
                                                                bilayer_atoms=bilayer_atoms,
                                                                state="FM",
                                                                monolayer_dipole=monolayer_dipole,
                                                                mlmagmoms=ml_magmoms)
    
    # AFM Calculation
    calc_afm_D3 = calc_setup(bilayer_atoms, settings, mixertype, hubbardu, output_text='afm.txt')
 
    bilayer_AFM, eAFM, AFM_state, afm_magmoms, afm_magmom  = calc_gs(calc=calc_afm_D3, 
                                                                     bilayer_atoms=bilayer_atoms, 
                                                                     state="AFM", 
                                                                     monolayer_dipole=monolayer_dipole,
                                                                     mlmagmoms=ml_magmoms)


    eDIFF = eFM - eAFM

    if FM_state == 'FM' and AFM_state == 'AFM':
        if eDIFF > 0:
            #bilayer_AFM.calc.dft.write(f'gs.gpw')
            shutil.copy("gs_AFM.gpw", "gs.gpw")
            bilayer_AFM.write(f'structure.json')
        else:
            #bilayer_FM.calc.dft.write(f'gs.gpw')
            shutil.copy("gs_FM.gpw", "gs.gpw")
            bilayer_FM.write(f'structure.json')

    if FM_state == 'FM' and AFM_state == 'illdefined':
        #bilayer_FM.calc.dft.write(f'gs.gpw')
        shutil.copy("gs_FM.gpw", "gs.gpw")
        bilayer_FM.write(f'structure.json')
    
    if FM_state == 'illdefined' and AFM_state == 'AFM':
        #bilayer_AFM.calc.dft.write(f'gs.gpw')
        shutil.copy("gs_AFM.gpw", "gs.gpw")
        bilayer_AFM.write(f'structure.json')

    # If both FM amd AFM calculations are done and we saved the most stable one as "gs.gpw" we delete FM and AFM gpw files to save space
    silentremove(filepath="gs_FM.gpw", checkotherfileexists="results-asr.interlayer_magnetic_exchange.json")
    silentremove(filepath="gs_AFM.gpw", checkotherfileexists="results-asr.interlayer_magnetic_exchange.json")

    results = {'eAFM': eAFM,
               'eFM': eFM,
               'eDIFF': eDIFF,
               'M_AFM': afm_magmom,      
               'M_FM': fm_magmom,
               'magmoms_afm': afm_magmoms,
               'magmoms_fm': fm_magmoms,
               'AFM_state': AFM_state,
               'FM_state': FM_state}

    return MagResult.fromdata(**results)

if __name__ == '__main__':
    main.cli()
