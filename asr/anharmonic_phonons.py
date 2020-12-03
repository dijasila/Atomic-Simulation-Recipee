#general python
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os.path
from os import path

#ase
from ase import Atoms
from ase.io import *
from ase.calculators.emt import EMT
from ase.parallel import world
from ase.build import make_supercell

#hiphive
from hiphive.structure_generation import generate_mc_rattled_structures
from hiphive.utilities import prepare_structures
from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential, ForceConstants
from hiphive.fitting import Optimizer

#asr
from asr.core import (command, option, DictStr, ASRResult,read_json, write_json, prepare_result,AtomsFile)

#phonopy & phono3py
from phonopy import Phonopy
from phono3py import Phono3py
from phonopy.structure.atoms import PhonopyAtoms
import phono3py
from phono3py.cui.create_force_constants import parse_forces, forces_in_dataset
from phono3py.file_IO import read_fc3_from_hdf5, read_fc2_from_hdf5
from phono3py.phonon3.fc3 import show_drift_fc3
from phonopy.harmonic.force_constants import show_drift_force_constants
from phonopy.interface.calculator import get_default_physical_units
import phonopy.cui.load_helper as load_helper

#gpaw

from gpaw import GPAW, PW, FermiDirac, mpi

def hiphive_fc23(atoms, cellsize, number_structures, rattle, mindistance, nat_dim, cut1, cut2, cut3, fc2n, fc3n, calculator):
   structures_fname = str(cellsize)+'_'+str(number_structures)+'_'+str(rattle)+'_'+str(mindistance)+'.extxyz'
   a=read(atoms)
   aa=Atoms(a)

   #2D or 3D calculation

   if nat_dim==3:
     #3D calc
     multiplier=np.array([[cellsize,0,0],[0,cellsize,0],[0,0,cellsize]])
   else:
     #2D calc
     multiplier=np.array([[cellsize,0,0],[0,cellsize,0],[0,0,1]])

   #calculator type
   atoms_ideal = make_supercell(aa, multiplier)
   atoms_ideal.pbc = (1, 1, 1)
   if calculator=='DFT':
     calc = GPAW(mode='lcao',
               basis='dzp',
               xc = 'PBE',
               h = 0.2,
               kpts={"size": (2,2,1), "gamma": True},
               symmetry={'point_group': False},
               convergence={'forces':1e-4},
               occupations=FermiDirac(0.05),
               txt= 'phono3py.txt')
   else:
     calc = EMT()
   #create rattled structures or read them from file

   if path.exists(structures_fname)==False:
     structures = generate_mc_rattled_structures(atoms_ideal, number_structures, rattle, mindistance)
     structures = prepare_structures(structures, atoms_ideal, calc)
     write(structures_fname,structures)
   else:
     structures = read(structures_fname+'@:')

   #proceed with cluster space generation and fcp optimization

   cutoffs = [cut1,cut2,cut3]
   cs = ClusterSpace(structures[0], cutoffs)
   sc = StructureContainer(cs)
   for structure in structures:
      sc.add_structure(structure)
   opt = Optimizer(sc.get_fit_data())
   opt.train()
   # construct force constant potential
   fcp = ForceConstantPotential(cs, opt.parameters)

   #get phono3py supercell and build phonopy object. Done in series

   if world.rank == 0:
      b=read(atoms)
      prim = Atoms(b)
      atoms_phonopy = PhonopyAtoms(symbols=prim.get_chemical_symbols(),
                                   scaled_positions=prim.get_scaled_positions(),
                                   cell=prim.cell)
      phonopy = Phonopy(atoms_phonopy, supercell_matrix=multiplier,
                        primitive_matrix=None)
      supercell = phonopy.get_supercell()
      supercell = Atoms(cell=supercell.cell, numbers=supercell.numbers, pbc=True,
                        scaled_positions=supercell.get_scaled_positions())

      #get force constants from fcp potentials
      fcs = fcp.get_force_constants(supercell)
      # write force constants 
      fcs.write_to_phonopy(fc2n)
      fcs.write_to_phono3py(fc3n)


def phono3py_lifetime(atoms, cellsize, nat_dim, mesh_ph3, fc2n, fc3n, t1, t2, tstep):
   # get phono3py supercell
   b=read(atoms)
   prim = Atoms(b)
   atoms_phonopy = PhonopyAtoms(symbols=prim.get_chemical_symbols(),
                                scaled_positions=prim.get_scaled_positions(),
                                cell=prim.cell)
   #2D or 3D calculation
   if nat_dim==3:
     #3D calc
     multiplier=np.array([[cellsize,0,0],[0,cellsize,0],[0,0,cellsize]])
     meshnu=[mesh_ph3, mesh_ph3, mesh_ph3]
   else:
     #2D calc
     multiplier=np.array([[cellsize,0,0],[0,cellsize,0],[0,0,1]])
     meshnu=[mesh_ph3, mesh_ph3, 1]
   if world.rank == 0:

      # read force constants from hdf5
      fc3 = read_fc3_from_hdf5(filename=fc3n)
      fc2 = read_fc2_from_hdf5(filename=fc2n)

      #create phono3py object
      ph3=Phono3py(atoms_phonopy, supercell_matrix=multiplier, primitive_matrix=None)
      ph3.mesh_numbers = meshnu
      ph3.set_fc3(fc3)
      ph3.set_fc2(fc2)

      #run thermal conductivity calculation
      ph3.set_phph_interaction()
      ph3.run_thermal_conductivity(temperatures=range(t1, t2, tstep), boundary_mfp=1e6, write_kappa=True)


@command('asr.anharmonic_phonons')
@option('--atoms', type=str, default='structure.json')
@option("--cellsize", help="supercell multiplication for hiphive", type=int)
@option("--calculator", help="calculator type. DFT is the default", type=str)
@option("--rattle", help="rattle standard hiphive", type=float)
@option("--cut1", help="cutoff 2nd", type=float)
@option("--cut2", help="cutoff 3rd", type=float)
@option("--cut3", help="cutoff 4th", type=float)
@option("--nat_dim", help="spatial dimension number: 2D or 3D calculation", type=int)
@option("--mindistance", help="minimum distance hiphive", type=float)
@option("--number_structures", help="no. of structures rattle hiphive", type=int)
@option("--mesh_ph3", help="phono3py mesh", type=int)
@option("--t1", help="first temperature for thermal conductivity calculation", type=int)
@option("--t2", help="last temperature for thermal conductivity calculation", type=int)
@option("--tstep", help=" temperature step for thermal conductivity calculation", type=int)

def main(atoms: Atoms, cellsize: int=5,calculator: str='DFT', rattle: float=0.03, nat_dim: int=2,cut1: float=6.0, cut2: float=5.0, cut3: float=4.0, mindistance: float=2.3, number_structures: int=10,mesh_ph3: int=10,t1=300,t2=301,tstep=1):

   fc2n='fc2.hdf5'
   fc3n='fc3.hdf5'

   #call the two main functions
   
   hiphive_fc23(atoms,cellsize,number_structures, rattle, mindistance, nat_dim, cut1, cut2, cut3, fc2n, fc3n, calculator)
   phono3py_lifetime(atoms, cellsize, nat_dim, mesh_ph3, fc2n, fc3n, t1, t2, tstep)

   # read the hdf5 file with the rta results

   if nat_dim==3:
     #3D calc
     filename='kappa-m'+str(mesh_ph3)+str(mesh_ph3)+str(mesh_ph3)+'.hdf5'
   else:
     #2D calc
     filename='kappa-m'+str(mesh_ph3)+str(mesh_ph3)+str(1)+'.hdf5'

   f = h5py.File(filename,'r')
   temperatures = f['temperature'][:]
   frequency = f['frequency'][:]
   gamma = f['gamma'][:]
   kappa = f['kappa'][:]

   #write results to json file

   results = {
     "temperatures": temperatures,
     "frequency": frequency,
     "gamma": gamma,
      "kappa": kappa,
   }
   return results

   #PLOT LIFETIMES
   ms = 4;fs = 16
   plt.figure()
   plt.plot(frequency.flatten(), gamma[0].flatten(), 'o', ms=ms)
   plt.xlabel('Frequency (THz)', fontsize=fs)
   plt.ylabel('$\Gamma$ (THz)', fontsize=fs)
   plt.xlim(left=0)
   plt.ylim(bottom=0)
   plt.title('T={:d}K'.format(int(temperatures[0])))
   plt.gca().tick_params(labelsize=fs)
   plt.tight_layout()
   plt.savefig('lifetime.pdf')
   
if __name__ == "__main__":
    main.cli()
