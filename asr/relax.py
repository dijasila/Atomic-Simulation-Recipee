"""Relax atomic structures.

By defaults read from "unrelaxed.json" from disk and relaxes
structures and saves the final relaxed structure in "structure.json".

The relax recipe has a couple of note-worthy features:

  - It automatically handles structures of any dimensionality
  - It tries to enforce symmetries
  - It continously checks after each step that no symmetries are broken,
    and raises an error if this happens.


The recipe also supports relaxing structure with vdW forces using DFTD3.
To install DFTD3 do

.. code-block:: console

   $ mkdir ~/DFTD3 && cd ~/DFTD3
   $ wget chemie.uni-bonn.de/pctc/mulliken-center/software/dft-d3/dftd3.tgz
   $ tar -zxf dftd3.tgz
   $ make
   $ echo 'export ASE_DFTD3_COMMAND=$HOME/DFTD3/dftd3' >> ~/.bashrc
   $ source ~/.bashrc

Examples
--------
Relax without using DFTD3

.. code-block:: console

   $ ase build -x diamond Si unrelaxed.json
   $ asr run "relax --nod3"

Relax using the LDA exchange-correlation functional

.. code-block:: console

   $ ase build -x diamond Si unrelaxed.json
   $ asr run "relax --calculator {'xc':'LDA',...}"



"""
import time
import typing
from math import sqrt
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import Trajectory, write
from ase.optimize.bfgs import BFGS
from ase.utils import IOContext
from ase.calculators.calculator import PropertyNotImplementedError

from asr.core import (ASRResult, AtomsFile, DictStr, command, option,
                      prepare_result)


class BrokenSymmetryError(Exception):
    pass


Uvalues = {}

# From [acs comb sci 2.011, 13, 383-390, Setyawan et al.]
UTM = {'Ti': 4.4, 'V': 2.7, 'Cr': 3.5, 'Mn': 4.0, 'Fe': 4.6,
       'Co': 5.0, 'Ni': 5.1, 'Cu': 4.0, 'Zn': 7.5, 'Ga': 3.9,
       'Nb': 2.1, 'Mo': 2.4, 'Tc': 2.7, 'Ru': 3.0, 'Rh': 3.3,
       'Pd': 3.6, 'Cd': 2.1, 'In': 1.9,
       'Ta': 2.0, 'W': 2.2, 'Re': 2.4, 'Os': 2.6, 'Ir': 2.8, 'Pt': 3.0}

for key, value in UTM.items():
    Uvalues[key] = ':d,{},0'.format(value)


def is_relax_done(atoms, fmax=0.01, smax=0.002,
                  smask=np.array([1, 1, 1, 1, 1, 1])):
    f = atoms.get_forces()
    if any(smask):
        s = atoms.get_stress() * smask
    else:
        s = np.zeros(6)
    done = (f**2).sum(1).max() <= fmax**2 and abs(s).max() <= smax

    return done


class SpgAtoms(Atoms):

    @classmethod
    def from_atoms(cls, atoms):
        # Due to technicalities we cannot mess with the __init__ constructor
        # -> therefore we make our own
        a = cls(atoms)
        a.set_symmetries([np.eye(3)], [[0, 0, 0]])
        return a

    def set_symmetries(self, symmetries, translations):
        self.t_sc = translations
        self.op_svv = [np.linalg.inv(self.cell).dot(op_cc.T).dot(self.cell) for
                       op_cc in symmetries]
        self.nsym = len(symmetries)
        tolerance = 1e-4
        spos_ac = self.get_scaled_positions()
        a_sa = []

        for op_cc, t_c in zip(symmetries, self.t_sc):
            symspos_ac = np.dot(spos_ac, op_cc.T) + t_c

            a_a = []
            for s_c in symspos_ac:
                diff_ac = spos_ac - s_c
                diff_ac -= np.round(diff_ac)
                mask_c = np.all(np.abs(diff_ac) < tolerance, axis=1)
                assert np.sum(mask_c) == 1, f'Bad symmetry, {mask_c}'
                ind = np.argwhere(mask_c)[0][0]
                assert ind not in a_a, f'Bad symmetry {ind}, {diff_ac}'
                a_a.append(ind)
            a_sa.append(a_a)

        self.a_sa = np.array(a_sa)

    def get_stress(self, voigt=True, *args, **kwargs):
        sigma0_vv = Atoms.get_stress(self, voigt=False, *args, **kwargs)

        sigma_vv = np.zeros((3, 3))
        for op_vv in self.op_svv:
            sigma_vv += np.dot(np.dot(op_vv, sigma0_vv), op_vv.T)
        sigma_vv /= self.nsym

        if voigt:
            return sigma_vv.flat[[0, 4, 8, 5, 2, 1]]

        return sigma_vv

    def get_forces(self, *args, **kwargs):
        f0_av = Atoms.get_forces(self, *args, **kwargs)
        f_av = np.zeros_like(f0_av)
        for map_a, op_vv in zip(self.a_sa, self.op_svv):
            for a1, a2 in enumerate(map_a):
                f_av[a2] += np.dot(f0_av[a1], op_vv)
        f_av /= self.nsym
        return f_av


class myBFGS(BFGS):
    def log(self, forces=None, stress=None):
        # We may have a cell filter; we want to get forces/stress
        # but not with the filter.  So get the real atoms:
        real_images = list(self.atoms.iterimages())
        assert len(real_images) == 1
        real_atoms = real_images[0]

        if forces is None:
            forces = real_atoms.get_forces()
        if stress is None:
            stress = real_atoms.calc.get_property(
                'stress', real_atoms, allow_calculation=False)
            if stress is None:
                # This is a lie, but we don't want to fix the
                # subsequent code.
                stress = np.zeros(6)

        fmax = sqrt((forces**2).sum(axis=1).max())
        smax = abs(stress).max()
        e = self.atoms.get_potential_energy(
            force_consistent=False)
        T = time.localtime()
        if self.logfile is not None:
            name = self.__class__.__name__
            if self.nsteps == 0:
                self.logfile.write(' ' * len(name)
                                   + '  {:<4} {:<8} {:<10} '.format('Step',
                                                                    'Time',
                                                                    'Energy')
                                   + '{:<10} {:<10}\n'.format('fmax',
                                                              'smax'))
                if False:
                    self.logfile.write(
                        '*Force-consistent energies used in optimization.\n')
            fc = '*' if False else ''
            self.logfile.write(f'{name}: {self.nsteps:<4} '
                               f'{T[3]:02d}:{T[4]:02d}:{T[5]:02d} '
                               f'{e:<10.6f}{fc} {fmax:<10.4f} {smax:<10.4f}\n')
            self.logfile.flush()


def relax(atoms, tmp_atoms_file,
          logfile, trajectory, emin=-np.inf, smask=None, dftd3=True,
          fixcell=False, allow_symmetry_breaking=False, dft=None,
          fmax=0.01, enforce_symmetry=False):

    if dftd3:
        from ase.calculators.dftd3 import DFTD3

    nd = sum(atoms.pbc)
    if smask is None:
        if fixcell:
            smask = [0, 0, 0, 0, 0, 0]
        elif nd == 3:
            smask = [1, 1, 1, 1, 1, 1]
        elif nd == 2:
            smask = [1, 1, 0, 0, 0, 1]
        else:
            assert atoms.pbc[2], "1D periodic axis should be the last one."
            smask = [0, 0, 1, 0, 0, 0]

    from asr.utils.symmetry import atoms2symmetry
    dataset = atoms2symmetry(atoms,
                             tolerance=1e-3,
                             angle_tolerance=0.1).dataset
    spgname = dataset['international']
    number = dataset['number']
    nsym = len(dataset['rotations'])
    atoms = SpgAtoms.from_atoms(atoms)
    if enforce_symmetry:
        atoms.set_symmetries(symmetries=dataset['rotations'],
                             translations=dataset['translations'])
    if dftd3:
        calc = DFTD3(dft=dft)
    else:
        calc = dft
    atoms.calc = calc

    # We are fixing atom=0 to reduce computational effort
    from ase.constraints import ExpCellFilter
    if fixcell:
        cellfilter = atoms
    else:
        cellfilter = ExpCellFilter(atoms, mask=smask)

    with myBFGS(cellfilter,
                logfile=logfile,
                trajectory=trajectory) as opt:

        # fmax=0 here because we have implemented our own convergence criteria
        for _ in opt.irun(fmax=0):
            # Check that the symmetry has not been broken
            newdataset = atoms2symmetry(atoms,
                                        tolerance=1e-3,
                                        angle_tolerance=0.1).dataset

            spgname2 = newdataset['international']
            number2 = newdataset['number']
            nsym2 = len(newdataset['rotations'])
            msg = (f'The initial spacegroup was {spgname} {number} '
                   f'but it changed to {spgname2} {number2} during '
                   'the relaxation.')
            if (not allow_symmetry_breaking
               and number != number2 and nsym > nsym2):
                # Log the last step
                opt.log()
                opt.call_observers()
                errmsg = 'The symmetry was broken during the relaxation! ' + msg
                raise BrokenSymmetryError(errmsg)
            elif number != number2:
                print('Not an error: The spacegroup has changed during relaxation. '
                      + msg)
                spgname = spgname2
                number = number2
                nsym = nsym2
                if enforce_symmetry:
                    atoms.set_symmetries(
                        symmetries=newdataset['rotations'],
                        translations=newdataset['translations'])

            if is_relax_done(atoms, fmax=fmax, smax=0.002, smask=smask):
                opt.log()
                opt.call_observers()
                break

    return atoms


def set_initial_magnetic_moments(atoms):
    atoms.set_initial_magnetic_moments(np.ones(len(atoms), float))


@prepare_result
class Result(ASRResult):
    """Result class for :py:func:`asr.relax.main`."""

    version: int = 0

    atoms: Atoms
    images: typing.List[Atoms]
    etot: float
    edft: float
    spos: np.ndarray
    symbols: typing.List[str]
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    key_descriptions = \
        {'atoms': 'Relaxed atomic structure.',
         'images': 'Path taken when relaxing structure.',
         'etot': 'Total energy [eV]',
         'edft': 'DFT total energy [eV]',
         'spos': 'Array: Scaled positions',
         'symbols': 'Array: Chemical symbols',
         'a': 'Cell parameter a [Å]',
         'b': 'Cell parameter b [Å]',
         'c': 'Cell parameter c [Å]',
         'alpha': 'Cell parameter alpha [deg]',
         'beta': 'Cell parameter beta [deg]',
         'gamma': 'Cell parameter gamma [deg]'}


@command('asr.relax',
         creates=['structure.json'],
         returns=Result)
@option('-a', '--atoms', help='Atoms to be relaxed.',
        type=AtomsFile(), default='unrelaxed.json')
@option('--tmp-atoms', help='File containing recent progress.',
        type=AtomsFile(must_exist=False), default='relax.traj')
@option('--tmp-atoms-file', help='File to store snapshots of relaxation.',
        default='relax.traj', type=str)
@option('-c', '--calculator', help='Calculator and its parameters.',
        type=DictStr())
@option('--d3/--nod3', help='Relax with vdW D3.', is_flag=True)
@option('--fixcell/--dont-fixcell',
        help="Don't relax stresses.",
        is_flag=True)
@option('--allow-symmetry-breaking/--dont-allow-symmetry-breaking',
        help='Allow symmetries to be broken during relaxation.',
        is_flag=True)
@option('--fmax', help='Maximum force allowed.', type=float)
@option('--enforce-symmetry/--dont-enforce-symmetry',
        help='Symmetrize forces and stresses.', is_flag=True)
def main(atoms: Atoms,
         calculator: dict = {'name': 'gpaw',
                             'mode': {'name': 'pw', 'ecut': 800},
                             'xc': 'PBE',
                             'kpts': {'density': 6.0, 'gamma': True},
                             'basis': 'dzp',
                             'symmetry': {'symmorphic': False},
                             'convergence': {'forces': 1e-4},
                             'txt': 'relax.txt',
                             'occupations': {'name': 'fermi-dirac',
                                             'width': 0.05},
                             'charge': 0},
         tmp_atoms: typing.Optional[Atoms] = None,
         tmp_atoms_file: str = 'relax.traj',
         d3: bool = False,
         fixcell: bool = False,
         allow_symmetry_breaking: bool = False,
         fmax: float = 0.01,
         enforce_symmetry: bool = True) -> Result:
    """Relax atomic positions and unit cell.

    The relaxed structure is saved to `structure.json` which can be processed
    by other recipes.

    Parameters
    ----------
    atoms
        Atomic structure to relax.
    calculator
        Calculator dictionary description.
    tmp_atoms
        Atoms from a restarted calculation.
    tmp_atoms_file
        Filename to save relaxed trajectory in.
    d3
        Relax using DFTD3.
    fixcell
        Fix cell when relaxing, thus only relaxing atomic positions.
    allow_symmetry_breaking
        Allow structure to break symmetry.
    fmax
        Maximum force tolerance.
    enforce_symmetry
        Enforce symmetries. When enabled, the atomic structure, forces and
        stresses will be symmetrized at each step of the relaxation.

    """
    from ase.calculators.calculator import get_calculator_class

    if tmp_atoms is not None:
        atoms = tmp_atoms

    atoms = atoms.copy()
    if atoms.has('initial_magmoms'):
        initially_spinpol = any(atoms.get_initial_magnetic_moments())
    else:
        # We don't know whether the system is spin polarized,
        # so we must assume it is.
        initially_spinpol = True
        set_initial_magnetic_moments(atoms)

    calculatorname = calculator.pop('name')
    Calculator = get_calculator_class(calculatorname)

    # Some calculator specific parameters
    nd = sum(atoms.pbc)
    if calculatorname == 'gpaw':
        if 'kpts' in calculator:
            from ase.calculators.calculator import kpts2kpts
            if 'density' in calculator['kpts']:
                kpts = kpts2kpts(calculator['kpts'], atoms=atoms)
                calculator['kpts'] = kpts
        if nd == 2:
            assert not atoms.get_pbc()[2], \
                ('The third unit cell axis should be aperiodic for '
                 'a 2D material!')
            calculator['poissonsolver'] = {'dipolelayer': 'xy'}

    def do_relax():
        return relax(atoms, tmp_atoms_file=tmp_atoms_file, dftd3=d3,
                     fixcell=fixcell,
                     logfile=logfile,
                     trajectory=trajectory,
                     allow_symmetry_breaking=allow_symmetry_breaking,
                     dft=calc, fmax=fmax, enforce_symmetry=enforce_symmetry)

    # Previously the relax recipe would open the text file twice and
    # overwrite itself, except the files wouldn't be flushed at the
    # right time so actually both were likely flushed simultaneously,
    # leading to unreliable logfiles.
    #
    # We want a new text file only if the relaxation starts from scratch,
    # and we prefer to open the file only once.
    #
    # Also, since we don't trust calc.set() etc., we create two different
    # calculators instead of reusing the same one.
    #
    # Turns out the ASE paropen() implementation does not recognize
    # the 'a' flag, so we have to roll our own.
    txt = calculator.pop('txt', '-')
    if tmp_atoms is None:
        open_mode = 'w'
    else:
        open_mode = 'a'

    logfile = Path(tmp_atoms_file).with_suffix('.log')

    with IOContext() as io:
        from ase.parallel import world as comm
        # XXX Not so nice to have special cases
        if calculatorname == 'gpaw':
            calculator['txt'] = io.openfile(txt, mode=open_mode,
                                            comm=comm)
        logfile = io.openfile(logfile, mode=open_mode, comm=comm)
        trajectory = io.closelater(Trajectory(tmp_atoms_file, mode=open_mode))

        # TODO: Perform actual GPAW computations in a separate process.
        # This should simplify the hacky IO handling here by forcing
        # proper GC, flushing and closing in that process.

        calc = Calculator(**calculator)

        atoms = do_relax()

        # If the maximum magnetic moment on all atoms is big then
        if initially_spinpol:
            magmoms = atoms.get_magnetic_moments()

            if not abs(magmoms).max() > 0.1:
                atoms.set_initial_magnetic_moments([0] * len(atoms))
                atoms.calc = calc
                atoms = do_relax()

        # This is buggy for systems that are close to the limit
        # of spinpolarizedness.  In stiffness recipe some calculations
        # may come out spinpolarized and others not, causing a small
        # jump in energy and botched derivatives.  It would be better
        # for the algorithm to strictly respect the input parameters
        # ("spinpol", "spinpaired", or "guess")

        edft = calc.get_potential_energy(atoms)
        etot = atoms.get_potential_energy()

        # If stress is provided by the calculator (e.g. PW mode) and we
        # didn't use stress, then nevertheless we want to calculate it because
        # the stiffness recipe wants it.  Also, all the existing results
        # have stress.
        try:
            atoms.get_stress()
        except PropertyNotImplementedError:
            pass

        if calculatorname == 'gpaw':
            # GPAW will have calc.close() soon.
            # Until then, we abuse __del__() which happens to
            # be the same currently.
            # If we didn't do this, then the txt file will be closed
            # before timings are written which is bad.
            #
            # (Also, when testing we do not always have __del__.)
            if hasattr(calc, '__del__'):
                calc.__del__()

    write('structure.json', atoms)

    cellpar = atoms.cell.cellpar()

    with Trajectory(tmp_atoms_file, 'r') as trajectory:
        images = list(trajectory)

    return Result.fromdata(
        atoms=atoms.copy(),
        etot=etot,
        edft=edft,
        a=cellpar[0],
        b=cellpar[1],
        c=cellpar[2],
        alpha=cellpar[3],
        beta=cellpar[4],
        gamma=cellpar[5],
        spos=atoms.get_scaled_positions(),
        symbols=atoms.get_chemical_symbols(),
        images=images
    )


if __name__ == '__main__':
    main.cli()
