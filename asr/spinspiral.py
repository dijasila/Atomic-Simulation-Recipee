from asr.core import command, option, ASRResult, prepare_result, DictStr
from asr.utils.spinspiral import rotate_magmoms, get_magmom_bands, col_to_ncolx, \
    get_collinear_magmoms, get_noncollinear_magmoms, get_spiral_bandpath, \
    true_magnetic_atoms
from asr.collect_spiral import SpinSpiralCalculation
from ase.io import read
from ase.parallel import world
from typing import Union
import numpy as np
from os import path


def spinspiral(calculator: dict = {
        'mode': {'name': 'pw', 'ecut': 800, 'qspiral': [0, 0, 0]},
        'xc': 'LDA',
        'soc': False,
        'symmetry': 'off',
        'parallel': {'domain': 1, 'band': 1},
        'kpts': {'density': 12.0, 'gamma': True},
        'occupations': {'name': 'fermi-dirac',
                        'width': 0.05},
        'convergence': {'bands': 'CBM+3.0'},
        'txt': 'gsq.txt',
        'charge': 0},
        atoms=None, rotation_model='q.a',
        write_gpw: bool = True) -> dict:
    """Calculate the groundstate of a given spin spiral vector q_c"""
    from gpaw.new.ase_interface import GPAW
    from ase.dft.bandgap import bandgap

    if atoms is None:
        atoms = read('structure.json')

    try:
        gpwfile = calculator['txt'].replace('.txt', '.gpw')
        restart = path.isfile(gpwfile)
    except KeyError:
        gpwfile = 'gsq.gpw'
        restart = False

    q_c = calculator['mode']['qspiral']  # spiral vector must be provided
    magmoms = calculator.get("magmoms", None)
    if magmoms is None:
        magmoms = get_noncollinear_magmoms(atoms, calculator)

    if rotation_model is not None:
        magmoms = rotate_magmoms(atoms, magmoms, q_c, model=rotation_model)

    # Mandatory spin spiral parameters
    calculator["xc"] = 'LDA'
    calculator['soc'] = False
    calculator['magmoms'] = magmoms
    calculator["symmetry"] = 'off'
    calculator["parallel"] = {'domain': 1, 'band': 1}

    if restart:
        calc = GPAW(gpwfile)
    else:
        calc = GPAW(**calculator)

    # atoms.center(vacuum=6.0, axis=2)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    if write_gpw and not restart:
        atoms.calc.write(gpwfile)
    totmom_v, magmom_av = calc.dft.state.density.calculate_magnetic_moments()
    gap, _, _ = bandgap(calc)

    return {'energy': energy, 'totmom_v': totmom_v,
            'magmom_av': magmom_av, 'gap': gap}


@prepare_result
class Result(ASRResult):
    path: np.ndarray
    energies: np.ndarray
    key_descriptions = {"path": "List of Spin spiral vectors",
                        "energies": "Potential energy [eV]"}


@command(module='asr.spinspiral',
         requires=['structure.json'],
         returns=Result)
@option('-c', '--calculator', help='Calculator params.', type=DictStr())
@option('--q_path', help='Spin spiral high symmetry path eg. "GKMG"', type=str)
@option('--qpts', help='Number of sample points', type=int)
@option('--qdens', help='Density of sample points', type=float)
@option('--rotation_model', help='Rotate initial magmoms by given model', type=str)
@option('--clean_up', help='Remove gpw files after execution', type=bool)
@option('--eps', help='Bandpath symmetry threshold', type=float)
def main(calculator: dict = dict(mode={'name': 'pw', 'ecut': 800},
                                 kpts={'density': 6.0, 'gamma': True}),
         q_path: Union[str, None] = None,
         qpts: int = None, qdens: float = None,
         rotation_model: Union[str, None] = 'q.a', clean_up: bool = False,
         eps: float = None) -> Result:

    q_path, energies = _main(calculator, q_path, qpts, qdens,
                             rotation_model, clean_up, eps, submit=False)

    energies = np.asarray(energies)
    emin_idx = np.argmin(energies)
    if clean_up:
        import os
        from glob import glob
        gpw_list = glob('*.gpw')
        for gpw in gpw_list:
            if int(gpw[3:-6]) != emin_idx:
                if world.rank == 0:
                    os.remove(gpw)

    return Result.fromdata(path=q_path, energies=energies)


def _main(calculator: dict = dict(mode={'name': 'pw', 'ecut': 800},
                                  kpts={'density': 6.0, 'gamma': True}),
          q_path: Union[str, None] = None,
          qpts: int = None, qdens: float = None,
          rotation_model: Union[str, None] = 'q.a', clean_up: bool = False,
          eps: float = None,
          request: str = 'priority',
          submit: bool = False,
          cores: int = 24,
          tmax: str = '3h'):

    atoms = read('structure.json')

    q_path = get_spiral_bandpath(atoms=atoms, qdens=qdens, qpts=qpts,
                                 q_path=q_path, eps=eps)

    # Magnetic moments might be supplied in many different ways.
    # It might exist as calculator dictionary input in non-collinear
    # calculations.
    # Otherwise, in ASE atoms object there might exist pre-calculated
    # moments "magnetic moments" or "initial magnetic moments".
    # If none of these are provided, we guess m=1 on atoms with
    # d-orbital valence states.
    # Priority is: calculator dict > ASE calculated > ASE initialized > guess
    # Request keys: calculator, calculated, initial, guess
    col_magmoms = get_collinear_magmoms(atoms=atoms, request=request,
                                        calculator=calculator)

    # If we find an even number of magnetic moments in the unit cell,
    # we construct the possible collinear configurations
    arg = true_magnetic_atoms(atoms, col_magmoms)
    if sum(arg) > 0 and sum(arg) % 2 == 0:
        magmoms_bav = get_magmom_bands(arg, col_magmoms)
    else:
        magmoms_bav = [col_to_ncolx(col_magmoms)]

    energies = []
    for bidx, magmoms_av in enumerate(magmoms_bav):
        for qidx, qn_c in enumerate(q_path.kpts):
            calculator['mode']['qspiral'] = qn_c
            calculator['txt'] = f'gsq{qidx}b{bidx}.txt'
            calculator['magmoms'] = magmoms_av

            if submit:
                sscalc = submit_calculations(atoms, calculator, rotation_model,
                                             bidx, qidx, energies, cores=cores,
                                             tmax=tmax)
            else:
                if cannot_converge(qidx, bidx):
                    continue
                sscalc = execute_calculations(atoms, calculator, rotation_model,
                                              bidx, qidx, energies)

            if world.rank == 0:
                sscalc.save(f'datq{qidx}b{bidx}.json')

    energies = np.asarray(energies)

    return q_path, energies


def execute_calculations(atoms, calculator, rotation_model, bidx, qidx, energies):
    try:
        result = spinspiral(calculator=calculator,
                            atoms=atoms,
                            rotation_model=rotation_model)
        energies.append(result['energy'])
        sscalc = SpinSpiralCalculation([bidx, qidx],
                                       result['energy'],
                                       result['totmom_v'].tolist(),
                                       result['magmom_av'].tolist(),
                                       result['gap'])
    except Exception as e:
        print(f'Caught exception {e}')
        energies.append(0.0)
        sscalc = SpinSpiralCalculation([bidx, qidx], 0.0, [0, 0, 0],
                                       [[0, 0, 0]] * len(atoms), 0)
    return sscalc


def submit_calculations(atoms, calculator, rotation_model, bidx, qidx, energies,
                        cores=24, tmax='8h'):
    from myqueue.workflow import run
    r = run(function=spinspiral, name=f'gsq{qidx}b{bidx}',
            kwargs={'calculator': calculator,
                    'atoms': atoms,
                    'rotation_model': rotation_model},
            cores=cores, tmax=tmax)
    if r.done:
        energies.append(r.result['energy'])
        sscalc = SpinSpiralCalculation([bidx, qidx],
                                       r.result['energy'],
                                       r.result['totmom_v'].tolist(),
                                       r.result['magmom_av'].tolist(),
                                       r.result['gap'])
    else:
        sscalc = SpinSpiralCalculation([bidx, qidx], 0.0, [0, 0, 0],
                                       [[0, 0, 0]] * len(atoms), 0)
        energies.append(0.0)
    return sscalc


def cannot_converge(qidx, bidx):
    # IF the calculation has been run (gs.txt exist) but did not converge
    # (gs.gpw not exist) THEN raise exception UNLESS it did not finish (last calc)
    # (gsn.txt exist but gsn+1.txt does not)
    # Note, UNLESS imply negation in context of if-statement (and not)
    if path.isfile(f'gsq{qidx}b{bidx}.txt') and \
        not path.isfile(f'gsq{qidx}b{bidx}.gpw') and \
        not (path.isfile(f'gsq{qidx}b{bidx}.txt') and not
             path.isfile(f'gsq{qidx+1}b{bidx}.txt')):
        return True  # SFC finished but didn't converge
    return False
