import typing
from asr.core import command, ASRResult, prepare_result, option, read_json
from ase.calculators.calculator import get_calculator_class
import numpy as np
from ase.io import read
from pathlib import Path

def calc_vdw(folder):
    Calculator = get_calculator_class("dftd3")
    calc = Calculator()

    atoms = read(f"{folder}/structure.json")
    atoms.set_calculator(calc)

    e = atoms.get_potential_energy()
    
    return e


def cell_area(atoms):
    """Calculate the area of a unit cell.

    Assumes system is 2D.
    Assumes the non-periodic direction is in the z-direciton.
    """
    cell = atoms.cell
    
    a1 = cell[0]
    a2 = cell[1]
    
    return np.linalg.norm(np.cross(a1, a2))

def get_descriptor():
    p = Path('.')
    return [x for x in str(p.absolute()).split("/") if x != ""][-1]


@prepare_result
class Result(ASRResult):
    """Container for bilayer binding energy."""
    
    binding_energy: float
    interlayer_distance: float
    bilayer_id: str

    key_descriptions = dict(binding_energy='Binding energy [eV/Ang^2]',
                            interlayer_distance='IL distance [Ang]',
                            bilayer_id='Str describing the stacking configuration')



def get_IL_distance(atoms, h):
    """Calculate IL distance."""

    layer_width = np.max(atoms.positions[:, 2]) - np.min(atoms.positions[:, 2])

    dist = h - layer_width
    assert dist > 0, f'The distance was not postive: {dist}'
    return dist


@command(module='asr.bilayer_binding',
         dependencies=['asr.relax_bilayer'],
         returns=Result)
def main() -> Result:
    """Calculate the bilayer binding energy."""

    d = read_json('results-asr.relax_bilayer.json')
    bilayer_energy = d['energy']
    height = d['optimal_height']

    monolayer = read('../structure.json')
    # Calculate monolayer energy with vdw correction
    vdw_corr = calc_vdw('../')
    monolayer_energy = monolayer.get_potential_energy() + vdw_corr

    binding = (monolayer_energy * 2 - bilayer_energy) / cell_area(monolayer)

    return Result.fromdata(binding_energy=binding,
                           interlayer_distance=get_IL_distance(monolayer, height),
                           bilayer_id=get_descriptor())
                           


