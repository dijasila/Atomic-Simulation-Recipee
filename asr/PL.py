from ase.io import read
from ase.io.trajectory import Trajectory
import  csv
import pandas as pd
import phonopy
import sys
import ase.units as units
from math import sqrt, pi
import matplotlib.pyplot as plt
import numpy as np
from asr.core import write_json, command, option, ASRResult, read_json, prepare_result
from asr.lineshape import Lineshape

@prepare_result
class PLResults(ASRResult):
    """Container for excited results."""

    zpl: float
    delta_Q: float

    key_descriptions = dict(
        zpl='Zero phonon line energy.',
        delta_Q='Displacement from the ground state.')

@command('asr.PL')
@option('--excitation', type=str)
def main(excitation: str = 'alpha' or 'beta') -> PLResults:
    ground = read(f'excited_{excitation}/structure.json')
    excited = read(f'structure.json')
    phonon = phonopy.load(f'phonopy_params.yaml')
    ls = Lineshape(ground, excited, phonon, sigma=9e-3, gamma=9e-3, delta_t=0.1)
    delta_Q, _ = ls.get_delta_Q()
    print(delta_Q)
    s, _ = ls.get_partial_hr()
    s0 = s.sum()
    ls.get_info()
    print(s0)
    fig, ax = plt.subplots()
    S = ls.get_elph_function()
    Excited = Trajectory(f'excited_{excitation}/excited.traj')[-1]
    Ground = Trajectory(f'excited_{excitation}/ground.traj')[-1]
    ZPL = Excited.get_potential_energy() - Ground.get_potential_energy()
    ls.plot_spectral_function(ax=ax, ZPL=ZPL, filename=f'Emission_{excitation}')
    # define the 1D coordinate
    return PLResults.fromdata(
        delta_Q=delta_Q,
        zpl=ZPL)

def webpanel(result, row, key_descriptions):
    from asr.browser import fig, table

    if 'something' not in row.data:
        return None, []

    table1 = table(row,
                   'Property',
                   ['something'],
                   kd=key_descriptions)
    panel = ('Title',
             [[fig('something.png'), table1]])
    things = [(create_plot, ['something.png'])]
    return panel, things


def create_plot(row, fname):
    import matplotlib.pyplot as plt

    data = row.data.something
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(data.things)
    plt.savefig(fname)


group = 'property'
creates = ['something.json']  # what files are created
dependencies = []  # no dependencies
resources = '1:10m'  # 1 core for 10 minutes
diskspace = 0  # how much diskspace is used
restart = 0  # how many times to restart

if __name__ == '__main__':
    main()
