from asr.core import command, option, DictStr, ASRResult, prepare_result
from typing import List
import numpy as np
from os import path


def findOrthoNN(kpts: List[float], pbc: List[bool], n: int = 2):
    # Warning, might not find inversion symmetric points if k-points are not symmetric
    from scipy.spatial import cKDTree

    _, indices = cKDTree(kpts).query([0, 0, 0], k=len(kpts))
    indices = indices[1:]
    N = sum(pbc)
    orthNN = [[], [], []][:N]

    for direction in np.arange(N):
        orthoDirs = [(direction + 1) % 3, (direction + 2) % 3]
        i = 0
        for j, idx in enumerate(indices):
            if np.isclose(kpts[idx][orthoDirs[0]], 0) and \
               np.isclose(kpts[idx][orthoDirs[1]], 0):
                orthNN[direction].append(kpts[idx])
                i += 1
                if i == n:
                    break

    assert 0 not in np.shape(orthNN), 'No k-points in some orthogonal direction(s)'
    assert np.shape(orthNN)[-2] == n, 'Missing k-points in orthogonal directions'
    return np.round(np.array(orthNN), 16)


@prepare_result
class PreResult(ASRResult):
    Q: np.ndarray
    key_descriptions = {"Q": "nearest neighbour orthogonal q-vectors"}


@command(module='asr.dmi',
         resources='40:1h',
         requires=['structure.json'])
@option('-c', '--calculator', help='Calculator params.', type=DictStr())
def prepare_dmi(calculator: dict = {
        'name': 'gpaw',
        'mode': {'name': 'pw', 'ecut': 800, 'qspiral': [0, 0, 0]},
        'xc': 'LDA',
        'experimental': {'soc': False},
        'symmetry': 'off',
        'parallel': {'domain': 1, 'band': 1},
        'kpts': {'density': 22.0, 'gamma': True},
        'occupations': {'name': 'fermi-dirac',
                        'width': 0.05},
        'convergence': {'bands': 'CBM+3.0'},
        'nbands': '200%',
        'txt': 'gsq.txt',
        'charge': 0}) -> PreResult:
    from ase.io import read
    from ase.dft.kpoints import kpoint_convert, monkhorst_pack
    from ase.calculators.calculator import kpts2sizeandoffsets
    from asr.utils.spinspiral import spinspiral
    atoms = read('structure.json')
    
    size, offsets = kpts2sizeandoffsets(atoms=atoms, **calculator['kpts'])
    kpts_kc = monkhorst_pack(size) + offsets
    kpts_kv = kpoint_convert(atoms.cell, skpts_kc=kpts_kc)
    qpts_qv = findOrthoNN(kpts_kv, atoms.pbc, n=2)
    qpts_Rqc = 2 * kpoint_convert(atoms.cell, ckpts_kv=qpts_qv)

    for i, q_qc in enumerate(qpts_Rqc):
        for j, q_c in enumerate(q_qc):
            if not path.isfile(f'gsq{j}d{i}.gpw'):
                calculator['name'] = 'gpaw'
                calculator['txt'] = f'gsq{j}d{i}.txt'
                calculator['mode']['qspiral'] = q_c
                spinspiral(calculator, write_gpw=True)
    return PreResult.fromdata(Q=qpts_Rqc)


def webpanel(result, row, key_descriptions):
    from asr.database.browser import table, WebPanel
    dmi_table = table(row=result, title='Property', keys=['dmi'],
                      kd=key_descriptions)
    
    panel = {'title': 'Spin spirals',
             'columns': [[], [dmi_table]],
             'sort': 2}
    return [panel]

@prepare_result
class Result(ASRResult):
    Q: np.ndarray
    dmi: np.ndarray
    key_descriptions = {"Q": "nearest neighbour orthogonal q-vectors",
                        "dmi": "Components of projected soc in orthogonal directions"}
    formats = {"ase_webpanel": webpanel}

@command(module='asr.dmi',
         dependencies=['asr.dmi@prepare_dmi'],
         resources='1:1h',
         returns=Result)
def main() -> Result:
    from gpaw.spinorbit import soc_eigenstates
    from gpaw.occupations import create_occ_calc
    from ase.calculators.calculator import get_calculator_class
    from asr.core import read_json    
    name = 'gpaw'
    qpts_Rqc = read_json('results-asr.dmi@prepare_dmi.json')['Q']
    
    dmi_Rr = []
    for i, q_qc in enumerate(qpts_Rqc):
        en_q = []
        for j, q_c in enumerate(q_qc):
            calc = get_calculator_class(name)(f'gsq{j}d{i}.gpw')

            Ex, Ey, Ez = (soc_eigenstates(calc, projected=True,
                                          theta=theta, phi=phi).calculate_band_energy()
                          for theta, phi in [(90, 0), (90, 90), (0, 0)])
            en_q.append(np.array([Ex, Ey, Ez]))
        en_q = np.array(en_q)
        dmi_Rr.append(en_q[::2] - en_q[1::2])
    dmi_Rr = np.array(dmi_Rr)

    return Result.fromdata(Q=qpts_Rqc, dmi=dmi_Rr)


if __name__ == '__main__':
    main.cli()
