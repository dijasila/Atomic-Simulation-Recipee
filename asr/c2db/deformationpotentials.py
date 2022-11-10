"""Deformation potentials."""
import numpy as np
import typing

from asr.core import command, option, ASRResult, prepare_result
from asr.utils.gpw2eigs import calc2eigs
from asr.database.browser import href, make_panel_description


description_text = """\
The deformation potentials represent the energy shifts of the
bottom of the conduction band (CB) and the top of the valence band
(VB) at a given k-point, under an applied strain.

The panel shows the VB and CB deformation potentials at the
high-symmetry k-points, subdivided into the different strain
components. If the VBM and/or the CBM fall at any of the
special points, an asterisk is added to the k-point name.
If they are found at any other k-point, they are added to the list
as 'k<sub>VBM</sub>' / 'k<sub>VBM</sub>' / 'k<sub>VBM/CBM</sub>'.

All the values are calculated with spin-orbit coupling.
"""


panel_description = make_panel_description(
    description_text,
    articles=[
        href("""Wiktor, J. and Pasquarello, A., 2016. Absolute deformation potentials
of two-dimensional materials. Physical Review B, 94(24), p.245411""",
             "https://doi.org/10.1103/PhysRevB.94.245411")
    ],
)


def get_relevant_kpts(atoms, vbm, cbm, ibz_kpoints):
    """Obtain the high-symmetry k-points.

    If the band edges of the unstrained material are found away
    from any of the special points, the corresponding
    k-points will be added to the list
    """
    ivbm = vbm[1]
    icbm = cbm[1]
    kvbm = ibz_kpoints[ivbm]
    kcbm = ibz_kpoints[icbm]
    if ivbm == icbm:
        spec = {
            'VBM CBM': kvbm
        }
    else:
        spec = {
            'VBM': kvbm,
            'CBM': kcbm
        }

    icell = atoms.cell.reciprocal()
    bp = atoms.cell.bandpath(pbc=atoms.pbc, npoints=0)
    kpts = bp.special_points

    kpoints = spec.copy()
    for lab1, kpt1 in kpts.items():
        # Matching special points fractional coordinates
        # between atoms.cell.bandpath and atoms.cell.reciprocal()
        spec_bp = np.asarray(kpt1)
        spec_abs = np.dot(spec_bp, bp.icell)
        kpt_new = np.dot(spec_abs, np.linalg.inv(icell))

        label = lab1
        for lab2, kpt2 in spec.items():
            if np.allclose(kpt_new, kpt2, rtol=0.1, atol=1e-8):
                label += f' {lab2}'
                kpoints.pop(lab2)
        kpoints[label] = kpt_new

    return kpoints


def get_table_row(kpt, band, data):
    row = []
    for comp in ['xx', 'yy', 'xy']:
        row.append(data[kpt][comp][band])
    return np.asarray(row)


def webpanel(result, row, key_descriptions):
    from asr.database.browser import matrixtable, describe_entry, WebPanel

    def get_basename(kpt):
        chunks = kpt.split(' ')
        if chunks[0] == 'G':
            return 'Γ'
        elif chunks[0] in ('VBM', 'CBM'):
            try:
                if chunks[1] in ('VBM', 'CBM'):
                    return 'k<sub>VBM/CBM</sub>'
            except IndexError:
                return f'k<sub>{chunks[0]}</sub>'
        else:
            return chunks[0]

    description = describe_entry('Deformation potentials', panel_description)
    defpots = result['deformation_potentials_soc'].copy()
    columnlabels = ['xx', 'yy', 'xy']

    dp_gap = defpots.pop('Band Gap')
    dp_list = []
    dp_labels = []
    add_to_bottom = []
    for kpt in defpots:
        name = get_basename(kpt)
        for band, edge in zip(['VB', 'CB'], ['VBM', 'CBM']):
            row = get_table_row(kpt, band, defpots)
            label = name + f' ({band})'
            if 'k' in label:
                add_to_bottom.append((label, row))
                continue
            if edge in kpt:
                label += ' *'
            dp_labels.append(label)
            dp_list.append(row)

    for label, row in add_to_bottom:
        dp_labels.append(label)
        dp_list.append(row)
    dp_labels.append('Band Gap')
    dp_list.append([dp_gap[comp] for comp in columnlabels])

    dp_table = matrixtable(
        dp_list,
        digits=2,
        title='D (eV)',
        columnlabels=columnlabels,
        rowlabels=dp_labels
    )
    panel = WebPanel(
        description,
        columns=[[dp_table]],
        sort=4
    )
    return [panel]


@prepare_result
class Result(ASRResult):
    kpts: np.ndarray
    deformation_potentials_soc: object
    deformation_potentials_nosoc: object

    deformation_potentials_nosoc: typing.Dict[str, float]
    deformation_potentials_soc: typing.Dict[str, float]
    kpts: typing.Union[list, typing.Dict[str, float]]

    key_descriptions = {
        'deformation_potentials_nosoc': (
            'Deformation potentials under different types of '
            'deformations (xx, yy, zz, yz, xz, xy) at each k-point, '
            'without SOC'),
        'deformation_potentials_soc': (
            'Deformation potentials under different applied strains '
            '(xx, yy, zz, yz, xz, xy) at each k-point, with SOC'),
        'kpts': 'k-points at which deformation potentials were calculated'
    }

    formats = {"ase_webpanel": webpanel}

    key_descriptions = dict(
        kpts='K-points',
        deformation_potentials_soc='Deformation potentials including SOC.',
        deformation_potentials_nosoc='Deformation potentials without SOC.')


soclabels = {'deformation_potentials_nosoc': False,
             'deformation_potentials_soc': True}

ijlabels = {
    (0, 0): 'xx',
    (1, 1): 'yy',
    (2, 2): 'zz',
    (0, 1): 'xy',
    (0, 2): 'xz',
    (1, 2): 'yz',
}


@command('asr.c2db.deformationpotentials')
@option('-s', '--strain', type=float,
        help='percent strain applied to the material along all components')
@option('--all-ibz', is_flag=True, type=bool,
        help=('Calculate deformation potentials at all '
              'the irreducible Brillouin zone k-points.'))
def main(strain: float = 1.0, all_ibz: bool = False) -> Result:
    """Calculate deformation potentials.

    Calculate the deformation potentials both with and without spin orbit
    coupling, for both the conduction band and the valence band, and return as
    a dictionary. The dictionary has the following structure:

    {'deformation_potentials_soc': {'kpt_1': {'xx': {'CB': <value>,
                                                     'VB': <value>},

                                              'yy': {...},
                                              'xy': {...}}

                                    'kpt_2': {...},
                                      ...
                                    'kpt_N': {...}},

     'deformation_potentials_nosoc': ...}

    Parameters
    ----------
    strain-percent: float
        Percent strain to apply, in both direction and for all
        the relevant strain components, to the current material.
    all-ibz: bool
        If True, calculate the deformation potentials at all
        the k-points in the irreducible Brillouin zone.
        Otherwise, just use the special points and the k-points
        where the edge states are found (if they are not already
        at one of the special points).
    """
    from gpaw import GPAW
    from ase.io import read
    from asr.gs import vacuumlevels
    from ase.dft.bandgap import bandgap

    atoms = read('structure.json')
    calc = GPAW('gs.gpw')
    gap, vbm, cbm = bandgap(calc, output=None)
    if gap == 0.0:
        print("""\
        Deformation potentials cannot be defined for metals! Terminating recipe...
        """)
        return None

    ibz = calc.get_ibz_k_points()
    if all_ibz:
        kpts = ibz
    else:
        kpts = get_relevant_kpts(atoms, vbm, cbm, ibz)

    def gpaw_get_edges(folder, kpts, soc):
        """Obtain the edge states at the different k-points.

        Returns, for each k-point included in the calculation,
        the top eigenvalue of the valence band and the bottom
        eigenvalue of the conduction band.
        """
        atoms = read(f'{folder}/structure.json')
        gpw = GPAW(f'{folder}/gs.gpw').fixed_density(
            kpts=kpts,
            symmetry='off',
            txt=None
        )
        gpw.get_potential_energy()
        all_eigs, efermi = calc2eigs(gpw, soc=soc)
        vac = vacuumlevels(atoms, calc)

        # This will take care of the spin polarization
        if not soc:
            all_eigs = np.hstack(all_eigs)

        edges = np.zeros((len(all_eigs), 2))
        for i, eigs_k in enumerate(all_eigs):
            vb = [eig for eig in eigs_k if eig - efermi < 0]
            cb = [eig for eig in eigs_k if eig - efermi > 0]
            edges[i, 0] = max(vb)
            edges[i, 1] = min(cb)
        return edges - vac.evacmean

    results = _main(atoms.pbc, kpts, gpaw_get_edges, strain)

    # Extract band gap deformation potentials
    for key in ['deformation_potentials_soc', 'deformation_potentials_nosoc']:
        edge_states = {}
        result = results[key]
        for kpt in result:
            if 'VBM' in kpt:
                edge_states['VBM'] = get_table_row(kpt, 'VB', result)
            if 'CBM' in kpt:
                edge_states['CBM'] = get_table_row(kpt, 'CB', result)
        dp_gap = edge_states['CBM'] - edge_states['VBM']
        results[key]['Band Gap'] = {
            key: comp for key, comp in zip(['xx', 'yy', 'xy'], dp_gap)
        }

    return results


def _main(pbc, kpts, get_edges, strain):
    from collections import OrderedDict
    from asr.setup.strains import (get_relevant_strains,
                                   get_strained_folder_name)
    results = {
        'kpts': kpts
    }

    if isinstance(kpts, dict):
        kptlabels = list(kpts)
        kpts = list(kpts.values())
    else:
        kptlabels = kpts

    # Initialize strains and deformation potentials results
    strains = [-abs(strain), abs(strain)]
    results.update({
        socstr: {kpt: OrderedDict() for kpt in kptlabels} for socstr in soclabels
    })

    # Navigate the directories containing the ground states of
    # the strained structures and extract the band edges
    for socstr, soc in soclabels.items():
        for ij in get_relevant_strains(pbc):
            straincomp = ijlabels[ij]
            edges_ij = []
            for strain in strains:
                folder = get_strained_folder_name(strain, ij[0], ij[1])
                edges = get_edges(folder, kpts, soc)
                edges_ij.append(edges)

            # Actual calculation of the deformation potentials
            defpots_ij = np.squeeze(
                np.diff(edges_ij, axis=0) / (np.ptp(strains) * 0.01)
            )

            for dp, kpt in zip(defpots_ij, kptlabels):
                results[socstr][kpt][straincomp] = {
                    'VB': dp[0],
                    'CB': dp[1]
                }

    return results


if __name__ == '__main__':
    main.cli()
