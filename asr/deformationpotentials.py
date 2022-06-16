"""Deformation potentials."""
import numpy as np
import typing

from asr.core import command, option, ASRResult, prepare_result
from asr.gs import vacuumlevels
from asr.utils.gpw2eigs import calc2eigs
from asr.database.browser import (
    table, fig,
    entry_parameter_description,
    describe_entry, WebPanel,
    make_panel_description
)


description_text = """\
The deformation potentials represent the energy shifts of the
bottom of the conduction band (CB) and the top of the valence band
(VB) at a given k-point, under an applied strain.

This panel shows the VB and CB deformation potentials at the
high-symmetry k-points, subdivided into the different strain
components.

In case one or both the band extrema of the material are not found
at any of the special points, the corresponding k-point(s) are
added to the list as `VBM` and/or `CBM` (indirect-gap materials)
or `VBM/CBM` (direct-gap materials) """


panel_description = make_panel_description(
    description_text,
    articles=['C2DB'],
)


def is_in_list(element, lst):
    for obj in lst:
        if np.allclose(element, obj, rtol=0.05, atol=0):
            return True
    return False


def get_relevant_kpts(atoms, calc):
    """
    Obtain the high-symmetry k-points.

    If the band edges of the unstrained material are found away
    from any of the special points, the corresponding
    k-points will be added to the list as 'VBM' and 'CBM'
    or 'VBM/CBM' (direct-gap materials)
    """
    from ase.dft.bandgap import bandgap

    # XXX investigate issue with special points falling outside cell vectors
    specpts = atoms.cell.bandpath(pbc=atoms.pbc, npoints=0).special_points

    _, ivbm, icbm = bandgap(calc, output=None)
    if ivbm[1] == icbm[1]:
        kdict = {
                'VBM/CBM': ivbm
        }
    else:
        kdict = {
                'VBM': ivbm,
                'CBM': icbm
        }

    ibz_kpoints = calc.get_ibz_k_points()
    for label, i_kpt in kdict.items():
        kpt = ibz_kpoints[i_kpt[1]]
        if not is_in_list(kpt, specpts.values()):
            specpts[label] = kpt

    return specpts


def get_edges(calc, atoms, soc):
    """Obtain the edges states at the different k-points

    Returns, for each k-point included in the calculation,
    the top eigenvalue of the valence band and the bottom
    eigenvalue of the conduction band.
    """
    all_eigs, efermi = calc2eigs(calc, soc=soc)
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


soclabels = {
        'deformation_potentials_nosoc': False,
        'deformation_potentials_soc': True
}

ijlabels = {
        (0, 0): 'xx',
        (1, 1): 'yy',
        (2, 2): 'zz',
        (0, 1): 'xy',
        (0, 2): 'xz',
        (1, 2): 'yz',
}


@prepare_result
class Result(ASRResult):

    deformation_potentials_nosoc: typing.Dict[str, float]
    deformation_potentials_soc: typing.Dict[str, float]
    kpts: typing.Union[list, typing.Dict[str, float]]

    key_descriptions = {
            'deformation_potentials_nosoc': 'Deformation potentials under different types \
             of deformations (xx, yy, zz, yz, xz, xy) at each k-point, without SOC',
            'deformation_potentials_soc': 'Deformation potentials under different \
             applied strains (xx, yy, zz, yz, xz, xy) at each k-point, with SOC',
            'kpts': 'k-points at which deformation potentials were calculated'
    }

    formats = {"ase_webpanel": webpanel}


@command('asr.deformationpotentials',
         returns=Result)
@option('-s', '--percent-strain', help='Strain percentage', type=float)
@option('--all-ibz', is_flag=True,
        help="Calculate deformation potentials at all the irreducible Brillouin zone k-points.", type=bool)
def main(strain_percent=1.0, all_ibz=False) -> Result:
    """Calculate deformation potentials.

    Calculate the deformation potentials both with and without spin orbit
    coupling, for both the conduction band and the valence band, and return as
    a dictionary.

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
    #TODO complete documentation

    from gpaw import GPAW
    from ase.io import read
    from asr.setup.strains import (get_strained_folder_name,
                                   get_relevant_strains)
    from asr.core import read_json
    from ase.visualize import view

    atoms = read('structure.json')
    calc = GPAW('gs.gpw')
    results = {}

    if all_ibz:
        kpts = kptlabels = calc.get_ibz_k_points()
        results['kpts'] = kpts
    else:
        kptdict = get_relevant_kpts(atoms, calc)
        kptlabels = kptdict.values()
        results['kpts'] = kptdict

    strains = [-abs(strain_percent), abs(strain_percent)]
    results.update({socflag: {kpt: {} for kpt in kptlabels}
                    for socflag in soclabels})

    for socflag, soc in soclabels.items():
        for ij in get_relevant_strains(atoms.pbc):
            straincomp = ijlabels.get(ij)
            edges_ij = []
            for strain in strains:
                folder = get_strained_folder_name(strain, ij[0], ij[1])
                strainedatoms = read(f'{folder}/structure.json')
                gpw = GPAW(f'{folder}/gs.gpw').fixed_density(
                        kpts=list(kptdct),
                        symmetry='off',
                        txt=None
                )
                gpw.get_potential_energy()
                edges_ij.append(get_edges(gpw, strainedatoms, soc))

            # Actual calculation of the deformation potentials
            defpots_ij = np.squeeze(
                np.diff(edges_ij, axis=0) / (np.ptp(strains) * 0.01))

            for dp, kpt in zip(defpots_ij, kptlabels):
                results[socflag][kpt][straincomp] = {
                        'VB': dp[0],
                        'CB': dp[1]
                }

    return results


if __name__ == '__main__':
    main.cli()
