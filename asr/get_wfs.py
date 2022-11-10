import typing
import click
from pathlib import Path
from asr.core import command, option, ASRResult, prepare_result


@prepare_result
class WaveFunctionResult(ASRResult):
    """Container for results of specific wavefunction for one spin channel."""

    state: int
    spin: int
    energy: float

    key_descriptions: typing.Dict[str, str] = dict(
        state='State index.',
        spin='Spin index (0 or 1).',
        energy='Energy of the state (ref. to the vacuum level in 2D) [eV].')


@prepare_result
class Result(ASRResult):
    """Container for asr.get_wfs results."""

    wfs: typing.List[WaveFunctionResult]
    above_below: typing.Tuple[bool, bool]
    eref: float

    key_descriptions: typing.Dict[str, str] = dict(
        wfs='List of WaveFunctionResult objects for all states.',
        above_below='States within the gap above and below EF? '
                    '(ONLY for defect systems).',
        eref='Energy reference (vacuum level in 2D, 0 otherwise) [eV].')


@command(module='asr.get_wfs')
# requires=['gs.gpw', 'structure.json',
#           'results-asr.gs.json'],
# dependencies=['asr.gs@calculate', 'asr.gs'],
# resources='1:10m',
# returns=Result)
@option('--state', help='Specify state index that you want to '
        'write out. This option will not be used when '
        '"--get-gapstates" is used.', type=int)
@option('--erange', help='Specify the energy range (wrt. Fermi level) '
        'within which you want to write out the wavefunctions (lower '
        'threshold, upper threshold). This option will not be used when '
        '"--get-gapstates" is used.', nargs=2,
        type=click.Tuple([float, float]))
@option('--get-gapstates/-dont-get-gapstates', help='Save all wfs '
        'for states present inside the gap (only use this option for '
        'defect systems, where your folder structure has been set up '
        'with asr.setup.defects).', is_flag=True)
def main(state: int = 0,
         erange: typing.Tuple[float, float] = (0, 0),
         get_gapstates: bool = False) -> Result:
    """
    Perform fixed density calculation and write out wavefunctions.

    This recipe reads in an existing gs.gpw file, runs a fixed density
    calculation on it, and writes out wavefunctions for a state in
    "wf.{bandindex}_{spin}.cube" cube files. This state can either be
    a particular state given by the bandindex with the option "--state",
    or (for defect systems) be all of the states in the gap which will
    be evaluated with the option "--get-gapstates". Note, that when
    "--get-gapstates" is given, "--state" will be ignored.
    """
    from gpaw import restart
    from asr.core import read_json
    from asr.defect_symmetry import WFCubeFile

    # read in converged gs.gpw file and run fixed density calculation
    print('INFO: run fixed density calculation.')
    atoms, calc = restart('gs.gpw', txt='get_wfs.txt')
    calc = calc.fixed_density(kpts={'size': (1, 1, 1), 'gamma': True})
    print('INFO: ran fixed density calculation starting from gs.gpw.')

    # evaluate states in the gap (if '--get-gapstates' is active)
    ef = calc.get_fermi_level()
    if get_gapstates:
        print('INFO: evaluate gapstates.')
        states, above_below, eref = return_gapstates(calc)
    # get energy reference and convert given input state to correct format
    elif not get_gapstates:
        # for 2D systems, use the vacuum level as energy reference point
        if sum(atoms.pbc) == 3:
            eref = ef
        else:
            eref = read_json('results-asr.gs.json')['evac']
        # if no 'erange' is given, just use the input state to write out wfs
        if erange == (0, 0):
            states = [state]
        # otherwise, return all of the states in a particular energy range
        elif erange != (0, 0):
            evs = calc.get_eigenvalues()
            states = return_erange_states(evs, ef, erange)
        # specific to the defect project result (will be removed in 'master')
        above_below = (None, None)

    print(f'INFO: states to write to file: {states}.')

    # loop over all states and write the wavefunctions to file,
    # set up WaveFunctionResults
    wfs_results = []
    nspins = calc.get_number_of_spins()
    for state in states:
        for spin in range(nspins):
            wfs_result = get_wfs_results(calc, state, spin, eref)
            wf = calc.get_pseudo_wave_function(band=state, spin=spin)
            wfs_results.append(wfs_result)
            wfcubefile = WFCubeFile(spin=spin, band=state, wf_data=wf, calc=calc)
            wfcubefile.write_to_cubefile()

    return Result.fromdata(
        wfs=wfs_results,
        above_below=above_below,
        eref=eref)


def get_wfs_results(calc, state, spin, eref):
    """
    Return WaveFunctionResults for specific state, spin, energy reference.

    Write corresponding wavefunction cube file.
    """
    energy = calc.get_eigenvalues(spin=spin)[state] - eref

    return WaveFunctionResult.fromdata(
        state=state,
        spin=spin,
        energy=energy)


def get_reference_index(index, atoms):
    """Get index of atom furthest away from the atom i."""
    from ase.geometry import get_distances

    distances = []
    ref_index = None

    pos = atoms.get_positions()
    cell = atoms.get_cell()
    for i in range(len(atoms)):
        dist = get_distances(pos[i], pos[index],
                             cell=cell, pbc=True)[1][0, 0]
        distances.append(dist)

    for i, element in enumerate(distances):
        if element == max(distances):
            ref_index = i
            break

    return ref_index


def extract_atomic_potentials(calc_def, calc_pris, ref_index, is_vacancy):
    """Evaluate atomic potentials far away from the defect for pristine and defect."""
    struc_def = calc_def.atoms
    struc_pris = calc_pris.atoms

    pot_pris = calc_pris.get_atomic_electrostatic_potentials()[ref_index]
    if is_vacancy:
        def_index = ref_index - 1
    else:
        def_index = ref_index

    pot_def = calc_def.get_atomic_electrostatic_potentials()[def_index]

    # check whether chemical symbols of both reference atoms are equal
    if not (struc_def.symbols[def_index]
            == struc_pris.symbols[ref_index]):
        raise ValueError('chemical symbols of reference atoms '
                         'are not the same.')

    return pot_def, pot_pris


def return_gapstates(calc_def):
    """
    Evaluate states within the pristine band gap and return band indices.

    This function compares a defect calculation to a pristine one and
    evaluates the gap states by referencing both calculations to the
    electrostatic potential of an atom far away from the defect. Next, the
    VBM and CBM of the pristine system get projected onto the defect
    calculation and all defect states will be saved. Lastly, the absolute
    energy scale will be referenced back to the pristine vacuum level.

    Note, that this function only works for defect systems where the folder
    structure has been created with asr.setup.defects!
    """
    from asr.core import read_json
    from asr.defect_symmetry import check_and_return_input, DefectInfo
    from gpaw import restart

    # return index of the point defect in the defect structure
    structure, _, primitive, _ = check_and_return_input(
        structurefile='structure.json',
        primitivefile='../../unrelaxed.json')

    p = Path('.')
    defectinfo = DefectInfo(defectpath=p)
    def_index = defectinfo.specs[0]
    is_vacancy = defectinfo.is_vacancy(defectinfo.names[0])

    # get calculators and atoms for pristine and defect calculation
    try:
        p = Path('.')
        pristinelist = list(p.glob('./../../defects.pristine_sc*/'))
        pris_folder = pristinelist[0]
        res_pris = read_json(pris_folder / 'results-asr.gs.json')
        struc_pris, calc_pris = restart(pris_folder / 'gs.gpw', txt=None)
        struc_def, calc_def = restart(p / 'gs.gpw', txt=None)
    except FileNotFoundError as err:
        msg = (
            'does not find pristine gs, pristine results, or defect'
            ' results. Did you run setup.defects and calculate the ground'
            ' state for defect and pristine system?')
        raise RuntimeError(msg) from err

    # evaluate which atom possesses maximum distance to the defect site
    ref_index = get_reference_index(def_index, struc_pris)

    # get atomic electrostatic potentials at the atom far away for both the
    # defect and pristine system
    pot_def, pot_pris = extract_atomic_potentials(calc_def, calc_pris,
                                                  ref_index, is_vacancy)

    # get newly referenced eigenvalues for pristine and defect, as well as
    # pristine fermi level for evaluation of the band gap
    if sum(struc_def.pbc) == 3:
        evac = 0
    else:
        evac = res_pris['evac']

    vbm = res_pris['vbm'] - pot_pris
    cbm = res_pris['cbm'] - pot_pris

    ev_def = calc_def.get_eigenvalues() - pot_def
    ef_def = calc_def.get_fermi_level() - pot_def

    # evaluate whether there are states above or below the fermi level
    # and within the bandgap
    above_below = get_above_below(ev_def, ef_def, vbm, cbm)
    dif = pot_def - pot_pris + evac

    # check whether difference in atomic electrostatic potentials is
    # not too large
    assert abs(pot_def - pot_pris) < 0.1, ("large potential difference "
                                           "in energy referencing")

    # evaluate states within the gap
    statelist = [i for i, state in enumerate(ev_def) if vbm < state < cbm]

    return statelist, above_below, dif


def get_above_below(evs, ef, vbm, cbm):
    """Check whether there are states above/below EF in the gap."""
    above = False
    below = False
    for ev in evs:
        is_inside_gap = vbm < ev < cbm
        if is_inside_gap:
            above |= ev > ef
            below |= ev < ef

    return (above, below)


def return_erange_states(evs, ef, erange):
    """Return states within a certain energy range wrt. the Fermi level."""
    return [i for i, state in enumerate(evs) if (
        (ef + erange[0]) <= state <= (ef + erange[1]))]


if __name__ == '__main__':
    main.cli()
