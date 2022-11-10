from asr.core import ASRResult, prepare_result
from asr.database.browser import make_panel_description, href
from pathlib import Path
from ase.io import Trajectory
import numpy as np
import typing


panel_description = make_panel_description(
    """
Analysis of the thermodynamic stability of the defect using Slater-Janak
 transition state theory.
""",
    articles=[
        href("""M. Pandey et al. Defect-tolerant monolayer transition metal
dichalcogenides, Nano Letters, 16 (4) 2234 (2016)""",
             'https://doi.org/10.1021/acs.nanolett.5b04513'),
    ],
)


def get_transition_table(result, defstr):
    """Set up table for charge transition levels."""
    from asr.database.browser import describe_entry

    trans_results = result.transitions
    vbm = result.pristine.vbm
    transition_labels = []
    transition_array = np.zeros((len(trans_results), 2))
    for i, element in enumerate(trans_results):
        q = int(element['transition_name'].split('/')[-1])
        if q > 0:
            sign = 1
            transition_array[i, 1] = element['transition_values']['erelax']
        elif q < 0:
            sign = -1
            transition_array[i, 1] = element['transition_values']['erelax']
        transition_labels.append(f"{defstr} ({element['transition_name']})")
        transition_array[i, 0] = (element['transition_values']['transition']
                                  - element['transition_values']['evac']
                                  - vbm
                                  + sign * element['transition_values']['erelax'])

    argsort = transition_array[:, 0].argsort()
    transition_array = transition_array[argsort]
    transition_labels = [transition_labels[i] for i in argsort]

    # transition_table = matrixtable(
    #     transition_array,
    #     title='Transition',
    #     columnlabels=[describe_entry('Transition energy [eV]',
    #                                  description='SJ calculated transition level'),
    #                   describe_entry('Relaxation correction [eV]',
    #                                  description='Correction due to ion relaxation')],
    #     rowlabels=transition_labels)

    rows = []
    for i, element in enumerate(trans_results):
        rows.append((transition_labels[i],
                     describe_entry(f'{transition_array[i, 0]:.2f} eV',
                                    'SJ calculated thermodynamic transition.'),
                     describe_entry(f'{transition_array[i, 1]:.2f} eV',
                                    'Correction due to ion relaxation.')))

    transition_table = {'type': 'table',
                        'header': ['Transition', 'Transition energy',
                                   'Relaxation correction']}
    transition_table['rows'] = rows

    return transition_table


def get_summary_table(result):
    from asr.database.browser import table, describe_entry

    summary_table = table(result, 'Defect properties', [])
    summary_table['rows'].extend(
        [[describe_entry('Neutral formation energy',
                         description='Neutral formation energy [eV].'),
          f'{result.eform[0][0]:.2f} eV']])

    return summary_table


def get_formation_table(result, defstr):
    from asr.database.browser import table, describe_entry

    formation_table = table(result, 'Defect formation energy', [])
    for element in result.eform:
        formation_table['rows'].extend(
            [[describe_entry(f'{defstr} (q = {element[1]:1d} @ VBM)',
                             description='Formation energy for charge state q '
                                         'at the valence band maximum [eV].'),
              f'{element[0]:.2f} eV']])

    return formation_table


def webpanel(result, row, key_descriptions):
    from asr.database.browser import (fig, WebPanel,
                                      describe_entry)

    explained_keys = []
    for key in ['eform']:
        if key in result.key_descriptions:
            key_description = result.key_descriptions[key]
            explanation = key_description
            explained_key = describe_entry(key, description=explanation)
        else:
            explained_key = key
        explained_keys.append(explained_key)

    defname = row.defect_name
    defstr = f"{defname.split('_')[0]}<sub>{defname.split('_')[1]}</sub>"
    formation_table_sum = get_summary_table(result)
    formation_table = get_formation_table(result, defstr)
    # defectinfo = row.data.get('asr.defectinfo.json')
    transition_table = get_transition_table(result, defstr)

    panel = WebPanel(
        describe_entry('Formation energies and charge transition levels (Slater-Janak)',
                       panel_description),
        columns=[[describe_entry(fig('sj_transitions.png'),
                                 'Slater-Janak calculated charge transition levels.'),
                  transition_table],
                 [describe_entry(fig('formation.png'),
                                 'Formation energy diagram.'),
                  formation_table]],
        plot_descriptions=[{'function': plot_charge_transitions,
                            'filenames': ['sj_transitions.png']},
                           {'function': plot_formation_energies,
                            'filenames': ['formation.png']}],
        sort=29)

    summary = {'title': 'Summary',
               'columns': [[formation_table_sum],
                           []],
               'sort': 0}

    return [panel, summary]


@prepare_result
class PristineResults(ASRResult):
    """Container for pristine band gap results."""

    vbm: float
    cbm: float
    evac: float

    key_descriptions = dict(
        vbm='Pristine valence band maximum [eV].',
        cbm='Pristine conduction band minimum [eV]',
        evac='Pristine vacuum level [eV]')


@prepare_result
class TransitionValues(ASRResult):
    """Container for values of a specific charge transition level."""

    transition: float
    erelax: float
    evac: float

    key_descriptions = dict(
        transition='Charge transition level [eV]',
        erelax='Reorganization contribution  to the transition level [eV]',
        evac='Vacuum level for halfinteger calculation [eV]')


@prepare_result
class TransitionResults(ASRResult):
    """Container for charge transition level results."""

    transition_name: str
    transition_values: TransitionValues

    key_descriptions = dict(
        transition_name='Name of the charge transition (Initial State/Final State)',
        transition_values='Container for values of a specific charge transition level.')


@prepare_result
class TransitionListResults(ASRResult):
    """Container for all charge transition level results."""

    transition_list: typing.List[TransitionResults]

    key_descriptions = dict(
        transition_list='List of TransitionResults objects.')


@prepare_result
class StandardStateResult(ASRResult):
    """Container for results related to the standard state of the present defect."""

    element: str
    eref: float

    key_descriptions = dict(
        element='Atomic species.',
        eref='Reference energy extracted from OQMD [eV].')


@prepare_result
class Result(ASRResult):
    """Container for Slater Janak results."""

    transitions: typing.List[TransitionResults]
    pristine: PristineResults
    eform: typing.List[typing.Tuple[float, int]]
    standard_states: typing.List[StandardStateResult]
    hof: float

    key_descriptions = dict(
        transitions='Charge transition levels with [transition energy, '
                    'relax correction, reference energy] eV',
        pristine='Container for pristine band gap results.',
        eform='List of formation energy tuples (eform wrt. standard states [eV], '
              'charge state)',
        standard_states='List of StandardStateResult objects for each species.',
        hof='Heat of formation for the pristine monolayer [eV]')

    formats = {"ase_webpanel": webpanel}


# @command(module='asr.sj_analyze',
#         requires=['sj_+0.5/gs.gpw', 'sj_-0.5/gs.gpw',
#                   '../../unrelaxed.json', 'gs.gpw'],
#         resources='24:2h',
#         returns=Result)
# @option('--index', help='Specify index of the atom in the pristine supercell '
#        'that you want to use as a potential reference. Will be chosen '
#        'automatically if nothing is set.', type=int)
def main(index: int = None) -> Result:
    """Calculate charge transition levels for defect systems.

    This recipe uses SJ theory to calculate charge transition levels for defect systems.
    At least, asr.setup.sj had to be run in the charge_0 folder of a defect system and
    the half integer calculations have to be finished within the newly created folders.
    """
    from ase.db import connect
    from ase.io import read
    from asr.core import read_json
    from asr.defect_symmetry import DefectInfo

    p = Path('.')
    defectinfo = DefectInfo(defectpath=p)
    struc_pris, struc_def, calc_pris, calc_def = get_strucs_and_calcs(p)

    # get heat of formation
    c2db = connect('/home/niflheim/fafb/db/c2db_july20.db')
    primitive = read('../../unrelaxed.json')
    hof = get_heat_of_formation(c2db, primitive)

    # Obtain a list of all transitions with the respective ASRResults object
    # get index of the integer defect system
    ev = calc_def.get_eigenvalues()
    e_fermi = calc_def.get_fermi_level()
    N_homo_q = get_homo_index(ev, e_fermi)
    transition_list = calculate_transitions(index, N_homo_q, defectinfo)

    # get pristine band edges for correct referencing and plotting
    pristine = get_pristine_band_edges(index)

    # get neutral formation energy without chemical potentials applied
    p = Path('.')
    pris = list(p.glob('./../../defects.pristine_sc*'))[0]
    results_def = read_json('./results-asr.gs.json')
    results_pris = read_json(pris / 'results-asr.gs.json')
    etot_def = results_def['etot']
    etot_pris = results_pris['etot']
    oqmd = connect('/home/niflheim/fafb/db/oqmd12.db')
    eform, standard_states = calculate_neutral_formation_energy(etot_def, etot_pris,
                                                                oqmd, defectinfo)

    # get formation energies for all charge states based on neutral
    # formation energy, as well as charge transition levels, and pristine results
    vbm = pristine['vbm']
    eform = calculate_formation_energies(eform, transition_list, vbm)

    return Result.fromdata(eform=eform,
                           transitions=transition_list,
                           pristine=pristine,
                           standard_states=standard_states,
                           hof=hof)


def calculate_formation_energies(eform, transitions, vbm):
    """Calculate formation energies for all charge states at the VB band edge."""
    # CALCULATION OF FORMATION ENERGIES
    transitions = order_transitions(transitions)
    enlist = []
    for element in transitions:
        name = element['transition_name']
        q = int(name.split('/')[-1])
        if q < 0:
            enlist.append(element['transition_values']['transition']
                          - element['transition_values']['erelax']
                          - element['transition_values']['evac'])
        elif q > 0:
            enlist.append(element['transition_values']['transition']
                          + element['transition_values']['erelax']
                          - element['transition_values']['evac'])

    eform_list = [(eform, 0)]
    for i, element in enumerate(transitions):
        name = element['transition_name']
        q = int(name.split('/')[-1])
        if q < 0:
            enlist.append(element['transition_values']['transition']
                          - element['transition_values']['erelax']
                          - element['transition_values']['evac'])
        elif q > 0:
            enlist.append(element['transition_values']['transition']
                          + element['transition_values']['erelax']
                          - element['transition_values']['evac'])
        if name.split('/')[0].startswith('0') and name.split('/')[1].startswith('-'):
            y1 = eform
            y2 = eform
        elif name.split('/')[0].startswith('0'):
            y1 = eform
            y2 = eform
        elif not name.split('/')[1].startswith('-'):
            y2 = None
        else:
            y1 = None
        if name.split('/')[1].startswith('-'):
            a = float(name.split('/')[1])
            b = get_b(enlist[i], y2, a)
            if y1 is None:
                y1 = f(enlist[i], a, b)
            y2 = f(enlist[i + 1], a, b)
            eform_list.append((f(vbm, a, b), int(a)))
        elif not name.split('/')[0].startswith('-') or name.split('/').startswith('-'):
            a = float(name.split('/')[1])
            b = get_b(enlist[i], y1, a)
            if y2 is None:
                y2 = f(enlist[i], a, b)
            y1 = f(enlist[i + 1], a, b)
            eform_list.append((f(vbm, a, b), int(a)))

    return eform_list


def get_heat_of_formation(db, atoms):
    """Extract heat of formation from C2DB."""
    from asr.database.material_fingerprint import get_uid_of_atoms, get_hash_of_atoms

    hash = get_hash_of_atoms(atoms)
    uid = get_uid_of_atoms(atoms, hash)

    for row in db.select(uid=uid):
        hof = row.hform

    return hof


def calculate_transitions(index, N_homo_q, defectinfo):
    """Calculate all of the present transitions and return TransitionResults."""
    transition_list = []
    # First, get IP and EA (charge transition levels for the neutral defect
    if Path('./sj_+0.5/gs.gpw').is_file() and Path('./sj_-0.5/gs.gpw').is_file():
        for delta in [+1, -1]:
            transition = [0, delta]
            transition_results = get_transition_level(
                transition, 0, index, N_homo_q, defectinfo)
            transition_list.append(transition_results)

    for q in [-3, -2, -1, 1, 2, 3]:
        if q > 0 and Path('./../charge_{}/sj_+0.5/gs.gpw'.format(q)).is_file():
            transition = [q, q + 1]
            transition_results = get_transition_level(
                transition, q, index, N_homo_q, defectinfo)
            transition_list.append(transition_results)
        if q < 0 and Path('./../charge_{}/sj_-0.5/gs.gpw'.format(q)).is_file():
            transition = [q, q - 1]
            transition_results = get_transition_level(
                transition, q, index, N_homo_q, defectinfo)
            transition_list.append(transition_results)

    return transition_list


def get_pristine_band_edges(index, defectinfo) -> PristineResults:
    """Return band edges and vaccum level for the host system."""
    from asr.get_wfs import (get_reference_index,
                             extract_atomic_potentials)
    from asr.core import read_json

    # return index of the point defect in the defect structure
    def_index = defectinfo.specs[0]
    is_vacancy = defectinfo.is_vacancy(defectinfo.names[0])

    # get calculators and atoms for pristine and defect calculation
    p = Path('.')
    struc_pris, struc_def, calc_pris, calc_def = get_strucs_and_calcs(p)

    # evaluate which atom possesses maximum distance to the defect site
    if index is None:
        ref_index = get_reference_index(def_index, struc_pris)
    else:
        ref_index = index

    pot_def, pot_pris = extract_atomic_potentials(calc_def, calc_pris,
                                                  ref_index, is_vacancy)

    print('INFO: extract pristine band edges.')
    p = Path('.')
    pris = list(p.glob('./../../defects.pristine_sc*'))[0]
    if Path(pris / 'results-asr.gs.json').is_file():
        results_pris = read_json(pris / 'results-asr.gs.json')
        vbm = results_pris['vbm'] - pot_pris
        cbm = results_pris['cbm'] - pot_pris
    else:
        vbm = None
        cbm = None

    return PristineResults.fromdata(
        vbm=vbm,
        cbm=cbm,
        evac=0)


def obtain_chemical_potential(symbol, db):
    """Extract the standard state of a given element."""
    energies_ss = []
    if symbol == 'v' or symbol == 'i':
        eref = 0.
    else:
        for row in db.select(symbol, ns=1):
            energies_ss.append(row.energy / row.natoms)
        eref = min(energies_ss)

    return StandardStateResult.fromdata(
        element=symbol,
        eref=eref)


def calculate_neutral_formation_energy(etot_def, etot_pris, db, defectinfo):
    """Calculate the neutral formation energy with chemical potential shift applied.

    Only the neutral one is needed as for the higher charge states we will use the sj
    transitions for the formation energy plot.
    """
    eform = etot_def - etot_pris
    # next, extract standard state energies for particular defect
    for defect in defectinfo.names:
        def_add, def_remove = defectinfo.get_defect_type_and_kind_from_defectname(
            defect)
        # extract standard states of defect atoms from OQMD
        standard_states = []
        standard_states.append(obtain_chemical_potential(def_add, db))
        standard_states.append(obtain_chemical_potential(def_remove, db))
        # add, subtract chemical potentials to/from formation energy
        eform = eform - standard_states[0].eref + standard_states[1].eref

    return eform, standard_states


def get_strucs_and_calcs(path):
    from gpaw import restart

    try:
        pristinelist = list(path.glob('./../../defects.pristine_sc*/'))
        pris_folder = pristinelist[0]
        struc_pris, calc_pris = restart(pris_folder / 'gs.gpw', txt=None)
        struc_def, calc_def = restart(path / 'gs.gpw', txt=None)
    except FileNotFoundError as err:
        msg = ('does not find pristine gs, pristine results, or defect'
               ' results. Did you run setup.defects and calculate the ground'
               ' state for defect and pristine system?')
        raise RuntimeError(msg) from err

    return struc_pris, struc_def, calc_pris, calc_def


def get_half_integer_calc_and_index(charge, transition):
    """
    Return pos. or neg. half integer calculator based on 'transition'.

    Also, return index of the eigenvalue to be extracted (wrt. q HOMO index).
    """
    from gpaw import GPAW

    if transition[0] > transition[1]:
        identifier = '-0.5'
        delta_index = 1
    elif transition[1] > transition[0]:
        identifier = '+0.5'
        delta_index = 0
    parentpath = f'../charge_{charge}/sj_{identifier}'
    try:
        calc = GPAW(f'{parentpath}/gs.gpw', txt=None)
        print('INFO: calculate transition level q = {} -> q = {} transition.'.format(
            transition[0], transition[1]))
    except FileNotFoundError as err:
        msg = ('did not find gs.gpw in {parentpath}! Make sure to run asr.setup.defects'
               ' with the --halfinteger option and run the groundstate recipe in the '
               'newly created folders after.')
        raise RuntimeError(msg) from err

    return calc, delta_index


def get_transition_energy(evs, index):
    return evs[index]


def get_transition_level(transition,
                         charge,
                         index,
                         N_homo_q,
                         defectinfo) -> TransitionResults:
    """Calculate the charge transition level for a given charge transition."""
    from asr.get_wfs import (get_reference_index,
                             extract_atomic_potentials)
    from asr.defect_symmetry import DefectInfo

    # return index of the point defect in the defect structure
    p = Path('.')
    defectinfo = DefectInfo(defectpath=p)
    def_index = defectinfo.specs[0]
    is_vacancy = defectinfo.is_vacancy(defectinfo.names[0])

    # get string of the current charge
    charge = str(charge)

    # get calculators and atoms for pristine and defect calculation
    struc_pris, struc_def, calc_pris, calc_def = get_strucs_and_calcs(p)

    # evaluate which atom possesses maximum distance to the defect site
    if index is None:
        ref_index = get_reference_index(def_index, struc_pris)
    else:
        ref_index = index

    # extract homo and lumo of the half-integer system
    calc_half, delta = get_half_integer_calc_and_index(charge, transition)
    evs = calc_half.get_eigenvalues()
    e_trans = get_transition_energy(evs, N_homo_q + delta)

    # get atomic electrostatic potentials at the atom far away for both the
    # defect and pristine system
    pot_def, pot_pris = extract_atomic_potentials(calc_half, calc_pris,
                                                  ref_index, is_vacancy)

    # if possible, calculate correction due to relaxation in the charge state
    if Path('../charge_{}/results-asr.relax.json'.format(
            int(transition[1]))).is_file():
        print('INFO: calculate relaxation contribution to transition level.')
        traj = Trajectory('../charge_{}/relax.traj'.format(str(int(transition[1]))))
        e_cor = traj[0].get_potential_energy() - traj[-1].get_potential_energy()
    else:
        print('INFO: no relaxation for the charged state present. Do not calculate '
              'relaxation contribution to transition level.')
        e_cor = 0

    transition_name = f'{transition[0]}/{transition[1]}'

    transition_values = return_transition_values(e_trans, e_cor, pot_def)

    return TransitionResults.fromdata(
        transition_name=transition_name,
        transition_values=transition_values)


def get_homo_index(evs, ef):
    occ = []
    [occ.append(ev) for ev in evs if ev < ef]

    return len(occ) - 1


def return_transition_values(e_trans, e_cor, e_ref) -> TransitionValues:
    return TransitionValues.fromdata(
        transition=e_trans,
        erelax=e_cor,
        evac=e_ref)


def order_transitions(transitions):
    translist = []
    nameslist = []
    reflist = ['0/1', '1/2', '2/3', '0/-1', '-1/-2', '-2/-3']

    for element in transitions:
        if element['transition_name'] in reflist:
            translist.append(element)
            nameslist.append(element['transition_name'])

    ordered_list = []
    for element in reflist:
        for oelement in transitions:
            if element == oelement['transition_name']:
                ordered_list.append(oelement)

    return ordered_list


def get_b(x, y, a):
    return y - a * x


def f(x, a, b):
    return a * x + b


def plot_formation_energies(row, fname):
    """Plot formation energies and transition levels within the gap."""
    import matplotlib.pyplot as plt

    colors = {'0': 'C0',
              '1': 'C1',
              '2': 'C2',
              '3': 'C3',
              '-1': 'C4',
              '-2': 'C5',
              '-3': 'C6',
              '-4': 'C7',
              '4': 'C8'}

    data = row.data.get('results-asr.sj_analyze.json')

    vbm = data['pristine']['vbm']
    cbm = data['pristine']['cbm']
    gap = abs(cbm - vbm)
    eform = data['eform']
    transitions = data['transitions']

    fig, ax1 = plt.subplots()

    ax1.axvspan(-20, 0, color='grey', alpha=0.5)
    ax1.axvspan(gap, 20, color='grey', alpha=0.5)
    ax1.axhline(0, color='black', linestyle='dotted')
    ax1.axvline(gap, color='black', linestyle='solid')
    ax1.axvline(0, color='black', linestyle='solid')
    for element in eform:
        ax1.plot([0, gap], [f(0, element[1], element[0]),
                            f(gap, element[1], element[0])],
                 color=colors[str(element[1])],
                 label=element[1])

    ax1.set_xlim(-0.2 * gap, gap + 0.2 * gap)
    yrange = ax1.get_ylim()[1] - ax1.get_ylim()[0]
    ax1.text(-0.1 * gap, 0.5 * yrange, 'VBM', ha='center',
             va='center', rotation=90, weight='bold', color='white',
             fontsize=12)
    ax1.text(gap + 0.1 * gap, 0.5 * yrange, 'CBM', ha='center',
             va='center', rotation=90, weight='bold', color='white',
             fontsize=12)

    tickslist = []
    labellist = []
    energies = []
    for i, element in enumerate(transitions):
        name = element['transition_name']
        q = int(name.split('/')[-1])
        if q < 0:
            energy = (element['transition_values']['transition']
                      - element['transition_values']['erelax']
                      - element['transition_values']['evac'] - vbm)
        elif q > 0:
            energy = (element['transition_values']['transition']
                      + element['transition_values']['erelax']
                      - element['transition_values']['evac'] - vbm)
        energies.append(energy)
        if energy > 0 and energy < (gap):
            tickslist.append(energy)
            labellist.append(name)
            ax1.axvline(energy, color='grey', linestyle='dotted')
    energies.append(100)

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks([])
    ax1.set_xlabel(r'$E_\mathrm{F} - E_\mathrm{VBM}}$ [eV]')
    ax1.set_ylabel(r'$E^f$ (wrt. standard states) [eV]')
    ax1.legend()

    plt.tight_layout()

    plt.savefig(fname)
    plt.close()


def plot_charge_transitions(row, fname):
    """Plot calculated CTL along with the pristine bandgap."""
    import matplotlib.pyplot as plt

    colors = {'0': 'C0',
              '1': 'C1',
              '2': 'C2',
              '3': 'C3',
              '-1': 'C4',
              '-2': 'C5',
              '-3': 'C6',
              '-4': 'C7',
              '4': 'C8'}

    data = row.data.get('results-asr.sj_analyze.json')

    vbm = data['pristine']['vbm']
    cbm = data['pristine']['cbm']

    gap = abs(cbm - vbm)

    transitions = data['transitions']

    plt.xlim(-1, 1)
    plt.ylim(-0.2 * gap, gap + 0.2 * gap)
    plt.xticks([], [])

    plt.axhspan(-5, 0, color='grey', alpha=0.5)
    plt.axhspan(gap, gap + 5, color='grey', alpha=0.5)
    plt.axhline(0, color='black', linestyle='solid')
    plt.axhline(gap, color='black', linestyle='solid')
    plt.text(0, -0.1 * gap, 'VBM', color='white',
             ha='center', va='center', weight='bold',
             fontsize=12)
    plt.text(0, gap + 0.1 * gap, 'CBM', color='white',
             ha='center', va='center', weight='bold',
             fontsize=12)

    i = 1
    for trans in transitions:
        name = trans['transition_name']
        q = int(name.split('/')[-1])
        q_new = int(name.split('/')[0])
        if q > 0:
            y = (trans['transition_values']['transition']
                 + trans['transition_values']['erelax']
                 - trans['transition_values']['evac'])
            color1 = colors[str(q)]
            color2 = colors[str(q_new)]
        elif q < 0:
            y = (trans['transition_values']['transition']
                 - trans['transition_values']['erelax']
                 - trans['transition_values']['evac'])
            color1 = colors[str(q)]
            color2 = colors[str(q_new)]
        if y <= (cbm + 0.2 * gap) and y >= (vbm - 0.2 * gap):
            plt.plot(np.linspace(-0.9, 0.5, 20), 20 * [y - vbm],
                     label=trans['transition_name'],
                     color=color1, mec=color2, mfc=color2, marker='s', markersize=3)
            i += 1

    plt.legend(loc='center right')
    plt.ylabel(r'$E - E_{\mathrm{VBM}}$ [eV]')
    plt.yticks()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


if __name__ == '__main__':
    main.cli()
