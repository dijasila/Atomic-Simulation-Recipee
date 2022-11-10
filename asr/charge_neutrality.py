"""Self-consistent EF calculation for defect systems.."""
from asr.core import ASRResult, prepare_result
from ase.dft.bandgap import bandgap
from ase.io import read
from asr.database.browser import (table, describe_entry, dl, code,
                                  make_panel_description, href, fig,
                                  WebPanel)
from asr.defect_symmetry import DefectInfo
import typing
import numpy as np


# TODO: automate degeneracy counting


panel_description = make_panel_description(
    """
Equilibrium defect energetics evaluated by solving E<sub>F</sub> self-consistently
until charge neutrality is achieved.
""",
    articles=[
        href("""J. Buckeridge, Equilibrium point defect and charge carrier
 concentrations in a meterial determined through calculation of the self-consistent
 Fermi energy, Comp. Phys. Comm. 244 329 (2019)""",
             'https://doi.org/10.1016/j.cpc.2019.06.017'),
    ],
)


def webpanel(result, row, key_descriptions):
    unit = result.conc_unit
    unitstring = f"cm<sup>{unit.split('^')[-1]}</sup>"
    panels = []
    for i, scresult in enumerate(result.scresults):
        condition = scresult.condition
        tables = []
        for element in scresult.defect_concentrations:
            conc_table = get_conc_table(result, element, unitstring)
            tables.append(conc_table)
        scf_overview, scf_summary = get_overview_tables(scresult, result, unitstring)
        plotname = f'neutrality-{condition}.png'
        panel = WebPanel(
            describe_entry(f'Equilibrium energetics: all defects ({condition})',
                           panel_description),
            columns=[[fig(f'{plotname}'), scf_overview], tables],
            plot_descriptions=[{'function': plot_formation_scf,
                                'filenames': [plotname]}],
            sort=25 + i)
        panels.append(panel)

    return panels


def get_overview_tables(scresult, result, unitstring):
    ef = scresult.efermi_sc
    gap = result.gap
    if ef < (gap / 4.):
        dopability = '<b style="color:red;">p-type</b>'
    elif ef > (3 * gap / 4.):
        dopability = '<b style="color:blue;">n-type</b>'
    else:
        dopability = 'intrinsic'

    # get strength of p-/n-type dopability
    if ef < 0:
        ptype_val = '100+'
        ntype_val = '0'
    elif ef > gap:
        ptype_val = '0'
        ntype_val = '100+'
    else:
        ptype_val = int((1 - ef / gap) * 100)
        ntype_val = int((100 - ptype_val))
    pn_strength = f'{ptype_val:3}% / {ntype_val:3}%'
    pn = describe_entry(
        'p-type / n-type balance',
        'Balance of p-/n-type dopability in percent '
        f'(normalized wrt. band gap) at T = {int(result.temperature):d} K.'
        + dl(
            [
                [
                    '100/0',
                    code('if E<sub>F</sub> at VBM')
                ],
                [
                    '0/100',
                    code('if E<sub>F</sub> at CBM')
                ],
                [
                    '50/50',
                    code('if E<sub>F</sub> at E<sub>gap</sub> * 0.5')
                ]
            ],
        )
    )

    is_dopable = describe_entry(
        'Intrinsic doping type',
        'Is the material intrinsically n-type, p-type or intrinsic at '
        f'T = {int(result.temperature):d} K?'
        + dl(
            [
                [
                    'p-type',
                    code('if E<sub>F</sub> < 0.25 * E<sub>gap</sub>')
                ],
                [
                    'n-type',
                    code('if E<sub>F</sub> 0.75 * E<sub>gap</sub>')
                ],
                [
                    'intrinsic',
                    code('if 0.25 * E<sub>gap</sub> < E<sub>F</sub> < '
                         '0.75 * E<sub>gap</sub>')
                ],
            ],
        )
    )

    scf_fermi = describe_entry(
        'Fermi level position',
        'Self-consistent Fermi level wrt. VBM at which charge neutrality condition is '
        f'fulfilled at T = {int(result.temperature):d} K [eV].')

    scf_summary = table(result, 'Charge neutrality', [])
    scf_summary['rows'].extend([[is_dopable, dopability]])
    scf_summary['rows'].extend([[scf_fermi, f'{ef:.2f} eV']])
    scf_summary['rows'].extend([[pn, pn_strength]])

    scf_overview = table(result,
                         f'Equilibrium properties @ {int(result.temperature):d} K', [])
    scf_overview['rows'].extend([[is_dopable, dopability]])
    scf_overview['rows'].extend([[scf_fermi, f'{ef:.2f} eV']])
    scf_overview['rows'].extend([[pn, pn_strength]])
    if scresult.n0 > 1e-5:
        n0 = scresult.n0
    else:
        n0 = 0
    scf_overview['rows'].extend(
        [[describe_entry('Electron carrier concentration',
                         'Equilibrium electron carrier concentration at '
                         f'T = {int(result.temperature):d} K.'),
          f'{n0:.1e} {unitstring}']])
    if scresult.p0 > 1e-5:
        p0 = scresult.p0
    else:
        p0 = 0
    scf_overview['rows'].extend(
        [[describe_entry('Hole carrier concentration',
                         'Equilibrium hole carrier concentration at '
                         f'T = {int(result.temperature):d} K.'),
          f'{p0:.1e} {unitstring}']])

    return scf_overview, scf_summary


def get_conc_table(result, element, unitstring):
    from asr.database.browser import table, describe_entry
    from asr.defectlinks import get_defectstring_from_defectinfo

    token = element['defect_name']
    defectinfo = DefectInfo(defecttoken=token)
    defectstring = get_defectstring_from_defectinfo(
        defectinfo, charge=0)  # charge is only a dummy parameter here
    # remove the charge string from the defectstring again
    clean_defectstring = defectstring.split('(charge')[0]
    scf_table = table(result, f'Eq. concentrations of '
                              f'{clean_defectstring} [{unitstring}]', [])
    for altel in element['concentrations']:
        if altel[0] > 1e1:
            scf_table['rows'].extend(
                [[describe_entry(f'<b>Charge {altel[1]:1d}</b>',
                                 description='Equilibrium concentration '
                                             'in charge state q at T = '
                                             f'{int(result.temperature):d} K.'),
                  f'<b>{altel[0]:.1e}</b>']])
        else:
            scf_table['rows'].extend(
                [[describe_entry(f'Charge {altel[1]:1d}',
                                 description='Equilibrium concentration '
                                             'in charge state q at T = '
                                             f'{int(result.temperature):d} K.'),
                  f'{altel[0]:.1e}']])

    return scf_table


@prepare_result
class ConcentrationResult(ASRResult):
    """Container for concentration results of a specific defect."""

    defect_name: str
    concentrations: typing.List[typing.Tuple[float, float, int]]

    key_descriptions = dict(
        defect_name='Name of the defect (see "defecttoken" of DefectInfo).',
        concentrations='List of concentration tuples containing (conc., eform @ SCEF, '
                       'chargestate).')


@prepare_result
class SelfConsistentResult(ASRResult):
    """Container for results under certain chem. pot. condition."""

    condition: str
    efermi_sc: float
    n0: float
    p0: float
    defect_concentrations: typing.List[ConcentrationResult]
    dopability: str

    key_descriptions: typing.Dict[str, str] = dict(
        condition='Chemical potential condition, e.g. A-poor. '
                  'If one is poor, all other potentials are in '
                  'rich conditions.',
        efermi_sc='Self-consistent Fermi level at which charge '
                  'neutrality condition is fulfilled [eV].',
        n0='Electron carrier concentration at SC Fermi level.',
        p0='Hole carrier concentration at SC Fermi level.',
        defect_concentrations='List of ConcentrationResult containers.',
        dopability='p-/n-type or intrinsic nature of material.')


@prepare_result
class Result(ASRResult):
    """Container for asr.charge_neutrality results."""

    scresults: typing.List[SelfConsistentResult]
    temperature: float
    gap: float
    conc_unit: str

    key_descriptions: typing.Dict[str, str] = dict(
        scresults='List of charge neutrality results for a given '
                  'chemical potential limit.',
        temperature='Temperature [K].',
        gap='Electronic band gap [eV].',
        conc_unit='Unit of calculated concentrations.')

    formats = {"ase_webpanel": webpanel}


# @command(module='asr.charge_neutrality',
#          requires=['gs.gpw'],
#          dependencies=['asr.gs@calculate'],
#          resources='1:10m',
#          returns=ASRResult)
# @option('--temp1', help='Synthesis temperature at which defects got created [K].',
#         type=float)
# @option('--temp2', help='Evaluation temperature at which charge neutrality is '
#         'evaluated [K]. If no input is given (i.e. temp2=-1), '
#         'it will default to T1.', type=float)
# @option('--defects', help='Defect dictionary. If no dict is given '
#         'as an input, the self-consistency will be run for defect '
#         'formation energies extracted from the asr.sj_analyze Recipe.',
#         type=DictStr())
# @option('--ni', help='Set a fixed background concentration '
#         'which should be considered for the self-consitent evaluation '
#         'in cm^{-d} where d is the dimensionality of the system. Positive '
#         'values for holes, negative values for electrons.', type=float)
# @option('--mu', help='Chemical potentials. If no dict is given '
#         'as an input, the self-consistency will be run for the input '
#         'defect dictionary.', type=DictStr())
# @option('--cornerpoints/--no-cornerpoints',
#         help='Evaluate charge neutrality at chemical potential '
#         'region cornerpoints evaluated from asr.chemical_potential '
#         'Recipe.', is_flag=True)
def main(temp1: float = 300,
         temp2: float = -1,
         defects: dict = {},
         ni: float = 0.0,
         mu: dict = {},
         cornerpoints: bool = False) -> ASRResult:
    """Calculate self-consistent Fermi energy for defect systems.

    This recipe calculates the self-consistent Fermi energy for a
    specific host system. It needs the defect folder structure
    that gets created with asr.setup.defects. If you do not have a
    defect folder structure present, please use the '--defects' option.
    It is structured the following way (at the example of MoS2):

    - defect_dict = {'defect_name': [(form. en. at VBM, charge state), (.., ..)],
                     ...}
    """
    from gpaw import restart
    # test input and read in defect dictionary from asr.sj_analyze results
    if defects == {}:
        inputdict = get_defect_dict_from_files()
    else:
        inputdict = defects

    # if T2 = -1, set T2 equal to T1. Otherwise, simply use input T2
    if temp2 == -1:
        temp2 = temp1

    # evaluate host crystal elements and hof
    host = read('../unrelaxed.json')
    # el_list = get_element_list(host)

    # read in pristine ground state calculation and evaluate,
    # renormalize density of states
    atoms, calc = restart('gs.gpw', txt=None)
    dos, EF, gap = get_dos(calc)
    dos = renormalize_dos(calc, dos, EF)

    # Calculate initial electron and hole carrier concentration
    n0, p0 = integrate_electron_hole_concentration(dos, EF, gap, temp2)

    unit = get_concentration_unit(host)

    # handle the different usages, either with cornerpoints, input chemical
    # potential dictionary, or none of the above
    if cornerpoints and mu != {}:
        raise TypeError('give either cornerpoints or mu-dict as input, not both!')
    elif cornerpoints:
        # TO BE IMPLEMENTED (JIBAN)
        resfile = ''
        mus = extract_cornerpoints(resfile)
        # TO BE IMPLEMENTED (JIBAN)
        adjust_eform = True
    elif mu != {}:
        mus = {'custom-chem.-pot.': mu}
        adjust_eform = True
    else:
        # dummy element
        mus = {'standard-states': 0}
        adjust_eform = False
    # note, that this can be a list with only one element solely for QPOD compatibility
    # and such that the old results will not be broken
    sc_results = []
    # convert units for input fixed concentration
    ni = convert_concentration_units(ni, atoms, True)
    for element in mus:
        # FIX - this needs to be adjusted once chemical potential recipe is ready
        if adjust_eform:
            defectdict = adjust_formation_energies(inputdict, mu)
            print(f'INFO: adjusted formation energies for {element}-conditions:')
            print(defectdict)
        else:
            defectdict = inputdict.copy()
        # Initialize self-consistent loop for finding Fermi energy
        E, d, i, maxsteps, E_step, epsilon, converged = initialize_scf_loop(gap)
        # Start the self-consistent loop
        while (i < maxsteps):
            E = get_new_sample_point(E, E_step, d)
            n0, p0 = integrate_electron_hole_concentration(dos, E, gap, temp2)
            # initialise lists for concentrations and charges
            conc_list = []
            charge_list = []
            sites, degeneracy = get_site_and_state_degeneracy()
            # loop over all defects
            for defecttype in defectdict:
                for defect in defectdict[defecttype]:
                    eform = get_formation_energy(defect, E)
                    conc_def = calculate_defect_concentration(eform,
                                                              sites,
                                                              degeneracy,
                                                              temp1)
                    conc_list.append(conc_def)
                    charge_list.append(defect[1])
            delta_new = calculate_delta(conc_list, charge_list, n0, p0, ni)
            converged = check_convergence(delta_new, conc_list, n0, p0, E_step, epsilon)
            if converged:
                break
            if i == 0:
                delta_old = delta_new
            else:
                E_step, d, delta_old = update_scf_parameters(delta_old, delta_new,
                                                             E_step, d)
            i += 1

        # if calculation is converged, show final results
        if converged:
            n0, p0 = integrate_electron_hole_concentration(dos,
                                                           E,
                                                           gap,
                                                           temp2)
            n0 = convert_concentration_units(n0, atoms)
            p0 = convert_concentration_units(p0, atoms)
            concentration_results = []
            for defecttype in defectdict:
                concentration_tuples = []
                for defect in defectdict[defecttype]:
                    eform = get_formation_energy(defect, E)
                    conc_def = calculate_defect_concentration(eform, sites, degeneracy,
                                                              temp1)
                    conc_def = convert_concentration_units(conc_def, atoms)
                    concentration_tuples.append((conc_def, int(defect[1]), eform))
                concentration_result = ConcentrationResult.fromdata(
                    defect_name=defecttype,
                    concentrations=concentration_tuples)
                concentration_results.append(concentration_result)
        else:
            raise RuntimeError('self-consistent E_F evaluation failed '
                               f'for {element} conditions!')

        dopability = get_dopability_type(E, gap)

        if E < 0 or E > gap:
            raise RuntimeError(
                f'{element} conditions, the self-consistent Fermi '
                'level position is outside the pristine band gap. Check your inputs, '
                'in particular the input intrinsic carrier concentration "ni"!')

        sc_results.append(SelfConsistentResult.fromdata(
            condition=element,
            efermi_sc=E,
            n0=n0,
            p0=p0,
            defect_concentrations=concentration_results,
            dopability=dopability))

    return Result.fromdata(
        scresults=sc_results,
        temperature=temp2,
        conc_unit=unit,
        gap=gap)


def extract_cornerpoints(resfile):
    """Extract cornerpoints from chemical potential Recipe."""
    # TODO: @Jiban, please implement the fucntionality here once
    #       your Recipe is in place
    return None


def return_mu_remove_add(element, mu):
    """
    Return chem. pot. contribution (mu_add, mu_remove) to formation energy.

    Returns mu_i * n_i where i is the defect species based on the name of
    the defect in 'element'.

    examples: * for vacancies: return mu_remove and set mu_add to zero
              * for substitutional defects: return both mu_remove and mu_add
              * for interstitials: return mu_add and set mu_remove to zero

    Note, that this function can currently just handle simple point defects and
    no defect complexes.
    """
    add = element.split('_')[0]
    remove = element.split('_')[1]
    if add == 'v':
        mu_add = 0
        mu_remove = mu[remove]
    elif add == 'i':
        mu_add = mu[remove]
        mu_remove = 0
    else:
        mu_add = mu[add]
        mu_remove = mu[remove]

    return mu_add, mu_remove


def extrapolate_formation(mu, eform_old, project_zero=False):
    """
    Return formation energies extrapolated to given chem. pot. values.

    Similar to 'return_formation_zerospace' but instead of mapping from
    reference to zero chemical potential space it maps from zero chemical
    potential space to mu space.
    """
    # set sign of chemical potential addition and removal based on whether
    # we project formation energy back to zero space (sign = -1) or
    # extrapolate it from zero space to a certain chemical potential value
    # (sign = +1)
    if project_zero:
        sign = -1
    else:
        sign = +1

    # define new dictionary of formation energies and calculate
    # extrapolated new values
    eform_new = {}
    for element in eform_old:
        mu_add, mu_remove = return_mu_remove_add(element, mu)
        oldform = eform_old[element]
        newform = []
        for element2 in oldform:
            form = element2[0]
            charge = element2[1]
            new = form - sign * (mu_add - mu_remove)
            newform.append((new, charge))
        eform_new[f'{element}'] = newform

    return eform_new


def check_convergence(delta, conc_list, n0, p0, E_step, epsilon):
    return check_delta_zero(delta, conc_list, n0, p0) or (E_step < epsilon)


def update_scf_parameters(delta_old, delta_new, E_step, d):
    if abs(delta_new) > abs(delta_old):
        E_step = E_step / 10.
        d = -1 * d
    delta_old = delta_new

    return E_step, d, delta_old


def get_dopability_type(energy, gap):
    if energy < 0.25 * gap:
        dopability = 'p-type'
    elif energy > 0.75 * gap:
        dopability = 'n-type'
    else:
        dopability = 'intrinsic'

    return dopability


def get_concentration_unit(atoms):
    """Evaluate conc. units based on the dimensionality of the system."""
    dim = sum(atoms.pbc)
    if dim == 2:
        unit = 'cm^-2'
    elif dim == 3:
        unit = 'cm^-3'

    return unit


def get_new_sample_point(energy, stepsize, d):
    return energy + stepsize * d


def get_site_and_state_degeneracy():
    # site and state dependent version to be implemented
    return 1, 1


def initialize_scf_loop(gap, E=0, maxsteps=1000, epsilon=1e-12):
    # d directional parameter
    # i loop index
    # maxsteps maximum number of steps for SCF loop
    # E_step initial step sizew
    # epsilon threshold for minimum step length
    # converged boolean to see whether calculation is converged
    return E, 1, 0, maxsteps, gap / 10., epsilon, False


def get_hof_from_sj_results():
    from asr.core import read_json
    from pathlib import Path

    p = Path('.')
    pathlist = list(p.glob('../defects.*/charge_0/results-asr.sj_analyze.json'))
    sj_res = read_json(pathlist[0])
    hof = sj_res['hof']

    return hof


def get_adjusted_chemical_potentials(host, hof, element):
    el_list = get_element_list(host)
    stoi = get_stoichiometry(host)
    sstates = {}
    for el in el_list:
        name = el
        if el == element:
            mu_el = hof / stoi[element]
            sstates[f'{name}'] = mu_el
        else:
            sstates[f'{name}'] = 0

    return sstates


def obtain_chemical_potential(symbol, db):
    """Extract the standard state of a given element."""
    energies_ss = []
    if symbol == 'v':
        eref = 0.
    else:
        for row in db.select(symbol, ns=1):
            energies_ss.append(row.energy / row.natoms)
        eref = min(energies_ss)
    return eref


def adjust_formation_energies(defectdict, mu):
    """Return defect dict extrapolated to mu chemical potential conditions."""
    newdict = {}
    # sstates = get_adjusted_chemical_potentials(host, hof, element)
    for defecttoken in defectdict:
        defectinfo = DefectInfo(defecttoken=defecttoken)
        defects = defectinfo.names
        # write adjusted formation energy for defect/defectcomplex to new dict
        tuple_list = []
        for tpl in defectdict[f'{defecttoken}']:
            eform = tpl[0]
            for defectname in defects:
                def_type, def_pos = defectinfo.get_defect_type_and_kind_from_defectname(
                    defectname)
                if def_type == 'v':
                    add = 0
                    remove = mu[f'{def_pos}']
                elif def_pos == 'i':
                    add = mu[f'{def_type}']
                    remove = 0
                else:
                    add = mu[f'{def_type}']
                    remove = mu[f'{def_pos}']
                # adjust formation energy for defects one by one
                eform = eform - add + remove
            tuple_list.append((eform,
                               tpl[1]))
        newdict[f'{defecttoken}'] = tuple_list

    return newdict


def get_stoichiometry(atoms, reduced=False):
    from ase.formula import Formula

    if reduced:
        w = Formula(atoms.get_chemical_formula()).stoichiometry()[1]
    else:
        w = Formula(atoms.get_chemical_formula())

    return w.count()


def get_element_list(atoms):
    """Return list of unique chem. elements of a structure."""
    symbollist = []
    for i, atom in enumerate(atoms):
        symbol = atoms.symbols[i]
        if symbol not in symbollist:
            symbollist.append(symbol)

    return symbollist


def convert_concentration_units(conc, atoms, invert=False):
    """
    Convert concentration to units of cm^-n.

    Note, that n is the dimensionality of the system. If invert is
    set to True, it does the inversion from cm^-n to per supercell.
    """
    volume = atoms.get_volume()
    dim = sum(atoms.pbc)
    # cell = atoms.get_cell()

    # conversion factor from \AA to cm
    ang_to_cm = 1. * 10 ** (-8)

    if dim == 1:
        raise NotImplementedError('Not implemented for 1D structures!')
    elif dim == 2:
        z = atoms.get_cell_lengths_and_angles()[2]
        volume = volume / z
    elif dim == 3:
        volume = volume
    if invert:
        conc = conc * (volume * (ang_to_cm ** dim))
    else:
        conc = conc / (volume * (ang_to_cm ** dim))

    return conc


def fermi_dirac_electrons(E, EF, T):
    _k = 8.617333262145e-5  # in eV/K

    return 1. / (np.exp((EF - E) / (_k * T)) + 1.)


def fermi_dirac_holes(E, EF, T):
    return 1 - fermi_dirac_electrons(E, EF, T)


def calculate_delta(conc_list, chargelist, n0, p0, ni):
    """
    Calculate charge balance for current energy.

    delta = n_0 - p_0 - sum_X(sum_q C_{X^q}).
    """
    delta = n0 - p0 + ni
    for i, c in enumerate(conc_list):
        delta = delta - c * chargelist[i]

    return delta


def check_delta_zero(delta_new, conc_list, n0, p0):
    argument = n0 + p0
    for c in conc_list:
        argument = argument - c
    if abs(delta_new) < abs(argument * 1e-12):
        return True
    else:
        return False


def calculate_defect_concentration(e_form, sites, degeneracy, T):
    """Calculate and return the defect concentration for a specific defect.

    For a particular charge state with the formation energy for a particular energy.

    Use C_X^q = N_X * g_{X^q} * exp(-E_f(X^q) / (k * T))

    Note, that e_form is the formation energy at the desired Fermi energy
    already! In general, it is not the one at the VBM.
    """
    _k = 8.617333262145e-5  # in eV/K
    return (sites * degeneracy * np.exp((-1) * e_form / (_k * T)))


def get_formation_energy(defect, energy):
    """Return formation energy of defect in a charge state at an energy."""
    E_form_0, charge = get_zero_formation_energy(defect)

    return E_form_0 + charge * energy


def get_zero_formation_energy(defect):
    """
    Return the formation energy of a given defect at the VBM.

    Note, that the VBM corresponds to energy zero.
    """
    eform = defect[0]
    charge = defect[1]

    return eform, charge


def get_defect_dict_from_files():
    """Read in the results of asr.sj_analyze and store the formation energies at VBM."""
    from asr.core import read_json
    from pathlib import Path

    p = Path('.')
    charged_folders = list(p.glob('./../defects.*/charge_0/'))
    sjflag = False

    defect_dict = {}
    for folder in charged_folders:
        respath = Path(folder / 'results-asr.sj_analyze.json')
        if respath.is_file():
            res = read_json(respath)
            defect_name = str(folder.absolute()).split('/')[-2].split('.')[-1]
            defect_dict[defect_name] = res['eform']
            sjflag = True

    if not sjflag:
        raise RuntimeError('No SJ results available for this material! Did you run '
                           'all preliminary calculation for this system?')

    print(f'INFO: read in formation energies of the defects: {defect_dict}.')

    return defect_dict


def integrate_electron_hole_concentration(dos, ef, gap, T):
    """
    Integrate electron and hole carrier concentration.

    Use the trapezoid rule to integrate the following expressions:

     - n0 = int_{E_gap}^{inf} dos(E) * f_e(EF, E, T) dE
     - p0 = int_{-inf}^{0} dos(E) * f_h(EF, E, T) dE
     - f_e(EF, E, T) = (exp((EF - E) / (k * T)) + 1)^{-1}
     - f_h(EF, E, T) = 1 - f_e(EF, E, T)
    """
    # define spacing for density of states integration
    dx = abs(dos[0][-1] - dos[0][0]) / len(dos[0])

    # electron carrier concentration integration
    int_el = []
    int_hole = []
    for i in range(len(dos[0])):
        energy = dos[0][i]
        rho = dos[1][i]
        if energy >= gap:
            int_el.append(rho * fermi_dirac_holes(energy, ef, T))
        elif energy <= 0:
            int_hole.append(rho * fermi_dirac_electrons(energy, ef, T))
    n0 = np.trapz(int_el, dx=dx)
    p0 = np.trapz(int_hole, dx=dx)

    return n0, p0


def renormalize_dos(calc, dos, ef):
    """Renormalize DOS according to number of electrons."""
    if calc.get_number_of_spins() == 2:
        Ne = calc.get_number_of_electrons() / 2.
    else:
        Ne = calc.get_number_of_electrons()

    Ne = calc.get_number_of_electrons()

    # integrate number of electrons
    dx = abs(dos[0][-1] - dos[0][0]) / len(dos[0])
    int_el = []
    for i in range(len(dos[0])):
        energy = dos[0][i]
        rho = dos[1][i]
        if energy <= ef:
            int_el.append(rho)
    n0 = np.trapz(int_el, dx=dx)

    print(f'INFO: number of electrons BEFORE renormalization: {n0}, '
          f'{calc.get_number_of_electrons()} (Reference).')
    for i in range(len(dos[0])):
        dos[1][i] = dos[1][i] * Ne / n0

    print('INFO: renormalize DOS with factor {}'.format(Ne / n0))

    # reintegrate up to Fermi energy to doublecheck whether number of
    # electrons is correct
    int_el = []
    for i in range(len(dos[0])):
        energy = dos[0][i]
        rho = dos[1][i]
        if energy <= ef:
            int_el.append(rho)
    n0 = np.trapz(int_el, dx=dx)

    print(f'INFO: number of electrons AFTER renormalization: {n0:.2f}, '
          f'{calc.get_number_of_electrons()} (Reference).')

    return dos


def get_band_edges(calc):
    """Return energy of VBM to reference to later."""
    gap, p1, p2 = bandgap(calc)
    if gap == 0:
        raise ValueError('No bandgap for the present host material!')

    evbm = calc.get_eigenvalues(spin=p1[0], kpt=p1[1])[p1[2]]
    ecbm = calc.get_eigenvalues(spin=p2[0], kpt=p2[1])[p2[2]]
    EF = calc.get_fermi_level()

    return evbm, ecbm, EF, gap


def get_dos(calc, npts=4001, width=0.01):
    """Return the density of states with energy set to zero for VBM."""
    dos = calc.get_dos(spin=0, npts=npts, width=width)
    if calc.get_number_of_spins() == 2:
        dos_1 = calc.get_dos(spin=1, npts=npts, width=width)
        for i in range(len(dos[0])):
            dos[1][i] = dos[1][i] + dos_1[1][i]
    else:
        print('INFO: non spin-polarized calculation! Only one spin channel present!')
    evbm, ecbm, EF, gap = get_band_edges(calc)

    # reference density of states such that E_VBM = 0
    for i in range(npts):
        dos[0][i] = dos[0][i] - evbm
    EF = EF - evbm

    return dos, EF, gap


# all of the following are just plotting functionalities
def plot_formation_scf(row, fname):
    """Plot formation energy diagram and SC Fermi level wrt. VBM."""
    import matplotlib.pyplot as plt

    data = row.data.get('results-asr.charge_neutrality.json')
    gap = data['gap']
    comparison = fname.split('neutrality-')[-1].split('.png')[0]
    fig, ax = plt.subplots()
    for j, condition in enumerate(data['scresults']):
        if comparison == condition['condition']:
            ef = condition['efermi_sc']
            for i, defect in enumerate(condition['defect_concentrations']):
                name = defect['defect_name']
                def_type = name.split('_')[0]
                def_name = name.split('_')[-1]
                namestring = f"{def_type}$_\\{'mathrm{'}{def_name}{'}'}$"
                array = np.zeros((len(defect['concentrations']), 2))
                for num, conc_tuple in enumerate(defect['concentrations']):
                    q = conc_tuple[1]
                    eform = conc_tuple[2]
                    array[num, 0] = eform + q * (-ef)
                    array[num, 1] = q
                array = array[array[:, 1].argsort()[::-1]]
                # plot_background(ax, array)
                plot_lowest_lying(ax, array, ef, gap, name=namestring, color=f'C{i}')
            draw_band_edges(ax, gap)
            set_limits(ax, gap)
            draw_ef(ax, ef)
            set_labels_and_legend(ax, comparison)

    plt.tight_layout()
    plt.savefig(fname)


def set_labels_and_legend(ax, title):
    ax.set_xlabel(r'$E_\mathrm{F} - E_{\mathrm{VBM}}$ [eV]')
    ax.set_ylabel('$E^f$ [eV]')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(0.5, 1.1), ncol=5, loc='lower center')


def draw_ef(ax, ef):
    ax.axvline(ef, color='red', linestyle='dotted',
               label=r'$E_\mathrm{F}^{\mathrm{sc}}$')


def set_limits(ax, gap):
    ax.set_xlim(0 - gap / 10., gap + gap / 10.)


def get_min_el(array):
    elements = []
    for i in range(len(array)):
        elements.append(array[i, 0])
    for i, el in enumerate(elements):
        if el == min(elements):
            return i


def get_crossing_point(y1, y2, q1, q2):
    """
    Calculate the crossing point between two charge states.

    f1 = y1 + x * q1
    f2 = y2 + x * q2
    x * (q1 - q2) = y2 - y1
    x = (y2 - y1) / (q1 - q2)
    """
    return (y2 - y1) / float(q1 - q2)


def clean_array(array):
    index = get_min_el(array)

    return array[index:, :]


def get_y(x, array, index):
    q = array[index, 1]

    return q * x + array[index, 0]


def get_last_element(array, x_axis, y_axis, gap):
    y_cbms = []
    for i in range(len(array)):
        q = array[i, 1]
        eform = array[i, 0]
        y_cbms.append(q * gap + eform)

    x_axis.append(gap)
    y_axis.append(min(y_cbms))

    return x_axis, y_axis


def get_line_segment(array, index, x_axis, y_axis, gap):
    xs = []
    ys = []
    for i in range(len(array)):
        if i > index:
            y1 = array[index, 0]
            q1 = array[index, 1]
            y2 = array[i, 0]
            q2 = array[i, 1]
            crossing = get_crossing_point(y1, y2, q1, q2)
            xs.append(crossing)
            ys.append(q1 * crossing + y1)
        else:
            crossing = 1000
            xs.append(gap + 10)
            ys.append(crossing)
    min_index = index + 1
    for i, x in enumerate(xs):
        q1 = array[index, 1]
        y1 = array[index, 0]
        if x == min(xs) and x > 0 and x < gap:
            min_index = i
            x_axis.append(xs[min_index])
            y_axis.append(q1 * xs[min_index] + y1)

    return min_index, x_axis, y_axis


def plot_background(ax, array_in, gap):
    for i in range(len(array_in)):
        q = array_in[i, 1]
        eform = array_in[i, 0]
        y0 = eform
        y1 = eform + q * gap
        ax.plot([0, gap], [y0, y1], color='grey',
                alpha=0.2)


def plot_lowest_lying(ax, array_in, ef, gap, name, color):
    array_tmp = array_in.copy()
    array_tmp = clean_array(array_tmp)
    xs = [0]
    ys = [array_tmp[0, 0]]
    index, xs, ys = get_line_segment(array_tmp, 0, xs, ys, gap)
    for i in range(len(array_tmp)):
        if len(array_tmp[:, 0]) <= 1:
            break
        index, xs, ys = get_line_segment(array_tmp, index, xs, ys, gap)
        if index == len(array_tmp):
            break
    xs, ys = get_last_element(array_tmp, xs, ys, gap)
    ax.plot(xs, ys, color=color, label=name)
    ax.set_xlabel(r'$E_\mathrm{F}$ [eV]')


def draw_band_edges(ax, gap):
    ax.axvline(0, color='black')
    ax.axvline(gap, color='black')
    ax.axvspan(-100, 0, alpha=0.5, color='grey')
    ax.axvspan(gap, 100, alpha=0.5, color='grey')


if __name__ == '__main__':
    main.cli()
