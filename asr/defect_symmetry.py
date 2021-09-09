from ase.io import read
from asr.core import command, option, ASRResult, prepare_result, read_json
from asr.database.browser import make_panel_description, href
from gpaw import restart
import typing
import numpy as np
from pathlib import Path
from ase import Atoms


panel_description = make_panel_description(
    """
Analysis of defect states localized inside the pristine bandgap (energetics and
 symmetry).
""",
    articles=[
        href("""S. Kaappa et al. Point group symmetry anallysis of the electronic structure
of bare and protected nanocrystals, J. Phys. Chem. A, 122, 43, 8576 (2018)""",
             'https://doi.org/10.1021/acs.jpca.8b07923'),
    ],
)


def get_symmetry_table(state_results, vbm, cbm, row):
    import numpy as np
    from asr.database.browser import (matrixtable, table,
        describe_entry)

    data = row.data.get('results-asr.defect_symmetry.json')
    gsdata = row.data.get('results-asr.gs.json')
    eref = row.data.get('results-asr.get_wfs.json')['eref']
    ef = gsdata['efermi'] - eref

    Nrows = len(state_results)
    state_array = np.empty((Nrows, 5), dtype='object')
    state_rowlabels_0 = []
    state_rowlabels_1 = []
    state_spins_0 = []
    state_energies_0 = []
    state_label_0 = []
    state_accuracy_0 = []
    state_loc_ratio_0 = []
    state_spins_1 = []
    state_energies_1 = []
    state_energies_1 = []
    state_label_1 = []
    state_accuracy_1 = []
    state_loc_ratio_1 = []
    for i, row in enumerate(state_array):
        rowname = f"{int(state_results[i]['state']):.0f}"
        label = str(state_results[i]['best'])
        labelstr = label.lower()
        splitstr = split(labelstr)
        if len(splitstr) == 2:
            labelstr = f'{splitstr[0]}<sub>{splitstr[1]}</sub>'
        if state_results[i]['energy'] < cbm and state_results[i]['energy'] > vbm:
            if int(state_results[i]['spin']) == 0:
                state_rowlabels_0.append(rowname)
                state_spins_0.append(f"{int(state_results[i]['spin']):.0f}")
                state_energies_0.append(f"{state_results[i]['energy']:.2f}")
                state_label_0.append(labelstr)
                state_accuracy_0.append(f"{state_results[i]['error']:.2f}")
                state_loc_ratio_0.append(f"{state_results[i]['loc_ratio']:.2f}")
            elif int(state_results[i]['spin']) == 1:
                state_rowlabels_1.append(rowname)
                state_spins_1.append(f"{int(state_results[i]['spin']):.0f}")
                state_energies_1.append(f"{state_results[i]['energy']:.2f}")
                state_label_1.append(labelstr)
                state_accuracy_1.append(f"{state_results[i]['error']:.2f}")
                state_loc_ratio_1.append(f"{state_results[i]['loc_ratio']:.2f}")
    Nrows_0 = len(state_rowlabels_0)
    Nrows_1 = len(state_rowlabels_1)
    state_array_0 = np.empty((Nrows_0, 5), dtype='object')
    state_rowlabels_0.sort(reverse=True)
    state_array_1 = np.empty((Nrows_1, 5), dtype='object')
    state_rowlabels_1.sort(reverse=True)
    for i in range(Nrows_0):
        state_array_0[i, 0] = state_label_0[i]
        state_array_0[i, 1] = state_spins_0[i]
        state_array_0[i, 2] = state_accuracy_0[i]
        state_array_0[i, 3] = state_loc_ratio_0[i]
        state_array_0[i, 4] = state_energies_0[i]
    state_array_0 = state_array_0[state_array_0[:, -1].argsort()]
    for i in range(Nrows_1):
        state_array_1[i, 0] = state_label_1[i]
        state_array_1[i, 1] = state_spins_1[i]
        state_array_1[i, 2] = state_accuracy_1[i]
        state_array_1[i, 3] = state_loc_ratio_1[i]
        state_array_1[i, 4] = state_energies_1[i]
    state_array_1 = state_array_1[state_array_1[:, -1].argsort()]

    N_homo = 0
    N_lumo = 0
    for i in range(len(state_array_0)):
        if float(state_array_0[i, 4]) > ef:
            N_lumo += 1
    for i in range(len(state_array_0)):
        if float(state_array_0[i, 4]) > ef:
            state_rowlabels_0[i] = f'LUMO + {N_lumo - 1}'
            N_lumo = N_lumo - 1
            if N_lumo == 0:
                state_rowlabels_0[i] = 'LUMO'
        elif float(state_array_0[i, 4]) <= ef:
            state_rowlabels_0[i] = f'HOMO - {N_homo}'
            if N_homo == 0:
                state_rowlabels_0[i] = 'HOMO'
            N_homo = N_homo + 1
    state_array_0 = np.delete(state_array_0, 2, 1)

    N_homo = 0
    N_lumo = 0
    for i in range(len(state_array_1)):
        if float(state_array_1[i, 4]) > ef:
            N_lumo += 1
    for i in range(len(state_array_1)):
        if float(state_array_1[i, 4]) > ef:
            state_rowlabels_1[i] = f'LUMO + {N_lumo - 1}'
            N_lumo = N_lumo - 1
            if N_lumo == 0:
                state_rowlabels_1[i] = 'LUMO'
        elif float(state_array_1[i, 4]) <= ef:
            state_rowlabels_1[i] = f'HOMO - {N_homo}'
            if N_homo == 0:
                state_rowlabels_1[i] = 'HOMO'
            N_homo = N_homo + 1

    state_array_1 = np.delete(state_array_1, 2, 1)
    state_table_0 = matrixtable(state_array_0,
                                digits=None,
                                title='Orbital',
                                columnlabels=['Symmetry',
                                              'Spin',
                                              # 'Accuracy',
                                              'Localization ratio',
                                              'Energy [eV]'],
                                rowlabels=state_rowlabels_0)
    state_table_1 = matrixtable(state_array_1,
                                digits=None,
                                title='Orbital',
                                columnlabels=['Symmetry',
                                              'Spin',
                                              # 'Accuracy',
                                              'Localization ratio',
                                              'Energy [eV]'],
                                rowlabels=state_rowlabels_1)

    return state_table_0, state_table_1


def get_state_table(state_results, vbm, cbm, row):
    import numpy as np
    from asr.database.browser import (matrixtable, table,
        describe_entry)

    data = row.data.get('results-asr.defect_symmetry.json')
    gsdata = row.data.get('results-asr.gs.json')
    eref = row.data.get('results-asr.get_wfs.json')['eref']
    ef = gsdata['efermi'] - eref

    Nrows = len(state_results)
    state_array = np.empty((Nrows, 2), dtype='object')
    state_rowlabels_0 = []
    state_rowlabels_1 = []
    state_spins_0 = []
    state_energies_0 = []
    state_spins_1 = []
    state_energies_1 = []
    for i, row in enumerate(state_array):
        rowname = f"{int(state_results[i]['state']):.0f}"
        if state_results[i]['energy'] < cbm and state_results[i]['energy'] > vbm:
            if int(state_results[i]['spin']) == 0:
                state_rowlabels_0.append(rowname)
                state_spins_0.append(f"{int(state_results[i]['spin']):.0f}")
                state_energies_0.append(f"{state_results[i]['energy']:.2f}")
            elif int(state_results[i]['spin']) == 1:
                state_rowlabels_1.append(rowname)
                state_spins_1.append(f"{int(state_results[i]['spin']):.0f}")
                state_energies_1.append(f"{state_results[i]['energy']:.2f}")
    Nrows_0 = len(state_rowlabels_0)
    Nrows_1 = len(state_rowlabels_1)
    state_array_0 = np.empty((Nrows_0, 2), dtype='object')
    state_rowlabels_0.sort(reverse=True)
    state_array_1 = np.empty((Nrows_1, 2), dtype='object')
    state_rowlabels_1.sort(reverse=True)
    for i in range(Nrows_0):
        state_array_0[i, 0] = state_spins_0[i]
        state_array_0[i, 1] = state_energies_0[i]
    state_array_0 = state_array_0[state_array_0[:, 1].argsort()]
    for i in range(Nrows_1):
        state_array_1[i, 0] = state_spins_1[i]
        state_array_1[i, 1] = state_energies_1[i]
    state_array_1 = state_array_1[state_array_1[:, 1].argsort()]

    N_homo = 0
    N_lumo = 0
    for i in range(len(state_array_0)):
        if float(state_array_0[i, 1]) > ef:
            N_lumo += 1

    E_homo = vbm
    E_lumo = cbm
    for i in range(len(state_array_0)):
        if float(state_array_0[i, 1]) > ef:
            state_rowlabels_0[i] = f'LUMO + {N_lumo - 1}'
            N_lumo = N_lumo - 1
            if N_lumo == 0:
                state_rowlabels_0[i] = 'LUMO'
                E_lumo = float(state_array_0[i, 1])
        elif float(state_array_0[i, 1]) <= ef:
            state_rowlabels_0[i] = f'HOMO - {N_homo}'
            if N_homo == 0:
                state_rowlabels_0[i] = 'HOMO'
                E_homo = float(state_array_0[i, 1])
            N_homo = N_homo + 1
    E_hl_0 = E_lumo - E_homo

    N_homo = 0
    N_lumo = 0
    for i in range(len(state_array_1)):
        if float(state_array_1[i, 1]) > ef:
            N_lumo += 1

    E_homo = vbm
    E_lumo = cbm
    for i in range(len(state_array_1)):
        if float(state_array_1[i, 1]) > ef:
            state_rowlabels_1[i] = f'LUMO + {N_lumo - 1}'
            N_lumo = N_lumo - 1
            if N_lumo == 0:
                state_rowlabels_1[i] = 'LUMO'
                E_lumo = float(state_array_1[i, 1])
        elif float(state_array_1[i, 1]) <= ef:
            state_rowlabels_1[i] = f'HOMO - {N_homo}'
            if N_homo == 0:
                state_rowlabels_1[i] = 'HOMO'
                E_homo = float(state_array_1[i, 1])
            N_homo = N_homo + 1
    E_hl_1 = E_lumo - E_homo

    state_table_0 = matrixtable(state_array_0,
                                digits=None,
                                title='Orbital',
                                columnlabels=['Spin',
                                              'Energy [eV]'],
                                rowlabels=state_rowlabels_0)
    state_table_1 = matrixtable(state_array_1,
                                digits=None,
                                title='Orbital',
                                columnlabels=['Spin',
                                              'Energy [eV]'],
                                rowlabels=state_rowlabels_1)

    transition_table = table(row, 'Kohn-Sham HOMO-LUMO gap', [])
    transition_table['rows'].extend(
        [[describe_entry('Spin 0',
                         'KS HOMO-LUMO gap for spin 0 channel.'),
          f'{E_hl_0:.2f} eV']])
    transition_table['rows'].extend(
        [[describe_entry('Spin 1',
                         'KS HOMO-LUMO gap for spin 1 channel.'),
          f'{E_hl_1:.2f} eV']])

    return state_table_0, state_table_1, transition_table


def webpanel(result, row, key_descriptions):
    from asr.database.browser import (WebPanel,
                                      describe_entry,
                                      table,
                                      matrixtable,
                                      fig,
                                      href)

    spglib = href('SpgLib', 'https://spglib.github.io/spglib/')

    eref = row.data.get('results-asr.get_wfs.json')['eref']
    gsdata = row.data.get('results-asr.gs.json')
    ef = gsdata['efermi'] - eref

    basictable = table(row, 'Defect properties', [])
    pg_string = result.defect_pointgroup
    pg_strlist = split(pg_string)
    sub = ''.join(pg_strlist[1:])
    pg_string = f'{pg_strlist[0]}<sub>{sub}</sub>'
    basictable['rows'].extend(
        [[describe_entry('Defect point group',
                         f'The defect point group is calculated with {spglib}.'),
          pg_string]])

    description = describe_entry('One-electron states', panel_description)

    vbm = result.pristine['vbm']
    cbm = result.pristine['cbm']
    # adjust = compute_offset(result.symmetries, vbm, cbm, ef)
    # vbm = vbm + adjust
    # cbm = cbm + adjust
    if result.symmetries[0]['best'] is None:
        print('WARNING: no symmetry analysis for this defect present. Only plot '
              'gapstates!')
        state_table_0, state_table_1, transition_table = get_state_table(
            result.symmetries, vbm, cbm, row)
        if len(state_table_0['rows']) > 1 and len(state_table_1['rows']) > 1:
            panel = WebPanel(description,
                             columns=[[state_table_0, fig('ks_gap.png')], [state_table_1,
                                                            transition_table]],
                             plot_descriptions=[{'function': plot_gapstates,
                                                 'filenames': ['ks_gap.png']}],
                             sort=30)
        elif len(state_table_0['rows']) == 1 and len(state_table_1['rows']) > 1:
            panel = WebPanel(description,
                             columns=[[fig('ks_gap.png')], [state_table_1,
                                                            transition_table]],
                             plot_descriptions=[{'function': plot_gapstates,
                                                 'filenames': ['ks_gap.png']}],
                             sort=30)
        elif len(state_table_1['rows']) == 1 and len(state_table_0['rows']) > 1:
            panel = WebPanel(description,
                             columns=[[state_table_0, fig('ks_gap.png')], [transition_table]],
                             plot_descriptions=[{'function': plot_gapstates,
                                                 'filenames': ['ks_gap.png']}],
                             sort=30)
        else:
            panel = WebPanel(description,
                             columns=[[fig('ks_gap.png')], [transition_table]],
                             plot_descriptions=[{'function': plot_gapstates,
                                                 'filenames': ['ks_gap.png']}],
                             sort=30)

    else:
        state_table_0, state_table_1, transition_table = get_state_table(
            result.symmetries, vbm, cbm, row)
        symmetry_table_0, symmetry_table_1 = get_symmetry_table(
            result.symmetries, vbm, cbm, row)
        if len(symmetry_table_0['rows']) > 1 and len(symmetry_table_1['rows']) > 1:
            panel = WebPanel(description,
                             columns=[[symmetry_table_0,
                                       fig('ks_gap.png')],
                                      [symmetry_table_1,
                                       transition_table]],
                             plot_descriptions=[{'function': plot_gapstates,
                                                 'filenames': ['ks_gap.png']}],
                             sort=30)
        elif len(symmetry_table_0['rows']) == 1 and len(symmetry_table_1['rows']) > 1:
            panel = WebPanel(description,
                             columns=[[fig('ks_gap.png')],
                                      [symmetry_table_1, transition_table]],
                             plot_descriptions=[{'function': plot_gapstates,
                                                 'filenames': ['ks_gap.png']}],
                             sort=30)
        elif len(symmetry_table_1['rows']) == 1 and len(symmetry_table_0['rows']) > 1:
            panel = WebPanel(description,
                             columns=[[symmetry_table_0,
                                       fig('ks_gap.png')],
                                      [transition_table]],
                             plot_descriptions=[{'function': plot_gapstates,
                                                 'filenames': ['ks_gap.png']}],
                             sort=30)
        else:
            panel = WebPanel(description,
                             columns=[[fig('ks_gap.png')], [transition_table]],
                             plot_descriptions=[{'function': plot_gapstates,
                                                 'filenames': ['ks_gap.png']}],
                             sort=30)


    summary = {'title': 'Summary',
               'columns': [[basictable], []],
               'sort': 2}

    return [panel, summary]


@prepare_result
class IrrepResult(ASRResult):
    """Container for results of an individual irreproducible representation."""

    sym_name: str
    sym_score: float

    key_descriptions: typing.Dict[str, str] = dict(
        sym_name='Name of the irreproducible representation.',
        sym_score='Score of the respective representation.')


@prepare_result
class SymmetryResult(ASRResult):
    """Container for symmetry results for a given state."""

    irreps: typing.List[IrrepResult]
    best: str
    error: float
    loc_ratio: float
    state: int
    spin: int
    energy: float

    key_descriptions: typing.Dict[str, str] = dict(
        irreps='List of irreproducible representations and respective scores.',
        best='Irreproducible representation with the best score.',
        error='Error of identification of the best irreproducible representation.',
        loc_ratio='Localization ratio for a given state.',
        state='Index of the analyzed state.',
        spin='Spin of the analyzed state (0 or 1).',
        energy='Energy of specific state aligned to pristine semi-core state [eV].'
    )


@prepare_result
class PristineResult(ASRResult):
    """Container for pristine band edge results."""

    vbm: float
    cbm: float
    gap: float

    key_descriptions: typing.Dict[str, str] = dict(
        vbm='Energy of the VBM (ref. to the vacuum level in 2D) [eV].',
        cbm='Energy of the CBM (ref. to the vacuum level in 2D) [eV].',
        gap='Energy of the bandgap [eV].')


@prepare_result
class Result(ASRResult):
    """Container for main results for asr.analyze_state."""

    defect_pointgroup: str
    defect_center: typing.Tuple[float, float, float]
    defect_name: str
    symmetries: typing.List[SymmetryResult]
    pristine: PristineResult

    key_descriptions: typing.Dict[str, str] = dict(
        defect_pointgroup='Point group in Schoenflies notation.',
        defect_center='Position of the defect [Å, Å, Å].',
        defect_name='Name of the defect ({type}_{position})',
        symmetries='List of SymmetryResult objects for all states.',
        pristine='PristineResult container.'
    )

    formats = {'ase_webpanel': webpanel}


@command(module='asr.defect_symmetry',
         requires=['structure.json', 'unrelaxed.json',
                   '../../unrelaxed.json'],
         dependencies=['asr.get_wfs'],
         resources='1:1h',
         returns=Result)
@option('--mapping/--no-mapping', help='Choose mapping if defect '
        'supercells are created with the general algorithm of '
        'asr.setup.defects, or if non-uniform supercells are used.'
        ' Use --no-mapping otherwise.', is_flag=True)
@option('--radius', help='Radius around the defect where the wavefunction '
        'gets analyzed.', type=float)
def main(mapping: bool = False,
         radius: float = 2.0) -> Result:
    """
    Analyze defect wavefunctions and their symmetries.

    Note, that you need to set up your folder structure with
    asr.setup.defects in order to correctly run this recipe. Furthermore,
    run asr.get_wfs beforehand to write out the needed wavefunctions.
    """
    from ase.io.cube import read_cube_data
    from gpaw.point_groups import SymmetryChecker, point_group_names

    # define path of the current directory
    defect = Path('.')

    # check whether input is correct and return important structures
    structure, unrelaxed, primitive, pristine = check_and_return_input()

    # construct mapped structure, or return relaxed defect structure in
    # case mapping is not needed
    if mapping:
        mapped_structure = get_mapped_structure(structure,
                                                unrelaxed,
                                                primitive,
                                                pristine,
                                                defect)
    else:
        mapped_structure = read('structure.json')

    # return point group of the defect structure
    point_group = get_spg_symmetry(mapped_structure)

    # evaluate coordinates of defect in the supercell
    defecttype, defectpos = get_defect_info(primitive, defect)
    defectname = defecttype + '_' + defectpos
    center = return_defect_coordinates(structure,
                                       unrelaxed,
                                       primitive,
                                       pristine,
                                       defect)
    print(f'INFO: defect position: {center}, structural symmetry: {point_group}')

    # return pristine results to visualise wavefunctions within the gap
    pris_result = get_pristine_result()

    # read in cubefiles of the wavefunctions
    cubefiles = list(defect.glob('*.cube'))
    if len(cubefiles) == 0:
        raise FileNotFoundError('WARNING: no cube files available in this '
                                'folder!')

    # check whether point group is implemented in GPAW, return results
    # without symmetry analysis if it is not implmented
    symmetry_results = []
    if point_group not in point_group_names:
        print(f'WARNING: point group {point_group} not implemented in GPAW. '
              'Return results without symmetry analysis of the wavefunctions.')
        for wf_file in cubefiles:
            spin = str(wf_file)[str(wf_file).find('_') + 1]
            band = str(wf_file)[str(wf_file).find('.') + 1: str(wf_file).find('_')]
            res_wf = find_wf_result(band, spin)
            energy = res_wf['energy']
            wf, atoms = read_cube_data(str(wf_file))
            localization = get_localization_ratio(atoms, wf)
            irrep_results = [IrrepResult.fromdata(
                sym_name=None,
                sym_score=None)]
            symmetry_result = SymmetryResult.fromdata(irreps=irrep_results,
                                                      best=None,
                                                      error=None,
                                                      loc_ratio=localization,
                                                      state=band,
                                                      spin=spin,
                                                      energy=energy)
            symmetry_results.append(symmetry_result)
        return Result.fromdata(
            defect_pointgroup=point_group,
            defect_center=center,
            defect_name=defectname,
            symmetries=symmetry_results,
            pristine=pris_result)

    # symmetry analysis
    checker = SymmetryChecker(point_group, center, radius=radius)

    print('spin  band     norm    normcut     best    '
          + ''.join(f'{x:8.3s}' for x in checker.group.symmetries) + 'error')

    labels_up = []
    labels_down = []

    # read in calc once
    atoms, calc = restart('gs.gpw', txt=None)

    symmetry_results = []
    for wf_file in cubefiles:
        spin = str(wf_file)[str(wf_file).find('_') + 1]
        band = str(wf_file)[str(wf_file).find('.') + 1: str(wf_file).find('_')]
        res_wf = find_wf_result(band, spin)
        energy = res_wf['energy']

        wf, atoms = read_cube_data(str(wf_file))
        localization = get_localization_ratio(atoms, wf)

        dct = checker.check_function(wf, (atoms.cell.T / wf.shape).T)
        best = dct['symmetry']
        norm = dct['norm']
        normcut = dct['overlaps'][0]
        error = (np.array(list(dct['characters'].values()))**2).sum()

        print(f'{spin:6} {band:5} {norm:6.3f} {normcut:9.3f} {best:>8}'
              + ''.join(f'{x:8.3f}' for x in dct['characters'].values())
              + '{:9.3}'.format(error))

        [labels_up, labels_down][0].append(best)

        irrep_results = []
        for element in dct['characters']:
            irrep_result = IrrepResult.fromdata(sym_name=element,
                                                sym_score=dct['characters'][element])
            irrep_results.append(irrep_result)

        symmetry_result = SymmetryResult.fromdata(irreps=irrep_results,
                                                  best=best,
                                                  error=error,
                                                  loc_ratio=localization,
                                                  state=band,
                                                  spin=spin,
                                                  energy=energy)
        symmetry_results.append(symmetry_result)

    return Result.fromdata(
        defect_pointgroup=point_group,
        defect_center=center,
        defect_name=defectname,
        symmetries=symmetry_results,
        pristine=pris_result)


def get_pristine_result():
    """
    Return PristineResult object.

    In 2D, the reference will be the vacuum level of the pristine calculation.
    In 3D, the reference will be None (vacuum level doesn't make sense here).
    """
    from asr.core import read_json

    try:
        p = Path('.')
        # sc = str(p.absolute()).split('/')[-2].split('_')[1].split('.')[0]
        # pristinelist = list(p.glob(f'./../../defects.pristine_sc.{sc}/'))
        # pris_folder = pristinelist[0]
        pris = list(p.glob('./../../defects.pristine_sc*'))[0]
        res_pris = read_json(pris / 'results-asr.gs.json')
    except FileNotFoundError:
        print('ERROR: does not find pristine results. Did you run setup.defects '
              'and calculate the ground state for the pristine system?')

    try:
        read_json('results-asr.get_wfs.json')
    except FileNotFoundError:
        print('ERROR: does not find get_wfs results. Did you run asr.get_wfs? ')

    ref_pris = res_pris['evac']
    if ref_pris is None:
        ref_pris = 0

    return PristineResult.fromdata(
        vbm=res_pris['vbm'] - ref_pris,
        cbm=res_pris['cbm'] - ref_pris,
        gap=res_pris['gap'])


def get_localization_ratio(atoms, wf):
    """
    Return the localization ratio of the wavefunction.

    It is defined as the volume of the cell divided the
    integral of the fourth power of the wavefunction.
    """
    grid_vectors = (atoms.cell.T / wf.shape).T
    dv = abs(np.linalg.det(grid_vectors))
    V = atoms.get_volume()

    IPR = 1 / ((wf**4).sum() * dv)
    local_ratio = V / IPR

    return local_ratio


def find_wf_result(state, spin):
    """Read in results of asr.get_wfs and returns WaveFunctionResult."""
    res = read_json('results-asr.get_wfs.json')
    wfs = res['wfs']
    for wf in wfs:
        if int(wf['state']) == int(state) and int(wf['spin']) == int(spin):
            return wf

    print('ERROR: wf result to given wavefunction file not found! Make sure that '
          'you have consitent wavefunctions or delete old wavefunctions and rerun'
          ' asr.get_wfs.')
    raise Exception('ERROR: can not find corresponging wavefunction result for '
                    f'wavefunction no. {state}/{spin}!')
    return None


def get_mapped_structure(structure, unrelaxed, primitive, pristine, defect):
    """Return centered and mapped structure."""
    import numpy as np
    for delta in [0, 0.03, 0.05, 0.1, -0.03, -0.05, -0.1]:
        for cutoff in np.arange(0.1, 0.8, 0.04):
            threshold = 0.99
            translation = return_defect_coordinates(structure, unrelaxed, primitive,
                                                    pristine, defect)
            rel_struc, ref_struc, artificial, cell, N = recreate_symmetric_cell(
                structure,
                unrelaxed,
                primitive,
                pristine,
                translation,
                delta)
            indexlist = compare_structures(artificial, ref_struc, cutoff)
            ref_struc = remove_atoms(ref_struc, indexlist)
            rel_struc = remove_atoms(rel_struc, indexlist)
            indexlist = indexlist_cut_atoms(ref_struc, threshold)
            ref_struc = remove_atoms(ref_struc, indexlist)
            rel_struc = remove_atoms(rel_struc, indexlist)
            if not conserved_atoms(ref_struc, primitive, N, defect):
                threshold = 1.01
                rel_struc, ref_struc, artificial, cell, N = recreate_symmetric_cell(
                    structure,
                    unrelaxed,
                    primitive,
                    pristine,
                    translation,
                    delta)
                indexlist = compare_structures(artificial, ref_struc, cutoff)
                ref_struc = remove_atoms(ref_struc, indexlist)
                rel_struc = remove_atoms(rel_struc, indexlist)
                indexlist = indexlist_cut_atoms(ref_struc, threshold)
                ref_struc = remove_atoms(ref_struc, indexlist)
                rel_struc = remove_atoms(rel_struc, indexlist)
            if conserved_atoms(ref_struc, primitive, N, defect):
                break
        if conserved_atoms(ref_struc, primitive, N, defect):
            break
    if not conserved_atoms(ref_struc, primitive, N, defect):
        raise ValueError('number of atoms wrong in {}! Mapping not correct!'.format(
            defect.absolute()))

    return rel_struc


def get_spg_symmetry(structure, symprec=0.1):
    """Return the symmetry of a given structure evaluated with spglib."""
    import spglib as spg

    spg_sym = spg.get_spacegroup(structure, symprec=symprec, symbol_type=1)

    return spg_sym.split('^')[0]


def conserved_atoms(ref_struc, primitive, N, defectpath):
    """Return whether number of atoms is correct after the mapping or not."""
    if (is_vacancy(defectpath) and len(ref_struc) != (N * N * len(primitive) - 1)):
        return False
    elif (not is_vacancy(defectpath) and len(ref_struc) != (N * N * len(primitive))):
        return False
    else:
        print('INFO: number of atoms correct in {}'.format(
            defectpath.absolute()))
        return True


def remove_atoms(structure, indexlist):
    indices = np.array(indexlist)
    indices = np.sort(indices)[::-1]
    for element in indices:
        structure.pop(element)
    return structure


def indexlist_cut_atoms(structure, threshold):
    indexlist = []
    for i in range(len(structure)):
        pos = structure.get_scaled_positions()[i]
        # save indices that are outside the new cell
        if abs(max(pos) > threshold) or min(pos) < -0.01:
            indexlist.append(i)

    return indexlist


def compare_structures(artificial, unrelaxed_rattled, cutoff):
    indexlist = []
    rmindexlist = []
    for i in range(len(unrelaxed_rattled)):
        for j in range(len(artificial)):
            if (abs(max((artificial.get_positions()[j]
                         - unrelaxed_rattled.get_positions()[i]))) < cutoff
               and i not in indexlist):
                indexlist.append(i)
    for i in range(len(unrelaxed_rattled)):
        if i not in indexlist:
            rmindexlist.append(i)
    return rmindexlist


def recreate_symmetric_cell(structure, unrelaxed, primitive, pristine,
                            translation, delta):
    """
    Recreate a symmetric supercell with atomic positions of the general supercell.

    Function that analyses supercell created by the general algorithm and
    creates symmetric supercell with the atomic positions of the general
    supercell.

    Note: The atoms are not correctly mapped in yet, and also the number
    of atoms is not correct here. It is done in the mapping functions.
    """
    reference = primitive.copy()
    N = get_supercell_shape(primitive, pristine)
    reference = reference.repeat((N, N, 1))
    cell = reference.get_cell()
    scell = structure.get_cell()

    # create intermediate big structure for the relaxed structure
    bigatoms_rel = structure.repeat((5, 5, 1))
    positions = bigatoms_rel.get_positions()
    positions += [-translation[0], -translation[1], 0]
    positions += -2.0 * scell[0] - 1.0 * scell[1]
    positions += (0.5 + delta) * cell[0] + (0.5 + delta) * cell[1]
    kinds = bigatoms_rel.get_chemical_symbols()
    rel_struc = Atoms(symbols=kinds, positions=positions, cell=cell)

    # create intermediate big structure for the unrelaxed structure
    bigatoms_rel = unrelaxed.repeat((5, 5, 1))
    positions = bigatoms_rel.get_positions()
    positions += [-translation[0], -translation[1], 0]
    positions += -2.0 * scell[0] - 1.0 * scell[1]
    positions += (0.5 + delta) * cell[0] + (0.5 + delta) * cell[1]
    kinds = bigatoms_rel.get_chemical_symbols()
    ref_struc = Atoms(symbols=kinds, positions=positions, cell=cell)

    refpos = reference.get_positions()
    refpos += [-translation[0], -translation[1], 0]
    refpos += (0.5 + delta) * cell[0] + (0.5 + delta) * cell[1]
    reference.set_positions(refpos)
    reference.wrap()

    return rel_struc, ref_struc, reference, cell, N


def get_supercell_shape(primitive, pristine):
    """
    Calculate which (NxNx1) supercell would be closest to the given supercell.

    Returns: N
    """
    N = len(pristine) / len(primitive)
    N = int(np.floor(np.sqrt(N)))
    reconstruct = primitive.copy()
    reconstruct = reconstruct.repeat((N, N, 1))
    rcell = reconstruct.get_cell()
    pcell = pristine.get_cell()
    if rcell[1, 1] > pcell[1, 1]:
        N -= 1
    return N


def is_vacancy(defectpath):
    """Check whether current defect is a vacancy."""
    try:
        defecttype = str(defectpath.absolute()).split(
            '/')[-2].split('_')[-2].split('.')[-1]
        if defecttype == 'v':
            return True
        else:
            return False
    except IndexError:
        return False


def get_defect_info(primitive, defectpath):
    """Return defecttype, and kind."""
    defecttype = str(defectpath.absolute()).split(
        '/')[-2].split('_')[-2].split('.')[-1]
    defectpos = str(defectpath.absolute()).split(
        '/')[-2].split('_')[-1]

    return defecttype, defectpos


def return_defect_coordinates(structure, unrelaxed, primitive, pristine,
                              defectpath):
    """Return the coordinates of the present defect."""
    deftype, defpos = get_defect_info(primitive, defectpath)
    if not is_vacancy(defectpath):
        for i in range(len(primitive)):
            if not (primitive.get_chemical_symbols()[i]
                    == structure.get_chemical_symbols()[i]):
                label = i
                break
            else:
                label = 0
    elif is_vacancy(defectpath):
        for i in range(len(primitive)):
            if not (primitive.get_chemical_symbols()[i]
                    == structure.get_chemical_symbols()[i]):
                label = i
                break
            else:
                label = 0

    pos = pristine.get_positions()[label]

    return pos


def draw_band_edge(energy, edge, color, offset=2, ax=None):
    if edge == 'vbm':
        eoffset = energy - offset
        elabel = energy - offset / 2
    elif edge == 'cbm':
        eoffset = energy + offset
        elabel = energy + offset / 2

    ax.plot([0, 1], [energy] * 2, color='black', zorder=1)
    ax.fill_between([0, 1], [energy] * 2, [eoffset] * 2, color='grey', alpha=0.5)
    ax.text(0.5, elabel, edge.upper(), color='w', weight='bold', ha='center',
            va='center')


def split(word):
    return [char for char in word]


class Level:
    """Class to draw a single defect state level in the gap."""

    def __init__(self, energy, size=0.05, ax=None):
        self.size = size
        self.energy = energy
        self.ax = ax

    def draw(self, spin, deg, off):
        """Draw the defect state according to spin and degeneracy."""
        xpos_deg = [[1 / 8, 3 / 8], [5 / 8, 7 / 8]]
        xpos_nor = [1 / 4, 3 / 4]
        if deg == 2:
            relpos = xpos_deg[spin][off]
        elif deg == 1:
            relpos = xpos_nor[spin]
        # if deg - 1:
        #     relpos = [[1 / 8, 3 / 8], [5 / 8, 7 / 8]][off][spin]
        # else:
        #     relpos = [[1 / 4, [1 / 8, 3 / 8]],
        #                [3 / 4, [5 / 8, 7 / 8]]][spin][deg - 1]
        pos = [relpos - self.size, relpos + self.size]
        self.relpos = relpos
        self.spin = spin
        self.deg = deg
        self.off = off

        if deg == 1:
            self.ax.plot(pos, [self.energy] * 2, '-k')

        if deg == 2:
            self.ax.plot(pos, [self.energy] * 2, '-k')

    def add_occupation(self, length):
        """Draw an arrow if the defect state if occupied."""
        updown = [1, -1][self.spin]
        self.ax.arrow(self.relpos,
                      self.energy - updown * length / 2,
                      0,
                      updown * length,
                      head_width=0.01,
                      head_length=length / 5, fc='C3', ec='C3')

    def add_label(self, label, static=None):
        """Add symmetry label of the irrep of the point group."""
        shift = self.size / 5
        labelcolor ='C3'
        if static is None:
            labelstr = label.lower()
            splitstr = split(labelstr)
            if len(splitstr) == 2:
                labelstr = f'{splitstr[0]}$_{splitstr[1]}$'
        else:
            labelstr = 'a'
        if (self.off == 0 and self.spin == 0):
            self.ax.text(self.relpos - self.size - shift,
                         self.energy,
                         labelstr,
                         va='center',
                         ha='right',
                         color=labelcolor)
        if (self.off == 0 and self.spin == 1):
            self.ax.text(self.relpos + self.size + shift,
                         self.energy,
                         labelstr,
                         va='center',
                         ha='left',
                         color=labelcolor)
        if (self.off == 1 and self.spin == 0):
            self.ax.text(self.relpos - self.size - shift,
                         self.energy,
                         labelstr,
                         va='center',
                         ha='right',
                         color=labelcolor)
        if (self.off == 1 and self.spin == 1):
            self.ax.text(self.relpos + self.size + shift,
                         self.energy,
                         labelstr,
                         va='center',
                         ha='left',
                         color=labelcolor)


def compute_offset(symmetrydata, evbm, ecbm, ef):
    closelist = []
    energylist_0 = []
    energylist_1 = []
    for sym in symmetrydata:
        if int(sym.spin) == 0:
            energylist_0.append(sym.energy)
        elif int(sym.spin) == 1:
            energylist_1.append(sym.energy)
    energylist_0.sort()
    energylist_1.sort()
    energy = 0
    for i, en in enumerate(energylist_0):
        try:
            if abs(en - energylist_0[i + 4]) < 0.3:
                energy = energylist_0[i + 4]
            else:
                break
        except:
            continue
    abovelist = []
    belowlist = []
    for element in energylist_0:
        if element > ef:
            abovelist.append(element)
        elif element < ef:
            belowlist.append(element)
    if len(abovelist) == 0:
        above = ecbm
    else:
        above = min(abovelist)
    if len(belowlist) == 0:
        below = evbm
    else:
        below = max(belowlist)
    # if ef < ecbm and ef > evbm:
    if energy != 0:
        return energy - evbm
    #     return 0
    # else:
    elif ef < evbm:
        shift = evbm - ef + (above - ef) / 2
        return energy - shift
    elif ef > ecbm:
        shift = ecbm - ef - (ef - below) / 2
        return energy + shift
    else:
        return energy

    # return 0


def plot_gapstates(row, fname):
    from matplotlib import pyplot as plt

    data = row.data.get('results-asr.defect_symmetry.json')
    gsdata = row.data.get('results-asr.gs.json')

    fig, ax = plt.subplots()

    # extract pristine data
    evbm = data.pristine.vbm
    ecbm = data.pristine.cbm
    gap = data.pristine.gap
    eref = row.data.get('results-asr.get_wfs.json')['eref']
    ef = gsdata['efermi'] - eref
    adjust = compute_offset(data.data['symmetries'], evbm, ecbm, ef)
    evbm = evbm + adjust
    ecbm = ecbm + adjust

    # Draw band edges
    draw_band_edge(evbm, 'vbm', 'C0', offset=gap / 5, ax=ax)
    draw_band_edge(ecbm, 'cbm', 'C1', offset=gap / 5, ax=ax)
    # Loop over eigenvalues to draw the level

    degoffset = 0

    if data.symmetries[0].best is None:
        levelflag = False
    else:
        levelflag = True

    # start with first spin channel
    i = 0
    for sym in data.data['symmetries']:
        if int(sym.spin) == 0:
            ene = sym.energy
            if ene < ecbm and ene > evbm:
                spin = int(sym.spin)
                irrep = sym.best
                if levelflag:
                    deg = [1, 2]['E' in irrep]
                else:
                    deg = 1
                    degoffset = 1
                if deg == 2 and i == 0:
                    degoffset = 0
                    i = 1
                elif deg == 2 and i == 1:
                    degoffset = 1
                    i = 0
                lev = Level(ene, ax=ax)
                lev.draw(spin=spin, deg=deg, off=degoffset)
                if ene <= ef:
                    lev.add_occupation(length=gap / 15.)
                if levelflag:
                    lev.add_label(irrep)
                elif not levelflag:
                    lev.add_label(irrep, 'A')
    # start with first spin channel
    i = 0
    for sym in data.data['symmetries']:
        if int(sym.spin) == 1:
            ene = sym.energy
            if ene < ecbm and ene > evbm:
                spin = int(sym.spin)
                irrep = sym.best
                if levelflag:
                    deg = [1, 2]['E' in irrep]
                else:
                    deg = 1
                    degoffset = 1
                if deg == 2 and i == 0:
                    degoffset = 0
                    i = 1
                elif deg == 2 and i == 1:
                    degoffset = 1
                    i = 0
                lev = Level(ene, ax=ax)
                lev.draw(spin=spin, deg=deg, off=degoffset)
                if ene <= ef:
                    lev.add_occupation(length=gap / 15)
                if levelflag:
                    lev.add_label(irrep)
                elif not levelflag:
                    lev.add_label(irrep, 'A')

    ax1 = ax.twinx()
    ax.set_xlim(0, 1)
    ax.set_ylim(evbm - gap / 5, ecbm + gap / 5)
    ax1.set_ylim(evbm - gap / 5, ecbm + gap / 5)
    ax1.plot([0, 1], [ef] * 2, '--k')
    ax1.set_yticks([ef])
    ax1.set_yticklabels([r'E$_\mathrm{F}$'])
    ax.set_xticks([])
    ax.set_ylabel(r'$E-E\mathrm{vac}$ [eV]')

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def check_and_return_input():
    """Check whether folder structure is correct and return input."""
    pristinepath = list(Path('.').glob('../../defects.pristine*'))[0]
    try:
        pris_struc = read(pristinepath / 'structure.json')
    except FileNotFoundError:
        print('ERROR: pristine structure not available!')
    try:
        struc = read('structure.json')
        unrel = read('unrelaxed.json')
    except FileNotFoundError:
        print('ERROR: defect structure(s) not available!')
    try:
        prim_unrel = read('../../unrelaxed.json')
    except FileNotFoundError:
        print('ERROR: primitive unrelaxed structure not available!')

    return struc, unrel, prim_unrel, pris_struc


if __name__ == '__main__':
    main.cli()
